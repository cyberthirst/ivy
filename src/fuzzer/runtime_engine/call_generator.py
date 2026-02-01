from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Deque, List, Optional, Tuple

from vyper.semantics.types.function import ContractFunctionT

from fuzzer.mutator.value_mutator import ValueMutator
from fuzzer.mutator.argument_mutator import ArgumentMutator
from fuzzer.trace_types import CallTrace, Env, Tx


@dataclass
class GeneratedCall:
    contract_address: str
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    func_t: Optional[ContractFunctionT] = None
    sender: Optional[str] = None


CallKey = Tuple[str, str]  # (contract_address, function_name)


@dataclass
class Seed:
    call: GeneratedCall
    score: int = 0
    times_mutated: int = 0
    last_used_step: int = 0


@dataclass
class Corpus:
    seeds_by_func: Dict[CallKey, Deque[Seed]] = field(default_factory=dict)
    max_seeds_per_func: int = 16
    max_seeds_total: int = 512

    def add_seed(self, call: GeneratedCall, score: int, step: int) -> None:
        key = (call.contract_address, call.function_name)
        if key not in self.seeds_by_func:
            self.seeds_by_func[key] = deque(maxlen=self.max_seeds_per_func)

        seed = Seed(call=call, score=score, times_mutated=0, last_used_step=step)
        func_seeds = self.seeds_by_func[key]

        # If at capacity, replace lowest score seed if new seed is better
        if len(func_seeds) >= self.max_seeds_per_func:
            min_seed = min(func_seeds, key=lambda s: s.score)
            if score > min_seed.score:
                func_seeds.remove(min_seed)
                func_seeds.append(seed)
        else:
            func_seeds.append(seed)

        self._enforce_total_limit()

    def _enforce_total_limit(self) -> None:
        total = sum(len(seeds) for seeds in self.seeds_by_func.values())
        while total > self.max_seeds_total:
            # Remove lowest score seed globally
            min_score = float("inf")
            min_key: Optional[CallKey] = None
            min_seed: Optional[Seed] = None
            for key, seeds in self.seeds_by_func.items():
                for seed in seeds:
                    if seed.score < min_score:
                        min_score = seed.score
                        min_key = key
                        min_seed = seed
            if min_key and min_seed:
                self.seeds_by_func[min_key].remove(min_seed)
                if not self.seeds_by_func[min_key]:
                    del self.seeds_by_func[min_key]
            total -= 1

    def get_seed_for_func(self, key: CallKey, rng: random.Random) -> Optional[Seed]:
        if key not in self.seeds_by_func or not self.seeds_by_func[key]:
            return None
        seeds = list(self.seeds_by_func[key])
        # Weight by freshness (lower times_mutated) and score
        weights = [max(1, s.score + 1) / max(1, s.times_mutated + 1) for s in seeds]
        return rng.choices(seeds, weights=weights, k=1)[0]

    def get_any_seed(self, rng: random.Random) -> Optional[Seed]:
        all_seeds = [s for seeds in self.seeds_by_func.values() for s in seeds]
        if not all_seeds:
            return None
        weights = [max(1, s.score + 1) / max(1, s.times_mutated + 1) for s in all_seeds]
        return rng.choices(all_seeds, weights=weights, k=1)[0]

    def has_seeds_for_func(self, key: CallKey) -> bool:
        return key in self.seeds_by_func and len(self.seeds_by_func[key]) > 0

    def total_seeds(self) -> int:
        return sum(len(seeds) for seeds in self.seeds_by_func.values())


class CallGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.value_mutator = ValueMutator(rng)
        self.argument_mutator = ArgumentMutator(rng, self.value_mutator)

    def get_external_functions(
        self, contract: Any
    ) -> List[Tuple[str, ContractFunctionT]]:
        if not hasattr(contract, "compiler_data"):
            return []

        global_ctx = contract.compiler_data.global_ctx
        return [(fn_t.name, fn_t) for fn_t in global_ctx.exposed_functions if fn_t.name]

    def generate_call_for_function(
        self,
        contract_address: str,
        function_name: str,
        func_t: ContractFunctionT,
        sender: Optional[str] = None,
    ) -> GeneratedCall:
        args = []
        for arg in func_t.arguments:
            args.append(self.value_mutator.generate(arg.typ))

        value = 0
        if func_t.is_payable:
            value = self.rng.choice([0, 1, 10**18])

        return GeneratedCall(
            contract_address=contract_address,
            function_name=function_name,
            args=args,
            kwargs={"value": value},
            func_t=func_t,
            sender=sender,
        )

    def mutate_call(self, call: GeneratedCall) -> GeneratedCall:
        if call.func_t is None or not call.args:
            return call

        mutated_args, mutated_value = self.argument_mutator.mutate_function_call(
            call.func_t,
            call.args,
            call.kwargs.get("value", 0),
        )

        return GeneratedCall(
            contract_address=call.contract_address,
            function_name=call.function_name,
            args=mutated_args,
            kwargs={"value": mutated_value},
            func_t=call.func_t,
            sender=call.sender,
        )

    def mutate_single_arg(self, call: GeneratedCall) -> GeneratedCall:
        """Mutate exactly one argument (low edit distance)."""
        if call.func_t is None or not call.args:
            return call

        mutated_args = list(call.args)
        arg_types = call.func_t.argument_types

        # Pick one argument to mutate
        idx = self.rng.randint(0, len(mutated_args) - 1)
        mutated_args[idx] = self.value_mutator.mutate(mutated_args[idx], arg_types[idx])

        # Occasionally also mutate ETH value for payable functions
        mutated_value = call.kwargs.get("value", 0)
        if call.func_t.is_payable and self.rng.random() < 0.2:
            mutated_value = self.value_mutator.mutate_eth_value(
                mutated_value, is_payable=True
            )

        return GeneratedCall(
            contract_address=call.contract_address,
            function_name=call.function_name,
            args=mutated_args,
            kwargs={"value": mutated_value},
            func_t=call.func_t,
            sender=call.sender,
        )

    def mutate_havoc(self, call: GeneratedCall) -> GeneratedCall:
        """Havoc mutation: mutate multiple args (escalation for plateau)."""
        if call.func_t is None or not call.args:
            return call

        mutated_args = list(call.args)
        arg_types = call.func_t.argument_types

        # Mutate 2-3 arguments
        num_to_mutate = min(len(mutated_args), self.rng.randint(2, 3))
        indices = self.rng.sample(range(len(mutated_args)), num_to_mutate)

        for idx in indices:
            if self.rng.random() < 0.3:
                # Regenerate from scratch (boundary generator)
                mutated_args[idx] = self.value_mutator.generate(arg_types[idx])
            else:
                mutated_args[idx] = self.value_mutator.mutate(
                    mutated_args[idx], arg_types[idx]
                )

        # More aggressive ETH value mutation
        mutated_value = call.kwargs.get("value", 0)
        if self.rng.random() < 0.4:
            mutated_value = self.rng.choice([0, 1, 10**18, 2**128 - 1])

        return GeneratedCall(
            contract_address=call.contract_address,
            function_name=call.function_name,
            args=mutated_args,
            kwargs={"value": mutated_value},
            func_t=call.func_t,
            sender=call.sender,
        )

    def mutate_sender(
        self, call: GeneratedCall, sender_pool: List[str]
    ) -> GeneratedCall:
        new_sender = self.rng.choice(sender_pool)
        return GeneratedCall(
            contract_address=call.contract_address,
            function_name=call.function_name,
            args=call.args,
            kwargs=call.kwargs,
            func_t=call.func_t,
            sender=new_sender,
        )

    def call_trace_from_generated(
        self, call: GeneratedCall, to_address: str
    ) -> CallTrace:
        env = None
        if call.sender is not None:
            env = Env(tx=Tx(origin=call.sender))
        return CallTrace(
            output=None,
            call_args={"to": to_address, "value": call.kwargs.get("value", 0)},
            call_succeeded=None,
            env=env,
            python_args={"method": call.function_name, "args": call.args, "kwargs": {}},
            function_name=call.function_name,
        )

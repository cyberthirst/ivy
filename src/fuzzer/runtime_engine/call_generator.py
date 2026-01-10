from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vyper.semantics.types.function import ContractFunctionT

from src.fuzzer.mutator.value_mutator import ValueMutator
from src.fuzzer.mutator.argument_mutator import ArgumentMutator
from src.fuzzer.trace_types import CallTrace


@dataclass
class GeneratedCall:
    contract_address: str
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    func_t: Optional[ContractFunctionT] = None


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
        )

    def call_trace_from_generated(
        self, call: GeneratedCall, to_address: str
    ) -> CallTrace:
        return CallTrace(
            output=None,
            call_args={"to": to_address, "value": call.kwargs.get("value", 0)},
            call_succeeded=None,
            env=None,
            python_args={"method": call.function_name, "args": call.args, "kwargs": {}},
            function_name=call.function_name,
        )

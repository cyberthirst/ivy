"""
Trace mutation utilities for fuzzing.

This module provides functions to mutate call traces including:
- Reordering traces
- Duplicating traces
- Dropping traces
- Mutating function arguments
"""

import random
from typing import List, Union, Optional, Dict, Any
from copy import deepcopy

from .export_utils import (
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
)
from .value_mutator import ValueMutator


class TraceMutator:
    """Mutates call traces for fuzzing."""

    def __init__(
        self, rng: random.Random, value_mutator: Optional[ValueMutator] = None
    ):
        self.rng = rng
        self.value_mutator = value_mutator or ValueMutator(rng)

    def mutate_trace_sequence(
        self,
        traces: List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace]],
        max_traces: int = 12,
        reorder_prob: float = 0.3,
        duplicate_prob: float = 0.2,
        drop_prob: float = 0.2,
        mutate_args_prob: float = 0.5,
    ) -> List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace]]:
        """
        Mutate a sequence of traces.

        Note: Only CallTraces are mutated. SetBalanceTrace and ClearTransientStorageTrace
        are preserved in their original positions.

        Args:
            traces: Original trace sequence
            max_traces: Maximum number of traces in the result
            reorder_prob: Probability of reordering call traces
            duplicate_prob: Probability of duplicating a call trace
            drop_prob: Probability of dropping a call trace
            mutate_args_prob: Probability of mutating call arguments

        Returns:
            Mutated trace sequence
        """
        # Separate call traces from other traces
        call_traces = []
        other_traces_with_index = []

        for i, trace in enumerate(traces):
            if isinstance(trace, CallTrace):
                call_traces.append((i, trace))
            else:
                other_traces_with_index.append((i, trace))

        # If no call traces, return original
        if not call_traces:
            return traces.copy()

        # Extract just the call traces for mutation
        call_trace_list = [trace for _, trace in call_traces]

        # Apply mutations to call traces
        mutated_calls = self._mutate_call_list(
            call_trace_list,
            max_traces - len(other_traces_with_index),  # Leave room for non-call traces
            reorder_prob,
            duplicate_prob,
            drop_prob,
            mutate_args_prob,
        )

        # Reconstruct the full trace list maintaining relative positions
        # Strategy: Place mutated calls in positions relative to other traces
        result = []
        mutated_call_iter = iter(mutated_calls)
        other_trace_iter = iter(other_traces_with_index)

        next_other = next(other_trace_iter, None)
        next_call = next(mutated_call_iter, None)

        # Simple strategy: interleave based on original positions
        for i in range(len(traces) + len(mutated_calls)):
            if next_other and (not next_call or i == next_other[0]):
                result.append(next_other[1])
                next_other = next(other_trace_iter, None)
            elif next_call:
                result.append(next_call)
                next_call = next(mutated_call_iter, None)

        return result[:max_traces]

    def _mutate_call_list(
        self,
        calls: List[CallTrace],
        max_calls: int,
        reorder_prob: float,
        duplicate_prob: float,
        drop_prob: float,
        mutate_args_prob: float,
    ) -> List[CallTrace]:
        """Mutate a list of call traces."""
        mutated = []

        # Start with a copy of the calls
        working_calls = calls.copy()

        # Reorder
        if self.rng.random() < reorder_prob and len(working_calls) > 1:
            self.rng.shuffle(working_calls)

        # Process each call
        for call in working_calls:
            # Drop
            if self.rng.random() < drop_prob:
                continue

            # Mutate arguments
            if self.rng.random() < mutate_args_prob:
                mutated_call = self.mutate_call_args(call)
            else:
                mutated_call = deepcopy(call)

            mutated.append(mutated_call)

            # Duplicate
            if self.rng.random() < duplicate_prob and len(mutated) < max_calls:
                # Mutate the duplicate too
                dup_call = self.mutate_call_args(mutated_call)
                mutated.append(dup_call)

        return mutated[:max_calls]

    def mutate_call_args(self, trace: CallTrace) -> CallTrace:
        """Mutate arguments of a single call trace."""
        # Deep copy to avoid modifying original
        mutated = deepcopy(trace)

        # Mutate python args if available
        if mutated.python_args is not None:
            mutated_python_args = self._mutate_python_args(mutated.python_args)
            mutated.python_args = mutated_python_args

        # Mutate call args (sender, value, etc.)
        if self.rng.random() < 0.3:
            # Change sender to one of the standard test addresses
            senders = [
                "0x0000000000000000000000000000000000000001",
                "0x0000000000000000000000000000000000000002",
                "0x0000000000000000000000000000000000000003",
            ]
            mutated.call_args["sender"] = self.rng.choice(senders)

        if self.rng.random() < 0.3:
            # Mutate value
            value_choices = [0, 1, 10**18, 2**128 - 1]  # 0, 1 wei, 1 ether, max uint128
            mutated.call_args["value"] = self.rng.choice(value_choices)

        return mutated

    def _mutate_python_args(self, python_args: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate python args dict."""
        mutated = deepcopy(python_args)

        # Mutate positional args
        if "args" in mutated and mutated["args"]:
            mutated_args = []
            for arg in mutated["args"]:
                if self.rng.random() < 0.5:
                    # Try to mutate the argument
                    mutated_arg = self._mutate_value(arg)
                    mutated_args.append(mutated_arg)
                else:
                    mutated_args.append(arg)
            mutated["args"] = mutated_args

        # Mutate keyword args (except 'value' which is handled separately)
        if "kwargs" in mutated:
            for key, value in mutated["kwargs"].items():
                if key != "value" and self.rng.random() < 0.5:
                    mutated["kwargs"][key] = self._mutate_value(value)

        return mutated

    def _mutate_value(self, value: Any) -> Any:
        """Mutate a single value based on its type."""
        if isinstance(value, int):
            # Use boundary values
            if value >= 0:
                choices = [0, 1, value - 1, value + 1, 2**256 - 1, 2**128 - 1]
            else:
                choices = [0, -1, value - 1, value + 1, -(2**255), -(2**127)]
            # Filter out invalid values
            choices = [v for v in choices if -(2**255) <= v < 2**256]
            return self.rng.choice(choices)

        elif isinstance(value, str):
            # Possibly an address or hex string
            if value.startswith("0x") and len(value) == 42:
                # Ethereum address - use test addresses
                addresses = [
                    "0x0000000000000000000000000000000000000000",
                    "0x0000000000000000000000000000000000000001",
                    "0x0000000000000000000000000000000000000002",
                    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                ]
                return self.rng.choice(addresses)
            # Otherwise keep the string
            return value

        elif isinstance(value, bool):
            # Flip boolean
            return not value

        elif isinstance(value, list):
            # Mutate list elements
            if not value:
                return value
            mutated_list = value.copy()
            # Mutate one random element
            if mutated_list:
                idx = self.rng.randint(0, len(mutated_list) - 1)
                mutated_list[idx] = self._mutate_value(mutated_list[idx])
            return mutated_list

        else:
            # Unknown type, return as-is
            return value

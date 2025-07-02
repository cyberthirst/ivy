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

from ..export_utils import (
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
)
from .value_mutator import ValueMutator
from .argument_mutator import ArgumentMutator


class TraceMutator:
    """Mutates call traces for fuzzing."""

    def __init__(
        self,
        rng: random.Random,
        value_mutator: Optional[ValueMutator] = None,
        argument_mutator: Optional[ArgumentMutator] = None,
    ):
        self.rng = rng
        self.value_mutator = value_mutator or ValueMutator(rng)
        self.argument_mutator = argument_mutator or ArgumentMutator(
            rng, self.value_mutator
        )

    def mutate_trace_sequence(
        self,
        traces: List[Union[CallTrace, SetBalanceTrace, ClearTransientStorageTrace]],
        max_traces: int = 12,
        reorder_prob: float = 0.3,
        duplicate_prob: float = 0.2,
        drop_prob: float = 0.2,
        mutate_args_prob: float = 0.5,
        deployment_compiler_data: Optional[Dict[str, Any]] = None,
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
            deployment_compiler_data,
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
        deployment_compiler_data: Optional[Dict[str, Any]] = None,
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
                mutated_call = self.mutate_call_args(call, deployment_compiler_data)
            else:
                mutated_call = deepcopy(call)

            mutated.append(mutated_call)

            # Duplicate
            if self.rng.random() < duplicate_prob and len(mutated) < max_calls:
                # Mutate the duplicate too
                dup_call = self.mutate_call_args(mutated_call, deployment_compiler_data)
                mutated.append(dup_call)

        return mutated[:max_calls]

    def mutate_call_args(
        self,
        trace: CallTrace,
        deployment_compiler_data: Optional[Dict[str, Any]] = None,
    ) -> CallTrace:
        """Mutate arguments of a single call trace."""
        # Deep copy to avoid modifying original
        mutated = deepcopy(trace)

        # If we have a function name, we can do type-aware mutation
        if trace.function_name and deployment_compiler_data:
            # Get the contract address
            contract_address = trace.call_args.get("to")
            if contract_address in deployment_compiler_data:
                # Get compiler data and module type
                compiler_data = deployment_compiler_data[contract_address]
                try:
                    module_t = compiler_data.annotated_vyper_module._metadata["type"]

                    # Find the function in exposed functions
                    function = None
                    for func in module_t.exposed_functions:
                        if func.name == trace.function_name:
                            function = func
                            break

                    if function:
                        # Get current args and value
                        current_args = (
                            trace.python_args.get("args", [])
                            if trace.python_args
                            else []
                        )
                        current_value = trace.call_args.get("value", 0)

                        normalized_args = (
                            self.argument_mutator.normalize_arguments_with_types(
                                function.argument_types, current_args
                            )
                        )

                        mutated_args, mutated_value = (
                            self.argument_mutator.mutate_call_args(
                                function, normalized_args, current_value
                            )
                        )

                        # Update the trace
                        if trace.python_args:
                            mutated.python_args = deepcopy(trace.python_args)
                            mutated.python_args["args"] = mutated_args
                        mutated.call_args["value"] = mutated_value
                        return mutated

                except Exception as e:
                    # Log and fall through to no mutation
                    import logging

                    logging.debug(f"Failed to get type info for mutation: {e}")

        # No function name = low-level call, skip mutation for now
        # TODO: Implement ABI decoding for low-level calls to enable type-aware mutation
        # TODO: Or implement byte-level mutations on raw calldata

        # No mutation - return the copy as-is
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

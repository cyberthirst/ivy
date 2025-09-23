"""
Trace mutation utilities for fuzzing.

This module provides functions to mutate:
- Deployment traces (source mutations and constructor args)
- Call trace arguments
"""

import random
from typing import Optional, Dict, Any
from copy import deepcopy

from ..export_utils import (
    CallTrace,
    DeploymentTrace,
)
from .value_mutator import ValueMutator
from .argument_mutator import ArgumentMutator
from .ast_mutator import AstMutator


class TraceMutator:
    """Mutates call traces for fuzzing."""

    def __init__(
        self,
        rng: random.Random,
        value_mutator: Optional[ValueMutator] = None,
        argument_mutator: Optional[ArgumentMutator] = None,
        ast_mutator: Optional[AstMutator] = None,
    ):
        self.rng = rng
        self.value_mutator = value_mutator or ValueMutator(rng)
        self.argument_mutator = argument_mutator or ArgumentMutator(
            rng, self.value_mutator
        )
        self.ast_mutator = ast_mutator or AstMutator(rng)

    # TODO split into 2 funs - mutation and normalization
    def mutate_and_normalize_call_args(
        self,
        trace: CallTrace,
        do_mutation: bool,
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

                    assert function is not None
                    # Get current args and value
                    current_args = (
                        trace.python_args.get("args", []) if trace.python_args else []
                    )
                    current_value = trace.call_args.get("value", 0)

                    normalized_args = (
                        self.argument_mutator.normalize_arguments_with_types(
                            function.argument_types, current_args
                        )
                    )

                    mutated_args, mutated_value = (normalized_args, current_value)
                    if do_mutation:
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

    def mutate_deployment_trace(
        self,
        trace: DeploymentTrace,
        compiler_data: Optional[Any] = None,
    ) -> DeploymentTrace:
        """
        This function can mutate:
        - Source code (using AST mutation)
        - Deployment arguments (constructor args)
        - Deployment value
        """
        if not (
            isinstance(trace, DeploymentTrace) and trace.deployment_type == "source"
        ):
            return trace

        mutated_deployment = None

        if trace.source_code and compiler_data:
            mutation_result = self.ast_mutator.mutate_source_with_compiler_data(
                compiler_data
            )
            if mutation_result and mutation_result.source != trace.source_code:
                mutated_deployment = deepcopy(trace)
                mutated_deployment.source_code = mutation_result.source
                mutated_deployment.compilation_xfails = list(
                    trace.compilation_xfails
                ) + list(mutation_result.compilation_xfails)
                mutated_deployment.runtime_xfails = list(trace.runtime_xfails) + list(
                    mutation_result.runtime_xfails
                )

        if trace.python_args and compiler_data:
            deploy_args = trace.python_args.get("args", [])

            module_t = compiler_data.annotated_vyper_module._metadata["type"]
            init_function = module_t.init_function

            if init_function and deploy_args:
                normalized_args = self.argument_mutator.normalize_arguments_with_types(
                    init_function.argument_types, deploy_args
                )
            else:
                normalized_args = deploy_args

            mutated_args, mutated_value = self.argument_mutator.mutate_deployment_args(
                init_function,
                normalized_args,
                trace.value,
            )

            if not mutated_deployment:
                mutated_deployment = deepcopy(trace)

            mutated_deployment.python_args = deepcopy(trace.python_args)
            mutated_deployment.python_args["args"] = mutated_args
            mutated_deployment.value = mutated_value

        return mutated_deployment if mutated_deployment else trace

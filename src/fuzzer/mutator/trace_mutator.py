"""
Trace mutation utilities for fuzzing.

This module provides functions to mutate:
- Deployment traces (source mutations and constructor args)
- Call trace arguments
"""

import random
from typing import Optional, Dict, Any

from fuzzer.trace_types import CallTrace, DeploymentTrace
from fuzzer.mutator.value_mutator import ValueMutator
from fuzzer.mutator.argument_mutator import ArgumentMutator
from fuzzer.mutator.ast_mutator import AstMutator


class TraceMutator:
    """Mutates call traces for fuzzing."""

    def __init__(
        self,
        rng: random.Random,
        value_mutator: Optional[ValueMutator] = None,
        argument_mutator: Optional[ArgumentMutator] = None,
        *,
        ast_mutator: AstMutator,
    ):
        self.rng = rng
        self.value_mutator = value_mutator or ValueMutator(rng)
        self.argument_mutator = argument_mutator or ArgumentMutator(
            rng, self.value_mutator
        )
        self.ast_mutator = ast_mutator

    def mutate_call_args(
        self,
        trace: CallTrace,
        do_mutation: bool,
        deployment_compiler_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mutate arguments of a single call trace in place."""
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

                    if do_mutation:
                        mutated_args, mutated_value = (
                            self.argument_mutator.mutate_call_args(
                                function, current_args, current_value
                            )
                        )
                        if trace.python_args:
                            trace.python_args["args"] = mutated_args
                        trace.call_args["value"] = mutated_value

                except Exception as e:
                    # Log and fall through to no mutation
                    import logging

                    logging.debug(f"Failed to get type info for mutation: {e}")

        # No function name = low-level call, skip mutation for now
        # TODO: Implement ABI decoding for low-level calls to enable type-aware mutation
        # TODO: Or implement byte-level mutations on raw calldata

    def mutate_deployment_trace(
        self,
        trace: DeploymentTrace,
        compiler_data: Optional[Any] = None,
    ) -> None:
        """
        Mutate deployment trace in place:
        - Source code (using AST mutation)
        - Deployment arguments (constructor args)
        - Deployment value
        """
        if not (
            isinstance(trace, DeploymentTrace) and trace.deployment_type == "source"
        ):
            return

        if trace.solc_json and compiler_data:
            mutation_result = self.ast_mutator.mutate_source_with_compiler_data(
                compiler_data
            )
            if mutation_result:
                sources = trace.solc_json.get("sources")
                if sources is None:
                    sources = {}
                    trace.solc_json["sources"] = sources
                changed = False
                for filename, content in mutation_result.sources.items():
                    entry = sources.get(filename)
                    if isinstance(entry, dict):
                        if entry.get("content") != content:
                            entry["content"] = content
                            changed = True
                    else:
                        if entry != content:
                            sources[filename] = {"content": content}
                            changed = True

                if changed:
                    # TODO: Vyper solc_json may include per-source sha256sum and
                    # top-level integrity metadata. Mutation updates content only,
                    # so hashes can become stale unless recomputed or cleared.
                    trace.compilation_xfails = list(trace.compilation_xfails) + list(
                        mutation_result.compilation_xfails
                    )
                    trace.runtime_xfails = list(trace.runtime_xfails) + list(
                        mutation_result.runtime_xfails
                    )

        if trace.python_args and compiler_data:
            module_t = compiler_data.annotated_vyper_module._metadata["type"]
            init_function = module_t.init_function

            deploy_args = trace.python_args.get("args", [])

            mutated_args, mutated_value = self.argument_mutator.mutate_deployment_args(
                init_function,
                deploy_args,
                trace.value,
            )

            trace.python_args["args"] = mutated_args
            trace.value = mutated_value

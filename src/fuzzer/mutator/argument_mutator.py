"""
Argument mutation utilities for function calls.

This module provides type-aware mutation of function arguments for both
deployment (constructor) and regular function calls.
"""

import random
from typing import List, Any, Optional, Tuple

from fuzzer.mutator.value_mutator import ValueMutator
from vyper.semantics.types.function import ContractFunctionT


class ArgumentMutator:
    """Mutates function arguments based on Vyper type information."""

    def __init__(
        self, rng: random.Random, value_mutator: Optional[ValueMutator] = None
    ):
        """Initialize the argument mutator.

        Args:
            rng: Random number generator for consistent randomness
            value_mutator: Value mutator instance (creates one if not provided)
        """
        self.rng = rng
        self.value_mutator = value_mutator or ValueMutator(rng)

    def mutate_arguments_with_types(
        self,
        arg_types: List[Any],  # VyperType objects
        args: List[Any],
        mutation_prob: float = 0.3,
    ) -> List[Any]:
        """Mutate arguments based on Vyper types.

        Args:
            arg_types: List of Vyper type objects
            args: List of argument values
            mutation_prob: Probability of mutating each argument

        Returns:
            List of potentially mutated arguments
        """
        mutated_args = args.copy()

        for i, (arg_type, arg_value) in enumerate(zip(arg_types, args)):
            if i < len(mutated_args) and self.rng.random() < mutation_prob:
                if self.rng.random() < 0.2:
                    mutated_args[i] = self.value_mutator.mutate(arg_value, arg_type)

        return mutated_args

    def mutate_function_call(
        self,
        function: Optional[ContractFunctionT],
        args: List[Any],
        value: int,
        mutation_prob: float = 0.3,
    ) -> Tuple[List[Any], int]:
        """Unified mutation logic for any function call (deployment or regular).

        This method provides consistent mutation behavior for both deployment
        and regular function calls, ensuring code reuse and maintainability.

        Args:
            function: Vyper function type (None for functions without type info)
            args: Function arguments
            value: ETH value sent with the call
            mutation_prob: Probability of mutating arguments

        Returns:
            Tuple of (mutated_args, mutated_value)
        """
        mutated_args = args.copy()

        # Mutate arguments if we have type information
        if function and args:
            arg_types = function.argument_types
            mutated_args = self.mutate_arguments_with_types(
                arg_types, args, mutation_prob
            )

        # Mutate ETH value based on payability
        mutated_value = self.value_mutator.mutate_eth_value(
            value,
            is_payable=function.is_payable if function else False,
        )

        return mutated_args, mutated_value

    def mutate_deployment_args(
        self,
        init_function: Optional[ContractFunctionT],
        deploy_args: List[Any],
        deploy_value: int,
    ) -> Tuple[List[Any], int]:
        """Mutate deployment arguments using init function type info.

        Args:
            init_function: Contract's init function type (None if no constructor)
            deploy_args: Constructor arguments
            deploy_value: ETH value sent with deployment

        Returns:
            Tuple of (mutated_args, mutated_value)
        """
        return self.mutate_function_call(init_function, deploy_args, deploy_value)

    def mutate_call_args(
        self, function: ContractFunctionT, call_args: List[Any], call_value: int = 0
    ) -> Tuple[List[Any], int]:
        """Mutate call arguments using function type info.

        Args:
            function: Function type information
            call_args: Function arguments
            call_value: ETH value sent with call

        Returns:
            Tuple of (mutated_args, mutated_value)
        """
        return self.mutate_function_call(function, call_args, call_value)

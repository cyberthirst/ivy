"""
Base class for type-dispatched value generation.

Provides the dispatch mechanism for generating values based on Vyper types.
Subclasses implement specific generation strategies (boundary values for ABI
fuzzing, small literals for codegen, etc.).
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from decimal import Decimal

from vyper.semantics.types import (
    AddressT,
    BoolT,
    BytesM_T,
    BytesT,
    DArrayT,
    DecimalT,
    FlagT,
    IntegerT,
    InterfaceT,
    SArrayT,
    StringT,
    StructT,
    TupleT,
    VyperType,
)


class BaseValueGenerator(ABC):
    """Base class for type-dispatched value generation."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

        self._generators = {
            TupleT: self._generate_tuple,
            StructT: self._generate_struct,
            SArrayT: self._generate_sarray,
            DArrayT: self._generate_darray,
            StringT: self._generate_string,
            BytesT: self._generate_bytes,
            IntegerT: self._generate_integer,
            BytesM_T: self._generate_bytes_m,
            BoolT: self._generate_bool,
            AddressT: self._generate_address,
            InterfaceT: self._generate_address,
            DecimalT: self._generate_decimal,
            FlagT: self._generate_flag,
        }

    def generate(self, vyper_type: VyperType) -> Any:
        """Generate a value for the given type."""
        generator = self._generators.get(type(vyper_type))
        assert generator is not None, f"No generator for {type(vyper_type).__name__}"
        return generator(vyper_type)

    # Composite types - shared implementation
    def _generate_tuple(self, vyper_type: TupleT) -> tuple:
        return tuple(self.generate(t) for t in vyper_type.member_types)

    def _generate_struct(self, vyper_type: StructT) -> dict:
        return {
            name: self.generate(field_type)
            for name, field_type in vyper_type.members.items()
        }

    def _generate_sarray(self, vyper_type: SArrayT) -> list:
        return [self.generate(vyper_type.value_type) for _ in range(vyper_type.length)]

    def _generate_darray(self, vyper_type: DArrayT) -> list:
        n = self.rng.randint(0, vyper_type.length)
        return [self.generate(vyper_type.value_type) for _ in range(n)]

    def _generate_bool(self, _vyper_type: BoolT) -> bool:
        return self.rng.choice([True, False])

    # Abstract methods - subclasses define behavior
    @abstractmethod
    def _generate_integer(self, vyper_type: IntegerT) -> int: ...

    @abstractmethod
    def _generate_address(self, vyper_type: AddressT) -> str: ...

    @abstractmethod
    def _generate_bytes_m(self, vyper_type: BytesM_T) -> bytes: ...

    @abstractmethod
    def _generate_bytes(self, vyper_type: BytesT) -> bytes: ...

    @abstractmethod
    def _generate_string(self, vyper_type: StringT) -> str: ...

    @abstractmethod
    def _generate_decimal(self, vyper_type: DecimalT) -> Decimal: ...

    @abstractmethod
    def _generate_flag(self, vyper_type: FlagT) -> int: ...

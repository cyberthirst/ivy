"""
Literal generator for contract codegen.

Generates small, readable literal values for use in generated contracts.
Occasionally delegates to ValueMutator for boundary values.
"""

import random
from typing import Any, Dict, Optional, Type

from decimal import Decimal

from vyper.semantics.types import (
    AddressT,
    BoolT,
    BytesM_T,
    BytesT,
    DArrayT,
    DecimalT,
    IntegerT,
    StringT,
    VyperType,
)

from fuzzer.mutator.base_value_generator import BaseValueGenerator
from fuzzer.mutator.value_mutator import ValueMutator


class LiteralGenerator(BaseValueGenerator):
    """Generates small, readable literals for codegen."""

    # BytesM_T and AddressT always delegate - they have fixed length so
    # there's no benefit to special "small" handling.
    DEFAULT_BOUNDARY_PROBS: Dict[Type, float] = {
        IntegerT: 0.0001,
        StringT: 0.0001,
        BytesT: 0.0001,
        BytesM_T: 1.0,
        AddressT: 1.0,
        # binary outcame, handled in the super class
        BoolT: 0.0,
    }

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        boundary_probs: Optional[Dict[Type, float]] = None,
    ):
        super().__init__(rng)
        self._boundary_probs = boundary_probs or self.DEFAULT_BOUNDARY_PROBS
        self._boundary_generator = ValueMutator(self.rng)

    def generate(self, vyper_type: VyperType) -> Any:
        # Check if we should delegate to boundary generator
        prob = self._boundary_probs.get(type(vyper_type), 0.0)
        if self.rng.random() < prob:
            return self._boundary_generator.generate(vyper_type)

        return super().generate(vyper_type)

    def _generate_integer(self, vyper_type: IntegerT) -> int:
        lo, hi = vyper_type.ast_bounds
        smol = [v for v in [-2, -1, 0, 1, 2] if lo <= v <= hi]
        if self.rng.random() < 0.99:
            return self.rng.choice(smol)
        return self.rng.choice([-10, 10, 100, 255, 256])

    def _generate_address(self, vyper_type: AddressT) -> str:
        # Always delegate to boundary generator
        return self._boundary_generator._generate_address(vyper_type)

    def _generate_string(self, vyper_type: StringT) -> str:
        max_length = vyper_type.length
        # Short readable strings
        choices = ["", "a", "ab", "abc", "test", "hello", "foo", "bar"]
        valid = [s for s in choices if len(s) <= max_length]
        return self.rng.choice(valid)

    def _generate_bytes(self, vyper_type: BytesT) -> bytes:
        max_length = vyper_type.length
        # Short byte sequences
        choices = [b"", b"\x00", b"\xff", b"\x01\x02", b"test"]
        valid = [b for b in choices if len(b) <= max_length]
        return self.rng.choice(valid)

    def _generate_bytes_m(self, vyper_type: BytesM_T) -> bytes:
        # Always delegate to boundary generator
        return self._boundary_generator._generate_bytes_m(vyper_type)

    def _generate_darray(self, vyper_type: DArrayT) -> list:
        # Bias toward small lengths (1-10), very rarely use full random length
        max_len = vyper_type.length
        if self.rng.random() < 0.01:
            n = self.rng.randint(0, max_len)
        else:
            n = self.rng.randint(0, min(10, max_len))
        return [self.generate(vyper_type.value_type) for _ in range(n)]

    def _generate_decimal(self, _vyper_type: DecimalT) -> Decimal:
        return self.rng.choice(
            [
                Decimal("0.0"),
                Decimal("1.0"),
                Decimal("-1.0"),
                Decimal("2.0"),
                Decimal("-2.0"),
                Decimal("0.5"),
                Decimal("-0.5"),
                Decimal("1.5"),
            ]
        )

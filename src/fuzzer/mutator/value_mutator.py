"""
Unified value mutation module for differential fuzzing.

This module provides consistent value generation and mutation
for both AST mutations and ABI fuzzing.
"""

import random
from typing import Any, List, Optional, Union
from vyper.semantics.types import (
    AddressT,
    BoolT,
    BytesM_T,
    BytesT,
    DArrayT,
    IntegerT,
    SArrayT,
    StringT,
    StructT,
    TupleT,
    VyperType,
)


class ValueMutator:
    """Generates and mutates values based on Vyper types."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate_value_for_type(self, vyper_type: VyperType) -> Any:
        """Generate a value for a given Vyper type."""
        if isinstance(vyper_type, TupleT):
            return tuple(
                self.generate_value_for_type(t) for t in vyper_type.member_types
            )

        if isinstance(vyper_type, StructT):
            return tuple(
                self.generate_value_for_type(t) for t in vyper_type.tuple_members()
            )

        if isinstance(vyper_type, SArrayT):
            return [
                self.generate_value_for_type(vyper_type.value_type)
                for _ in range(vyper_type.length)
            ]

        if isinstance(vyper_type, DArrayT):
            n = self.rng.randint(0, vyper_type.length)
            return [
                self.generate_value_for_type(vyper_type.value_type) for _ in range(n)
            ]

        if isinstance(vyper_type, StringT):
            return self._generate_string(vyper_type.length)

        if isinstance(vyper_type, BytesT):
            return self._generate_bytes(vyper_type.length)

        if isinstance(vyper_type, IntegerT):
            return self._generate_integer(vyper_type)

        if isinstance(vyper_type, BytesM_T):
            return self.rng.randbytes(vyper_type.length)

        if isinstance(vyper_type, BoolT):
            return self.rng.choice([True, False])

        if isinstance(vyper_type, AddressT):
            return self._generate_address()

        # Fallback
        return 0

    def mutate_value(self, value: Any, vyper_type: VyperType) -> Any:
        """Mutate a value based on its type."""
        if isinstance(vyper_type, IntegerT):
            return self._mutate_integer(value, vyper_type)

        if isinstance(vyper_type, AddressT):
            return self._mutate_address(value)

        if isinstance(vyper_type, BoolT):
            return not value

        if isinstance(vyper_type, StringT):
            return self._mutate_string(value, vyper_type.length)

        if isinstance(vyper_type, BytesT):
            return self._mutate_bytes(value, vyper_type.length)

        if isinstance(vyper_type, BytesM_T):
            return self._mutate_bytes_m(value, vyper_type.length)

        # For complex types, regenerate
        return self.generate_value_for_type(vyper_type)

    def get_boundary_values(self, vyper_type: VyperType) -> List[Any]:
        """Get boundary values for a type."""
        if isinstance(vyper_type, IntegerT):
            lo, hi = vyper_type.ast_bounds
            bits = vyper_type.bits

            # Generate boundary values programmatically
            values = [lo, hi, 0]

            if lo < 0:  # signed
                values.extend([1, -1, lo + 1, hi - 1])
                # Add powers of 2 boundaries within range
                for i in range(3, bits):
                    pow2 = 2**i
                    if -pow2 >= lo:
                        values.append(-pow2)
                    if pow2 - 1 <= hi:
                        values.append(pow2 - 1)
                    if -pow2 + 1 >= lo:
                        values.append(-pow2 + 1)
            else:  # unsigned
                values.extend([1, 2, hi - 1])
                # Add powers of 2 boundaries
                for i in range(3, bits):
                    pow2 = 2**i
                    if pow2 <= hi:
                        values.append(pow2)
                    if pow2 - 1 <= hi:
                        values.append(pow2 - 1)
                    if pow2 + 1 <= hi:
                        values.append(pow2 + 1)
                # Common byte boundaries
                for i in [8, 16, 32, 64, 128]:
                    if i < bits:
                        boundary = 2**i - 1
                        if boundary <= hi:
                            values.append(boundary)

            # Remove duplicates and values outside bounds
            values = list(set(v for v in values if lo <= v <= hi))
            return values

        elif isinstance(vyper_type, AddressT):
            return [
                "0x0000000000000000000000000000000000000000",
                "0x0000000000000000000000000000000000000001",
                "0x0000000000000000000000000000000000000002",
                "0x0000000000000000000000000000000000000003",
                "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "0xdEaDbEeFdEaDbEeFdEaDbEeFdEaDbEeFdEaDbEeF",
            ]

        elif isinstance(vyper_type, BoolT):
            return [True, False]

        elif isinstance(vyper_type, StringT):
            return [
                "",
                "a",
                "A" * min(100, vyper_type.length),
                "\x00",
                "test",
                "0x1234",
            ]

        elif isinstance(vyper_type, BytesT):
            return [
                b"",
                b"\x00",
                b"\xff",
                b"\x00" * min(32, vyper_type.length),
                b"\xff" * min(32, vyper_type.length),
            ]

        elif isinstance(vyper_type, BytesM_T):
            return [
                b"\x00" * vyper_type.length,
                b"\xff" * vyper_type.length,
                self.rng.randbytes(vyper_type.length),
            ]

        # Default fallback
        return [self.generate_value_for_type(vyper_type)]

    def _generate_integer(self, vyper_type: IntegerT) -> int:
        """Generate an integer value for IntegerT."""
        boundary_values = self.get_boundary_values(vyper_type)

        # 70% chance to use boundary value, 30% random
        if self.rng.random() < 0.7 and boundary_values:
            return self.rng.choice(boundary_values)
        else:
            lo, hi = vyper_type.ast_bounds
            return self.rng.randint(lo, hi)

    def _mutate_integer(self, value: int, vyper_type: IntegerT) -> int:
        """Mutate an integer value."""
        lo, hi = vyper_type.ast_bounds

        mutation_type = self.rng.choice(
            ["boundary", "add_one", "subtract_one", "bit_flip", "random"]
        )

        if mutation_type == "boundary":
            boundary_values = self.get_boundary_values(vyper_type)
            return self.rng.choice(boundary_values) if boundary_values else value
        elif mutation_type == "add_one":
            new_val = value + 1
            return new_val if lo <= new_val <= hi else value
        elif mutation_type == "subtract_one":
            new_val = value - 1
            return new_val if lo <= new_val <= hi else value
        elif mutation_type == "bit_flip":
            bit_position = self.rng.randint(0, vyper_type.bits - 1)
            new_val = value ^ (1 << bit_position)
            # For signed integers, handle two's complement
            if vyper_type.is_signed and new_val >= 2 ** (vyper_type.bits - 1):
                new_val -= 2**vyper_type.bits
            return new_val if lo <= new_val <= hi else value
        else:  # random
            return self.rng.randint(lo, hi)

    def _generate_address(self) -> str:
        """Generate an address value."""
        boundary_addresses = self.get_boundary_values(AddressT())
        if self.rng.random() < 0.7:
            return self.rng.choice(boundary_addresses)
        else:
            return f"0x{self.rng.randbytes(20).hex()}"

    def _mutate_address(self, value: str) -> str:
        """Mutate an address value."""
        if self.rng.random() < 0.5:
            return self.rng.choice(self.get_boundary_values(AddressT()))
        else:
            # Flip some bits in the address
            addr_bytes = bytes.fromhex(value[2:] if value.startswith("0x") else value)
            byte_array = bytearray(addr_bytes)
            # Flip a random byte
            idx = self.rng.randint(0, 19)
            byte_array[idx] ^= self.rng.randint(1, 255)
            return f"0x{byte_array.hex()}"

    def _generate_string(self, max_length: int) -> str:
        """Generate a string value."""
        boundary_values = [
            s
            for s in self.get_boundary_values(StringT(max_length))
            if isinstance(s, str) and len(s) <= max_length
        ]

        if self.rng.random() < 0.5 and boundary_values:
            return self.rng.choice(boundary_values)
        else:
            length = self.rng.randint(0, min(max_length, 100))
            return "".join(
                self.rng.choice(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                for _ in range(length)
            )

    def _mutate_string(self, value: str, max_length: int) -> str:
        """Mutate a string value."""
        boundary_values = [
            s
            for s in self.get_boundary_values(StringT(max_length))
            if isinstance(s, str) and len(s) <= max_length
        ]

        if self.rng.random() < 0.3 and boundary_values:
            return self.rng.choice(boundary_values)

        if not value:
            return "a"

        mutation_type = self.rng.choice(["add_char", "remove_char", "flip_char"])

        if mutation_type == "add_char" and len(value) < max_length:
            pos = self.rng.randint(0, len(value))
            char = self.rng.choice(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            return value[:pos] + char + value[pos:]
        elif mutation_type == "remove_char" and len(value) > 0:
            pos = self.rng.randint(0, len(value) - 1)
            return value[:pos] + value[pos + 1 :]
        elif mutation_type == "flip_char" and len(value) > 0:
            pos = self.rng.randint(0, len(value) - 1)
            chars = list(value)
            chars[pos] = self.rng.choice(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            return "".join(chars)

        return value

    def _generate_bytes(self, max_length: int) -> bytes:
        """Generate a bytes value."""
        boundary_values = [
            b
            for b in self.get_boundary_values(BytesT(max_length))
            if isinstance(b, bytes) and len(b) <= max_length
        ]

        if self.rng.random() < 0.5 and boundary_values:
            return self.rng.choice(boundary_values)
        else:
            length = self.rng.randint(0, min(max_length, 100))
            return self.rng.randbytes(length)

    def _mutate_bytes(self, value: bytes, max_length: int) -> bytes:
        """Mutate a bytes value."""
        boundary_values = [
            b
            for b in self.get_boundary_values(BytesT(max_length))
            if isinstance(b, bytes) and len(b) <= max_length
        ]

        if self.rng.random() < 0.3 and boundary_values:
            return self.rng.choice(boundary_values)

        if not value:
            return b"\x00"

        mutation_type = self.rng.choice(["add_byte", "remove_byte", "flip_byte"])

        byte_array = bytearray(value)

        if mutation_type == "add_byte" and len(byte_array) < max_length:
            pos = self.rng.randint(0, len(byte_array))
            byte_array.insert(pos, self.rng.randint(0, 255))
        elif mutation_type == "remove_byte" and len(byte_array) > 0:
            pos = self.rng.randint(0, len(byte_array) - 1)
            del byte_array[pos]
        elif mutation_type == "flip_byte" and len(byte_array) > 0:
            pos = self.rng.randint(0, len(byte_array) - 1)
            byte_array[pos] ^= self.rng.randint(1, 255)

        return bytes(byte_array)

    def _mutate_bytes_m(self, value: bytes, length: int) -> bytes:
        """Mutate a fixed-size bytes value."""
        if len(value) != length:
            # If wrong size, generate new
            return self.rng.randbytes(length)

        byte_array = bytearray(value)
        # Flip a random byte
        idx = self.rng.randint(0, length - 1)
        byte_array[idx] ^= self.rng.randint(1, 255)

        return bytes(byte_array)

    def mutate_eth_value(self, call_value: int, is_payable: bool) -> int:
        """Mutate ETH value for payable functions."""
        p = self.rng.random()
        vals = [call_value]
        if not is_payable and p < 0.01:
            vals = [1, 10**18, 2**128 - 1]

        if is_payable and self.rng.random() < 0.3:
            vals = [0, 1, 10**18, 2**128 - 1]

        return self.rng.choice(vals)

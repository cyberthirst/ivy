from __future__ import annotations

from typing import Union, Dict, Any, Optional, TypeVar, Generic
import copy
from decimal import Decimal

from eth_utils.address import to_canonical_address, to_checksum_address
from eth_typing.evm import Address as EthAddress

from vyper.semantics.types import (
    VyperType,
    StructT,
    HashMapT,
    FlagT,
    SArrayT,
    DArrayT,
    TupleT,
    IntegerT,
    BytesT,
    BytesM_T,
    StringT,
    AddressT,
    DecimalT,
    BoolT,
)
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.data_locations import DataLocation
from vyper.utils import unsigned_to_signed

from ivy.utils import lrudict, _trunc_div
from ivy.journal import Journal, JournalEntryType


class VyperValue:
    """Base class for all Vyper runtime values with type information."""

    __slots__ = ()

    typ: VyperType


T = TypeVar("T")


def boxed(value: T) -> T:
    """Assert and return a boxed value."""
    assert value is None or isinstance(value, VyperValue), f"Expected VyperValue, got {type(value)}"
    return value


def _assert_compatible_types(left: VyperValue, right: VyperValue, op: str) -> None:
    assert left.typ.compare_type(right.typ) or right.typ.compare_type(left.typ), (
        f"{op} type mismatch: {left.typ} vs {right.typ}"
    )


def _coerce_value(value: VyperValue, typ: VyperType) -> VyperValue:
    from ivy.expr.clamper import box_value

    boxed_value = box_value(value, typ)
    assert isinstance(boxed_value, VyperValue)
    return boxed_value


class VyperInt(VyperValue, int):
    """Boxed integer with type info and bounds validation at construction."""

    def __new__(cls, value: int, typ: IntegerT):
        lo, hi = typ.ast_bounds
        if not (lo <= value <= hi):
            from ivy.exceptions import Revert
            raise Revert(data=b"")
        instance = super().__new__(cls, value)
        instance.typ = typ
        return instance

    def __deepcopy__(self, memo):
        return VyperInt(int(self), self.typ)

    def _mask_to_bits(self, value: int) -> int:
        bits = self.typ.bits
        masked = value & ((1 << bits) - 1)
        if self.typ.is_signed:
            return unsigned_to_signed(masked, bits)
        return masked

    def _require_int_other(self, other: object, op: str) -> int:
        if isinstance(other, VyperInt):
            assert self.typ == other.typ, f"{op} type mismatch: {self.typ} != {other.typ}"
            return int(other)
        if isinstance(other, VyperValue):
            assert False, f"{op} expects integer operands"
        assert isinstance(other, int) and not isinstance(other, bool), (
            f"{op} expects int, got {type(other)}"
        )
        return int(other)


    def __add__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Add")
        return int(self) + other_int

    def __sub__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Sub")
        return int(self) - other_int

    def __mul__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Mul")
        return int(self) * other_int

    def __truediv__(self, other: object) -> int:
        assert False, "Use // for integer division"

    def __floordiv__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Floor division")
        if other_int == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return _trunc_div(int(self), other_int)

    def __mod__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Modulo")
        if other_int == 0:
            raise ValueError("Cannot modulo by 0")
        n = int(self)
        d = other_int
        return n - _trunc_div(n, d) * d

    def __pow__(self, other: object, modulo=None) -> int:
        if modulo is not None:
            right = self._require_int_other(other, "Exponentiation")
            return pow(int(self), right, modulo)

        right = self._require_int_other(other, "Exponentiation")
        if right < 0:
            raise ValueError("Exponentiation by negative number")

        left = int(self)
        if left not in (0, 1, -1) and abs(right) > 256:
            raise ValueError(f"Exponentiation {left} ** {right} too large for the given type")

        return pow(left, right)

    def __lshift__(self, other: object) -> int:
        shift = _require_unsigned_shift_amount(other, "Shift")
        assert self.typ.bits == 256, "Shift only supported for int256/uint256"
        if shift >= self.typ.bits:
            return 0
        return self._mask_to_bits(int(self) << shift)

    def __rshift__(self, other: object) -> int:
        shift = _require_unsigned_shift_amount(other, "Shift")
        assert self.typ.bits == 256, "Shift only supported for int256/uint256"
        return int(self) >> shift

    def __and__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Bitwise and")
        return self._mask_to_bits(int(self) & other_int)

    def __or__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Bitwise or")
        return self._mask_to_bits(int(self) | other_int)

    def __xor__(self, other: object) -> int:
        other_int = self._require_int_other(other, "Bitwise xor")
        return self._mask_to_bits(int(self) ^ other_int)

    def __invert__(self) -> int:
        mask = (1 << self.typ.bits) - 1
        return self._mask_to_bits(mask ^ int(self))

    def __neg__(self) -> int:
        assert self.typ.is_signed, "Unary negation only valid for signed integers"
        return -int(self)

    def __eq__(self, other: object) -> bool:
        other_int = self._require_int_other(other, "Equality")
        return int(self) == other_int

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        other_int = self._require_int_other(other, "Comparison")
        return int(self) < other_int

    def __le__(self, other: object) -> bool:
        other_int = self._require_int_other(other, "Comparison")
        return int(self) <= other_int

    def __gt__(self, other: object) -> bool:
        other_int = self._require_int_other(other, "Comparison")
        return int(self) > other_int

    def __ge__(self, other: object) -> bool:
        other_int = self._require_int_other(other, "Comparison")
        return int(self) >= other_int

    def __hash__(self) -> int:
        return int.__hash__(self)


def _require_unsigned_shift_amount(value: object, op: str) -> int:
    if isinstance(value, VyperInt):
        assert value >= 0, f"{op} expects non-negative shift amount"
        return int(value)
    if isinstance(value, VyperValue):
        assert False, f"{op} expects non-negative shift amount"
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{op} expects int shift amount, got {type(value)}"
    )
    assert value >= 0, f"{op} expects non-negative shift amount"
    return int(value)


class VyperBytes(VyperValue, bytes):
    """Boxed bytes with type info and length validation at construction."""

    def __new__(cls, value: bytes, typ: BytesT):
        if len(value) > typ.length:
            from ivy.exceptions import Revert
            raise Revert(data=b"")
        instance = super().__new__(cls, value)
        instance.typ = typ
        return instance

    def __deepcopy__(self, memo):
        return VyperBytes(bytes(self), self.typ)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperBytes):
            _assert_compatible_types(self, other, "Bytes equality")
            return bytes.__eq__(self, other)
        if isinstance(other, VyperValue):
            assert False, "Bytes equality requires Bytes operands"
        return bytes.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return bytes.__hash__(self)


class VyperString(VyperValue, bytes):
    """Boxed string stored as UTF-8 bytes internally.

    Vyper strings are byte sequences in memory. The length field stores
    the byte count, and operations like len() and slice() work on bytes.
    Since Vyper only allows ASCII in source literals, this is transparent
    for normal usage, but correctly handles non-ASCII bytes that may
    arrive via external calls or convert(Bytes, String).
    """

    def __new__(cls, value: Union[str, bytes], typ: StringT):
        if isinstance(value, str):
            value = value.encode("utf-8")
        if len(value) > typ.length:
            from ivy.exceptions import Revert

            raise Revert(data=b"")
        instance = super().__new__(cls, value)
        instance.typ = typ
        return instance

    def __str__(self) -> str:
        """Decode as UTF-8 for display, using surrogateescape for invalid bytes."""
        return self.decode("utf-8", errors="surrogateescape")

    def __repr__(self) -> str:
        return f"VyperString({str(self)!r})"

    def __deepcopy__(self, memo):
        return VyperString(bytes(self), self.typ)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperString):
            _assert_compatible_types(self, other, "String equality")
            return bytes.__eq__(self, other)
        if isinstance(other, VyperValue):
            assert False, "String equality requires String operands"
        return bytes.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return bytes.__hash__(self)


class VyperBytesM(VyperValue, bytes):
    """Boxed fixed-size bytes with type info and length validation at construction."""

    def __new__(cls, value: bytes, typ: BytesM_T):
        if len(value) != typ.length:
            from ivy.exceptions import Revert
            raise Revert(data=b"")
        instance = super().__new__(cls, value)
        instance.typ = typ
        return instance

    def __deepcopy__(self, memo):
        return VyperBytesM(bytes(self), self.typ)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperBytesM):
            self._require_bytesm_other(other)
            return bytes.__eq__(self, other)
        if isinstance(other, VyperValue):
            assert False, "BytesM equality requires BytesM operands"
        return bytes.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return bytes.__hash__(self)

    def _require_bytesm_other(self, other: object) -> VyperBytesM:
        if not isinstance(other, VyperBytesM):
            raise TypeError("BytesM operations require BytesM operands")
        if other.typ.length != self.typ.length:
            raise ValueError("BytesM length mismatch")
        return other

    def _require_bytes32(self, op: str) -> None:
        if self.typ.length != 32:
            raise ValueError(f"{op} only supported for bytes32")

    def __and__(self, other: object):
        other_bytes = self._require_bytesm_other(other)
        result = bytes(left_byte & right_byte for left_byte, right_byte in zip(self, other_bytes))
        return VyperBytesM(result, self.typ)

    def __or__(self, other: object):
        other_bytes = self._require_bytesm_other(other)
        result = bytes(left_byte | right_byte for left_byte, right_byte in zip(self, other_bytes))
        return VyperBytesM(result, self.typ)

    def __xor__(self, other: object):
        other_bytes = self._require_bytesm_other(other)
        result = bytes(left_byte ^ right_byte for left_byte, right_byte in zip(self, other_bytes))
        return VyperBytesM(result, self.typ)

    def __invert__(self):
        self._require_bytes32("Invert")
        val = int.from_bytes(self, "big")
        mask = (1 << (self.typ.length * 8)) - 1
        inverted = mask ^ val
        return VyperBytesM(inverted.to_bytes(self.typ.length, "big"), self.typ)

    def __lshift__(self, other: VyperInt):
        shift = _require_unsigned_shift_amount(other, "Shift")
        self._require_bytes32("Shift")
        if shift >= 256:
            return VyperBytesM(b"\x00" * self.typ.length, self.typ)
        val = int.from_bytes(self, "big")
        mask = (1 << (self.typ.length * 8)) - 1
        shifted = (val << shift) & mask
        return VyperBytesM(shifted.to_bytes(self.typ.length, "big"), self.typ)

    def __rshift__(self, other: VyperInt):
        shift = _require_unsigned_shift_amount(other, "Shift")
        self._require_bytes32("Shift")
        val = int.from_bytes(self, "big")
        shifted = val >> shift
        return VyperBytesM(shifted.to_bytes(self.typ.length, "big"), self.typ)


# Singleton BoolT instance for VyperBool
_BOOL_T = BoolT()


class VyperBool(VyperValue):
    """Boxed boolean with type info."""

    __slots__ = ("_value", "typ")

    def __init__(self, value: bool):
        self._value = bool(value)
        self.typ = _BOOL_T

    def __bool__(self) -> bool:
        return self._value

    def __int__(self) -> int:
        return int(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperBool):
            return self._value == other._value
        if isinstance(other, bool):
            return self._value == other
        if isinstance(other, VyperValue):
            assert False, "Boolean equality requires bool operands"
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"VyperBool({self._value})"

    def __deepcopy__(self, memo):
        return VyperBool(self._value)


# adapted from titanoboa: https://github.com/vyperlang/titanoboa/blob/bedd49e5a4c1e79a7d12c799e42a23a9dc449395/boa/util/abi.py#L20
# inherit from `str` so that users can compare with regular hex string
# addresses
class Address(VyperValue, str):
    # converting between checksum and canonical addresses is a hotspot;
    # this class contains both and caches recently seen conversions
    __slots__ = ("canonical_address", "typ")
    _cache = lrudict(1024)

    canonical_address: EthAddress

    def __new__(cls, address):
        if isinstance(address, Address):
            return address

        if isinstance(address, bytes):
            if len(address) > 20:
                raise ValueError(f"Invalid address length: {len(address)} bytes.")
            address = address.rjust(20, b"\x00").hex()

        if isinstance(address, int):
            address = hex(address)[2:]
            address = address.rjust(40, "0")

        try:
            return cls._cache[address]
        except KeyError:
            pass

        checksum_address = to_checksum_address(address)
        self = super().__new__(cls, checksum_address)
        self.canonical_address = to_canonical_address(address)
        self.typ = AddressT()
        cls._cache[address] = self
        return self

    def __repr__(self):
        checksum_addr = super().__repr__()
        return f"Address({checksum_addr})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Address):
            return str.__eq__(self, other)
        if isinstance(other, VyperValue):
            assert False, "Address equality requires Address operands"
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return str.__hash__(self)


class Flag(VyperValue):
    def __init__(self, typ: FlagT, source: Union[str, int]):
        self.typ = typ
        self.mask = (1 << len(typ._flag_members)) - 1
        if isinstance(source, str):
            if source not in typ._flag_members:
                raise AttributeError(f"'{typ.name}' flag has no member '{source}'")
            value = 2 ** typ._flag_members[source]
        else:
            assert source >> len(typ._flag_members) == 0
            value = source
        self.value = value & self.mask

    def _require_flag_other(self, other: object, op: str) -> "Flag":
        assert isinstance(other, Flag), f"{op} requires Flag operands"
        assert self.typ == other.typ, f"{op} flag type mismatch: {self.typ} != {other.typ}"
        return other

    def __eq__(self, other):
        other_flag = self._require_flag_other(other, "Flag equality")
        return self.value == other_flag.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __or__(self, other):
        other_flag = self._require_flag_other(other, "Flag or")
        return Flag(self.typ, self.value | other_flag.value)

    def __and__(self, other):
        other_flag = self._require_flag_other(other, "Flag and")
        return Flag(self.typ, self.value & other_flag.value)

    def __xor__(self, other):
        other_flag = self._require_flag_other(other, "Flag xor")
        return Flag(self.typ, self.value ^ other_flag.value)

    def __invert__(self):
        return Flag(self.typ, (~self.value) & self.mask)

    def __contains__(self, other):
        other_flag = self._require_flag_other(other, "Flag membership")
        return (self.value & other_flag.value) != 0

    def __hash__(self):
        return hash(self.value)

    def __deepcopy__(self, _):
        return Flag(self.typ, self.value)


class VyperDecimal(VyperValue):
    """Fixed‑point decimal matching Vyper/EVM 10‑dec semantics."""

    value: int
    PRECISION = 10
    SCALING_FACTOR = 10**PRECISION

    MAX_VALUE = 2**167 - 1
    MIN_VALUE = -(2**167)

    @classmethod
    def max(cls) -> VyperDecimal:
        result = cls(0)
        result.value = cls.MAX_VALUE
        return result

    @classmethod
    def min(cls) -> VyperDecimal:
        result = cls(0)
        result.value = cls.MIN_VALUE
        return result

    def __init__(self, value: Union[Decimal, int, float], *, scaled: bool = False):
        self.typ = DecimalT()
        if not scaled:
            # Convert non-numeric types (e.g., VyperBool) to int first
            if not isinstance(value, (int, float, Decimal)):
                value = int(value)
            self.value = int(value * self.SCALING_FACTOR)
        else:
            assert isinstance(value, int)
            self.value = value

        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError("Decimal value out of bounds")

    def _require_decimal_other(self, other: object, op: str) -> VyperDecimal:
        assert isinstance(other, VyperDecimal), f"{op} requires decimal operands"
        return other

    def __add__(self, other: object) -> VyperDecimal:
        other_dec = self._require_decimal_other(other, "Decimal add")
        return VyperDecimal(self.value + other_dec.value, scaled=True)

    def __sub__(self, other: object) -> VyperDecimal:
        other_dec = self._require_decimal_other(other, "Decimal sub")
        return VyperDecimal(self.value - other_dec.value, scaled=True)

    def __mul__(self, other: object) -> VyperDecimal:
        other_dec = self._require_decimal_other(other, "Decimal mul")
        product = self.value * other_dec.value
        scaled = _trunc_div(product, self.SCALING_FACTOR)

        if not (self.MIN_VALUE <= scaled <= self.MAX_VALUE):
            raise ValueError("Decimal multiplication overflow")

        return VyperDecimal(scaled, scaled=True)

    def __truediv__(self, other: object) -> VyperDecimal:
        other_dec = self._require_decimal_other(other, "Decimal div")
        if other_dec.value == 0:
            raise ZeroDivisionError("Division by zero")

        scaled = _trunc_div(self.value * self.SCALING_FACTOR, other_dec.value)

        if not (self.MIN_VALUE <= scaled <= self.MAX_VALUE):
            raise ValueError("Decimal division overflow")

        return VyperDecimal(scaled, scaled=True)

    def __floordiv__(self, other: object) -> VyperDecimal:
        assert False, "Floor division is invalid for decimal types"

    def __mod__(self, other: object) -> VyperDecimal:
        other_dec = self._require_decimal_other(other, "Decimal mod")
        if other_dec.value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")

        rem = self.value - _trunc_div(self.value, other_dec.value) * other_dec.value
        return VyperDecimal(rem, scaled=True)

    def __neg__(self) -> VyperDecimal:
        return VyperDecimal(-self.value, scaled=True)

    def __lt__(self, other: object) -> bool:
        other_dec = self._require_decimal_other(other, "Decimal comparison")
        return self.value < other_dec.value

    def __le__(self, other: object) -> bool:
        other_dec = self._require_decimal_other(other, "Decimal comparison")
        return self.value <= other_dec.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperDecimal):
            return self.value == other.value

        if isinstance(other, Decimal):
            self_as_decimal = Decimal(str(self))
            return self_as_decimal == other

        if isinstance(other, VyperValue):
            assert False, "Decimal equality requires decimal operands"

        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __ge__(self, other: object) -> bool:
        other_dec = self._require_decimal_other(other, "Decimal comparison")
        return self.value >= other_dec.value

    def __gt__(self, other: object) -> bool:
        other_dec = self._require_decimal_other(other, "Decimal comparison")
        return self.value > other_dec.value

    def __hash__(self) -> int:
        return hash(Decimal(str(self)))

    def truncate(self) -> VyperDecimal:
        truncated_int = (
            _trunc_div(self.value, self.SCALING_FACTOR) * self.SCALING_FACTOR
        )
        return VyperDecimal(truncated_int, scaled=True)

    def __str__(self) -> str:
        neg = self.value < 0
        s = f"{abs(self.value):0{self.PRECISION + 1}d}"
        int_part, dec_part = s[: -self.PRECISION] or "0", s[-self.PRECISION :]
        return f"{'-' if neg else ''}{int_part}.{dec_part}"

    def __repr__(self) -> str:
        return f"Decimal('{self}')"


T = TypeVar("T")


class _Container(VyperValue):
    def __init__(self, vyper_type: VyperType):
        self.typ: VyperType = vyper_type
        self._values: Dict[Any, Any] = {}

    def _journal(self, key: Any, loc: Optional[DataLocation] = None):
        if not (loc and Journal().journalable_loc(loc)):
            return

        if key in self._values:
            old_value = self._values[key]
        else:
            old_value = None

        Journal().record(JournalEntryType.STORAGE, self, key, old_value)

    def __deepcopy__(self, _):
        result = self.__class__.__new__(self.__class__)
        result.typ = self.typ  # Types are immutable
        result._values = {k: copy.deepcopy(v) for k, v in self._values.items()}
        return result


class _Sequence(_Container, Generic[T]):
    def __init__(self, typ: _SequenceT):
        assert isinstance(typ, _SequenceT)
        super().__init__(typ)
        self.value_type = typ.value_type
        self.capacity = typ.length
        self.length = 0

    def _normalize_index(self, idx: int) -> int:
        if isinstance(idx, VyperValue) and not isinstance(idx, VyperInt):
            assert False, f"Sequence index must be integer, got {type(idx)}"
        assert isinstance(idx, int) and not isinstance(idx, bool), (
            f"Sequence index must be integer, got {type(idx)}"
        )
        return int(idx)

    def _raise_index_error(self, idx: int):
        raise IndexError(
            f"Sequence index out of range: {idx} not in [0, {self.length})"
        )

    def __getitem__(self, idx: int) -> T:
        idx = self._normalize_index(idx)
        if idx >= self.length or idx < 0:
            self._raise_index_error(idx)
        # TODO: should we journal None or default value?
        # now we journal None, hence we need to check for None
        if idx not in self._values or self._values[idx] is None:
            from ivy.expr.default_values import get_default_value

            self._values[idx] = get_default_value(self.value_type)
        return self._values[idx]

    def __setitem__(self, idx: int, value: T, loc: Optional[DataLocation] = None):
        assert isinstance(value, VyperValue)
        idx = self._normalize_index(idx)
        if idx >= self.length or idx < 0:
            self._raise_index_error(idx)

        self._journal(idx, loc)

        self._values[idx] = _coerce_value(value, self.value_type)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        index = 0
        while True:
            try:
                value = self[index]
                yield value
                index += 1
            except IndexError:
                break

    def __contains__(self, item: object) -> bool:
        assert isinstance(item, VyperValue), "Membership requires a VyperValue operand"
        assert self.value_type.compare_type(item.typ), (
            f"Membership type mismatch: {item.typ} not compatible with {self.value_type}"
        )
        return any(item == value for value in self)

    def __str__(self):
        values = [str(self[i]) for i in range(self.length)]
        return f"[{', '.join(values)}]"

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.value_type = self.value_type
        result.length = self.length
        result.capacity = self.capacity
        return result


class StaticArray(_Sequence[T]):
    def __init__(self, typ: SArrayT, values: Optional[Dict[int, T]] = None):
        super().__init__(typ)
        self.length = self.capacity
        if values:
            self._values = values
            assert len(values) == self.capacity

    def append(self, value: T):
        raise AttributeError("'StaticArray' object has no attribute 'append'")

    def pop(self) -> T:
        raise AttributeError("'StaticArray' object has no attribute 'pop'")

    # Vyper doesn't allow calling `len` on StaticArrays
    # further, the length can exceed `sys.maxsize` - if that happens
    # the Python runtime throws
    def __len__(self) -> int:
        raise AttributeError("'StaticArray' object has no attribute 'len'")


class DynamicArray(_Sequence[T]):
    def __init__(self, typ: DArrayT, values: Optional[Dict[int, T]] = None):
        super().__init__(typ)
        self.length = 0
        if values:
            self._values = values
            assert len(values) <= self.capacity
            self.length = len(values)

    def __len__(self) -> int:
        return self.length

    def _journal_length(self, loc: Optional[DataLocation] = None):
        if not (loc and Journal().journalable_loc(loc)):
            return
        Journal().record(JournalEntryType.ARRAY_LENGTH, self, None, self.length)

    def append(self, value: T, loc: Optional[DataLocation] = None):
        assert isinstance(value, VyperValue)
        if len(self) >= self.capacity:
            raise ValueError(f"Cannot exceed maximum length {self.capacity}")

        self._journal_length(loc)
        idx = self.length
        self._journal(idx, loc)
        self._values[idx] = _coerce_value(value, self.value_type)
        self.length += 1

    def pop(self, loc: Optional[DataLocation] = None) -> T:
        if not self.length:
            raise IndexError("pop from empty array")

        self._journal_length(loc)
        idx = self.length - 1
        self._journal(idx, loc)
        self.length -= 1

        value = self._values.pop(idx)
        return value

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.length = self.length
        return result


class Map(_Container):
    def __init__(self, typ: HashMapT):
        super().__init__(typ)
        self.value_type = typ.value_type
        self.key_type = typ.key_type

    def __getitem__(self, key):
        if key not in self._values or self._values[key] is None:
            from ivy.expr.default_values import get_default_value

            self._values[key] = get_default_value(self.value_type)
        return self._values[key]

    def __setitem__(self, key, value, loc: Optional[DataLocation] = None):
        assert isinstance(value, VyperValue)
        self._journal(key, loc)
        self._values[key] = value

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.value_type = self.value_type
        result.key_type = self.key_type
        return result


class Struct(_Container):
    typ: StructT

    def __init__(
        self,
        typ: StructT,
        kws: Dict[str, Any],
    ):
        super().__init__(typ)
        self._values = {key: value for key, value in kws.items()}

    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        return self._values[key]

    def __setitem__(self, key: str, value: Any, loc: Optional[DataLocation] = None):
        assert isinstance(value, VyperValue)
        if key not in self._values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        self._journal(key, loc)
        self._values[key] = value

    def values(self):
        values = [self._values[k] for k, _ in self.typ.members.items()]
        return values

    def __str__(self):
        items = [f"{k}={str(self[k])}" for k in self.typ.members.keys()]
        return f"{self.typ.name}({', '.join(items)})"

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.typ = self.typ
        return result


class Tuple(_Container):
    def __init__(self, typ: TupleT):
        super().__init__(typ)
        assert isinstance(typ, TupleT)
        # Eagerly initialize all members with their default values
        from ivy.expr.default_values import get_default_value

        self.length = len(typ.member_types)
        self._values = {
            i: get_default_value(member_typ)
            for i, member_typ in enumerate(typ.member_types)
        }

    def __len__(self) -> int:
        return self.length

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Tuple index out of range: {idx} not in [0, {self.length})"
            )

    def __getitem__(self, idx: int):
        self._validate_index(idx)
        return self._values[idx]

    def __setitem__(self, idx: int, value: Any, loc: Optional[DataLocation] = None):
        assert isinstance(value, VyperValue)
        self._validate_index(idx)
        self._journal(idx, loc)
        self._values[idx] = value

    def __iter__(self):
        for i in range(self.length):
            yield self._values[i]

    def __str__(self):
        values = [str(self._values[i]) for i in range(self.length)]
        return f"({', '.join(values)})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tuple):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(self.length))
        if isinstance(other, tuple):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(self.length))
        return NotImplemented

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.length = self.length
        return result

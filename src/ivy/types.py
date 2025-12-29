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
)
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.data_locations import DataLocation

from ivy.utils import lrudict, _trunc_div
from ivy.journal import Journal, JournalEntryType


# adapted from titanoboa: https://github.com/vyperlang/titanoboa/blob/bedd49e5a4c1e79a7d12c799e42a23a9dc449395/boa/util/abi.py#L20
# inherit from `str` so that users can compare with regular hex string
# addresses
class Address(str):
    # converting between checksum and canonical addresses is a hotspot;
    # this class contains both and caches recently seen conversions
    __slots__ = ("canonical_address",)
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
        cls._cache[address] = self
        return self

    def __repr__(self):
        checksum_addr = super().__repr__()
        return f"Address({checksum_addr})"


class Flag:
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

    def __eq__(self, other):
        return self.value == other.value

    def __or__(self, other):
        return Flag(self.typ, self.value | other.value)

    def __and__(self, other):
        return Flag(self.typ, self.value & other.value)

    def __xor__(self, other):
        return Flag(self.typ, self.value ^ other.value)

    def __invert__(self):
        return Flag(self.typ, (~self.value) & self.mask)

    def __contains__(self, other):
        return (self.value & other.value) != 0

    def __hash__(self):
        return hash(self.value)


class VyperDecimal:
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
        if not scaled:
            self.value = int(value * self.SCALING_FACTOR)
        else:
            assert isinstance(value, int)
            self.value = value

        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError("Decimal value out of bounds")

    def __add__(self, other: VyperDecimal) -> VyperDecimal:
        return VyperDecimal(self.value + other.value, scaled=True)

    def __sub__(self, other: VyperDecimal) -> VyperDecimal:
        return VyperDecimal(self.value - other.value, scaled=True)

    def __mul__(self, other: VyperDecimal) -> VyperDecimal:
        product = self.value * other.value
        scaled = _trunc_div(product, self.SCALING_FACTOR)

        if not (self.MIN_VALUE <= scaled <= self.MAX_VALUE):
            raise ValueError("Decimal multiplication overflow")

        return VyperDecimal(scaled, scaled=True)

    def __truediv__(self, other: VyperDecimal) -> VyperDecimal:
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")

        scaled = _trunc_div(self.value * self.SCALING_FACTOR, other.value)

        if not (self.MIN_VALUE <= scaled <= self.MAX_VALUE):
            raise ValueError("Decimal division overflow")

        return VyperDecimal(scaled, scaled=True)

    def __floordiv__(self, other: VyperDecimal) -> VyperDecimal:
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")

        q = _trunc_div(self.value * self.SCALING_FACTOR, other.value)
        q = (q // self.SCALING_FACTOR) * self.SCALING_FACTOR  # drop frac part

        if not (self.MIN_VALUE <= q <= self.MAX_VALUE):
            raise ValueError("Decimal floor‑division overflow")

        return VyperDecimal(q, scaled=True)

    def __mod__(self, other: VyperDecimal) -> VyperDecimal:
        if other.value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")

        rem = self.value - _trunc_div(self.value, other.value) * other.value
        return VyperDecimal(rem, scaled=True)

    def __neg__(self) -> VyperDecimal:
        return VyperDecimal(-self.value, scaled=True)

    def __lt__(self, other: VyperDecimal) -> bool:
        return self.value < other.value

    def __le__(self, other: VyperDecimal) -> bool:
        return self.value <= other.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VyperDecimal):
            return self.value == other.value

        if isinstance(other, Decimal):
            self_as_decimal = Decimal(str(self))
            return self_as_decimal == other

        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __ge__(self, other: VyperDecimal) -> bool:
        return self.value >= other.value

    def __gt__(self, other: VyperDecimal) -> bool:
        return self.value > other.value

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


class _Container:
    def __init__(self, vyper_type: VyperType):
        self._typ: VyperType = vyper_type
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
        result._typ = self._typ  # Types are immutable
        result._values = {k: copy.deepcopy(v) for k, v in self._values.items()}
        return result


class _Sequence(_Container, Generic[T]):
    def __init__(self, typ: _SequenceT):
        assert isinstance(typ, _SequenceT)
        super().__init__(typ)
        self.value_type = typ.value_type
        self.capacity = typ.length
        self.length = 0

    def _raise_index_error(self, idx: int):
        raise IndexError(
            f"Sequence index out of range: {idx} not in [0, {self.length})"
        )

    def __getitem__(self, idx: int) -> T:
        if idx >= self.length or idx < 0:
            self._raise_index_error(idx)
        # TODO: should we journal None or default value?
        # now we journal None, hence we need to check for None
        if idx not in self._values or self._values[idx] is None:
            from ivy.expr.default_values import get_default_value

            self._values[idx] = get_default_value(self.value_type)
        return self._values[idx]

    def __setitem__(self, idx: int, value: T, loc: Optional[DataLocation] = None):
        if idx >= self.length or idx < 0:
            self._raise_index_error(idx)

        self._journal(idx, loc)

        self._values[idx] = value

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
        if len(self) >= self.capacity:
            raise ValueError(f"Cannot exceed maximum length {self.capacity}")

        self._journal_length(loc)
        idx = self.length
        self._journal(idx, loc)
        self._values[idx] = value
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
    _typ: StructT

    def __init__(
        self,
        typ: StructT,
        kws: Dict[str, Any],
    ):
        super().__init__(typ)
        self._values = {key: value for key, value in kws.items()}

    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"'{self._typ.name}' struct has no member '{key}'")
        return self._values[key]

    def __setitem__(self, key: str, value: Any, loc: Optional[DataLocation] = None):
        if key not in self._values:
            raise KeyError(f"'{self._typ.name}' struct has no member '{key}'")
        self._journal(key, loc)
        self._values[key] = value

    def values(self):
        values = [self._values[k] for k, _ in self._typ.members.items()]
        return values

    def __str__(self):
        items = [f"{k}={str(self[k])}" for k in self._typ.members.keys()]
        return f"{self._typ.name}({', '.join(items)})"

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result._typ = self._typ
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

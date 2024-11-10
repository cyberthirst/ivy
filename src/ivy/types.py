from typing import Union, Dict, Any, Optional, TypeVar, Generic
import copy

from eth_utils import to_canonical_address, to_checksum_address
from eth_typing.evm import Address as EthAddress

from vyper.semantics.types import (
    VyperType,
    StructT,
    HashMapT,
    FlagT,
    SArrayT,
    DArrayT,
)
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.data_locations import DataLocation

from ivy.utils import lrudict
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
    PRECISION = 10
    SCALING_FACTOR = 10**PRECISION

    MAX_VALUE = 2**167 - 1
    MIN_VALUE = -(2**167)

    @classmethod
    def max(cls) -> "VyperDecimal":
        result = cls(0)
        result.value = cls.MAX_VALUE
        return result

    @classmethod
    def min(cls) -> "VyperDecimal":
        result = cls(0)
        result.value = cls.MIN_VALUE
        return result

    def __init__(self, value: Union[str, int, float]):
        self.value = int(value * self.SCALING_FACTOR)

        if self.value > self.MAX_VALUE or self.value < self.MIN_VALUE:
            raise ValueError("Decimal value out of bounds")

    def __add__(self, other: "VyperDecimal") -> "VyperDecimal":
        result = VyperDecimal(0)
        result.value = self.value + other.value
        return result

    def __sub__(self, other: "VyperDecimal") -> "VyperDecimal":
        result = VyperDecimal(0)
        result.value = self.value - other.value
        return result

    def __mul__(self, other: "VyperDecimal") -> "VyperDecimal":
        result = VyperDecimal(0)
        result.value = self.value * other.value // self.SCALING_FACTOR
        return result

    def __truediv__(self, other: "VyperDecimal") -> "VyperDecimal":
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")
        result = VyperDecimal("0")
        result.value = (self.value * self.SCALING_FACTOR) // other.value
        return result

    def __floordiv__(self, other: "VyperDecimal") -> "VyperDecimal":
        if other.value == 0:
            raise ZeroDivisionError("Division by zero")
        result = VyperDecimal(0)
        result.value = (self.value * self.SCALING_FACTOR) // other.value
        result.value = (result.value // self.SCALING_FACTOR) * self.SCALING_FACTOR
        return result

    def __lt__(self, other: "VyperDecimal") -> bool:
        return self.value < other.value

    def __le__(self, other: "VyperDecimal") -> bool:
        return self.value <= other.value

    def __eq__(self, other: "VyperDecimal") -> bool:
        return self.value == other.value

    def __ne__(self, other: "VyperDecimal") -> bool:
        return self.value != other.value

    def __ge__(self, other: "VyperDecimal") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "VyperDecimal") -> bool:
        return self.value > other.value

    def __str__(self) -> str:
        is_negative = self.value < 0
        abs_value = abs(self.value)

        str_value = str(abs_value).zfill(self.PRECISION + 1)
        int_part = str_value[: -self.PRECISION] or "0"
        dec_part = str_value[-self.PRECISION :]

        return f"{'-' if is_negative else ''}{int_part}.{dec_part}"

    def __repr__(self) -> str:
        return f"Decimal('{self.__str__()}')"


T = TypeVar("T")


class _Container:
    def __init__(self, vyper_type: VyperType):
        self._typ = vyper_type
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
        self.length = typ.length

    def _raise_index_error(self, idx: int):
        raise IndexError(f"Sequence index out of range: {idx} not in [0, {len(self)})")

    def __getitem__(self, idx: int) -> T:
        if idx >= len(self) or idx < 0:
            self._raise_index_error(idx)
        # TODO: should we journal None or default value?
        # now we journal None, hence we need to check for None
        if idx not in self._values or self._values[idx] is None:
            from ivy.evaluator import VyperEvaluator

            self._values[idx] = VyperEvaluator.default_value(self.value_type)
        return self._values[idx]

    def __setitem__(self, idx: int, value: T, loc: Optional[DataLocation] = None):
        if idx >= len(self) or idx < 0:
            self._raise_index_error(idx)

        self._journal(idx, loc)

        self._values[idx] = value

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        values = [str(self[i]) for i in range(len(self))]
        return f"[{', '.join(values)}]"

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result.value_type = self.value_type
        result.length = self.length
        return result


class StaticArray(_Sequence[T]):
    def __init__(self, typ: SArrayT, values: Optional[Dict[int, T]] = None):
        super().__init__(typ)
        if values:
            self._values = values
            assert len(values) == self.length

    def append(self, value: T):
        raise AttributeError("'StaticArray' object has no attribute 'append'")

    def pop(self) -> T:
        raise AttributeError("'StaticArray' object has no attribute 'pop'")


class DynamicArray(_Sequence[T]):
    def __init__(self, typ: DArrayT, values: Optional[Dict[int, T]] = None):
        super().__init__(typ)
        self._length = 0
        if values:
            self._values = values
            assert len(values) <= self.length
            self._length = len(values)

    def __len__(self) -> int:
        return self._length

    def append(self, value: T):
        if len(self) >= self.length:
            raise ValueError(f"Cannot exceed maximum length {self.length}")

        # TODO journal the length change

        idx = self._length
        self._values[idx] = value
        self._length += 1

    def pop(self) -> T:
        if not self._length:
            raise IndexError("pop from empty array")

        # TODO journal the length change

        idx = self._length - 1
        self._length -= 1

        value = self._values.pop(idx)
        return value

    def __deepcopy__(self, _):
        result = super().__deepcopy__(_)
        result._length = self._length
        return result


class Map(_Container):
    def __init__(self, typ: HashMapT):
        super().__init__(typ)
        self.value_type = typ.value_type
        self.key_type = typ.key_type

    def __getitem__(self, key):
        if key not in self._values or self._values[key] is None:
            from ivy.evaluator import VyperEvaluator

            self._values[key] = VyperEvaluator.default_value(self.value_type)
        return self._values[key]

    def __setitem__(self, key, value, loc: Optional[DataLocation] = None):
        self._journal(key, loc)
        self._values[key] = value


class Struct(_Container):
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
        result.typ = self._typ
        return result

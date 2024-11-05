from typing import Union, Dict, Any, Optional, TypeVar, Generic

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


T = TypeVar("T")


class _Container:
    def __init__(self, vyper_type: VyperType):
        self._typ = vyper_type
        self._values: Dict[Any, Any] = {}

    def _journal(self, key: Any, old_value: Any, loc: Optional[DataLocation] = None):
        if loc and Journal().journalable_loc(loc):
            Journal().record(JournalEntryType.STORAGE, self, key, old_value)


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
        if idx not in self._values:
            from ivy.evaluator import VyperEvaluator

            self._values[idx] = VyperEvaluator.default_value(self.value_type)
        return self._values[idx]

    def __setitem__(self, idx: int, value: T, loc: Optional[DataLocation] = None):
        if idx >= len(self) or idx < 0:
            self._raise_index_error(idx)

        if idx in self._values:
            self._journal(idx, self._values[idx], loc)
        else:
            self._journal(idx, None, loc)

        self._values[idx] = value

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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


class Map(_Container):
    def __init__(self, typ: HashMapT):
        super().__init__(typ)
        self.value_type = typ.value_type
        self.key_type = typ.key_type

    def __getitem__(self, key):
        if key not in self._values:
            from ivy.evaluator import VyperEvaluator

            self._values[key] = VyperEvaluator.default_value(self.value_type)
        return self._values[key]

    def __setitem__(self, key, value, loc: Optional[DataLocation] = None):
        if key in self._values:
            self._journal(key, self._values[key], loc)
        else:
            self._journal(key, None, loc)
        self._values[key] = value


class Struct(_Container):
    def __init__(
        self,
        typ: StructT,
        kws: Dict[str, Any],
    ):
        super().__init__(typ)
        self.typ = typ
        self._values = {key: value for key, value in kws.items()}

    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        return self._values[key]

    def __setitem__(self, key: str, value: Any, loc: Optional[DataLocation] = None):
        if key not in self._values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        self._journal(key, self._values[key], loc)
        self._values[key] = value

    def values(self):
        values = [self._values[k] for k, _ in self.typ.members.items()]
        return values

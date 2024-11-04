from typing import Any, Dict, Generic, Optional, TypeVar

from vyper.semantics.types import VyperType, StructT, HashMapT
from vyper.semantics.types.subscriptable import _SequenceT
from vyper.semantics.data_locations import DataLocation

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType

T = TypeVar("T")


class Container:
    """Base class for all container types"""

    def __init__(
        self, vyper_type: VyperType, data_location: Optional[DataLocation] = None
    ):
        self.type = vyper_type
        self.data_location = data_location
        self.values: Dict[Any, Any] = {}

    def _should_journal(self) -> bool:
        return self.data_location is not None and Journal.journalable_loc(
            self.data_location
        )

    def _journal(self, key: Any, old_value: Any):
        """Journal a change if in journalable location"""
        if self._should_journal():
            Journal().record(JournalEntryType.STORAGE, self, key, old_value)


class Sequence(Container, Generic[T]):
    def __init__(
        self, value_type: VyperType, data_location: Optional[DataLocation] = None
    ):
        super().__init__(value_type, data_location)
        self.value_type = value_type

    def __getitem__(self, idx: int) -> T:
        if idx >= len(self):
            raise IndexError("sequence index out of range")
        if idx not in self.values:
            self.values[idx] = VyperEvaluator.default_value(self.value_type)
        return self.values[idx]

    def __setitem__(self, idx: int, value: T):
        if idx >= len(self):
            raise IndexError("sequence index out of range")

        if idx in self.values:
            self._journal(idx, self.values[idx])
        else:
            self._journal(idx, None)

        self.values[idx] = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class StaticArray(Sequence[T]):
    def __init__(
        self,
        length: int,
        value_type: VyperType,
        data_location: Optional[DataLocation] = None,
    ):
        super().__init__(value_type, data_location)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def append(self, value: T):
        raise AttributeError("'StaticArray' object has no attribute 'append'")

    def pop(self) -> T:
        raise AttributeError("'StaticArray' object has no attribute 'pop'")


class DynamicArray(Sequence[T]):
    def __init__(
        self,
        value_type: VyperType,
        data_location: Optional[DataLocation] = None,
        max_length: Optional[int] = None,
    ):
        super().__init__(value_type, data_location)
        self.max_length = max_length
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def append(self, value: T):
        if self.max_length and len(self) >= self.max_length:
            raise ValueError(f"Cannot exceed maximum length {self.max_length}")

        self._journal("length", self._length)

        idx = self._length
        self.values[idx] = value
        self._length += 1

    def pop(self) -> T:
        if not self._length:
            raise IndexError("pop from empty array")

        self._journal("length", self._length)

        idx = self._length - 1
        self._length -= 1

        value = self.values.pop(idx, VyperEvaluator.default_value(self.value_type))
        return value


class Map(Container, dict):
    def __init__(
        self,
        key_type: VyperType,
        value_type: VyperType,
        data_location: Optional[DataLocation] = None,
    ):
        Container.__init__(self, value_type, data_location)
        dict.__init__(self)
        self.value_type = value_type
        self.key_type = key_type

    def __getitem__(self, key):
        if key not in self.values:
            self.values[key] = VyperEvaluator.default_value(self.value_type)
            super().__setitem__(key, self.values[key])
        return self.values[key]

    def __setitem__(self, key, value):
        if key in self.values:
            self._journal(key, self.values[key])
        else:
            self._journal(key, None)

        self.values[key] = value
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if key in self.values:
            self._journal(key, self.values[key])
            del self.values[key]
            super().__delitem__(key)


class Struct(Container):
    def __init__(
        self,
        typ: StructT,
        kws: Dict[str, Any],
        data_location: Optional[DataLocation] = None,
    ):
        super().__init__(typ, data_location)
        self.typ = typ
        self.values = {key: value for key, value in kws.items()}

    def __getitem__(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        return self.values[key]

    def __setitem__(self, key: str, value: Any):
        if key not in self.values:
            raise KeyError(f"'{self.typ.name}' struct has no member '{key}'")
        self._journal(key, self.values[key])
        self.values[key] = value

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def items(self):
        return self.values.items()

    def values(self):
        return self.values.values()

    def keys(self):
        return self.values.keys()


def make_container(
    typ: VyperType,
    data_location: Optional[DataLocation] = None,
    initial_value: Any = None,
) -> Any:
    """Factory function to create appropriate container type based on VyperType"""
    if isinstance(typ, _SequenceT):
        if typ.count is None:
            return DynamicArray(typ.value_type, data_location)
        else:
            return StaticArray(typ.count, typ.value_type, data_location)
    elif isinstance(typ, HashMapT):
        return Map(typ.key_type, typ.value_type, data_location)
    elif isinstance(typ, StructT):
        kws = {k: VyperEvaluator.default_value(v) for k, v in typ.members.items()}
        if initial_value:
            kws.update(initial_value)
        return Struct(typ, kws, data_location)
    else:
        # For primitive types, just return the value
        return initial_value or VyperEvaluator.default_value(typ)

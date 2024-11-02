from typing import Any, Dict, Generic, Optional, TypeVar

from vyper.semantics.types import VyperType, StructT, HashMapT
from vyper.semantics.types.subscriptable import _SequenceT

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType, JournalableCell
from ivy.base_types import Struct


T = TypeVar("T")


class StorageSequence(Generic[T]):
    def __init__(self, value_type: VyperType):
        self.value_type = value_type
        self.cells: Dict[int, JournalableCell] = {}

    def _make_cell(self, idx: int) -> JournalableCell:
        """Create a new cell with default value"""
        default = VyperEvaluator.default_value(self.value_type)
        return JournalableCell(self, idx, default)

    def __getitem__(self, idx: int) -> T:
        if idx >= len(self):
            raise IndexError("sequence index out of range")
        if idx not in self.cells:
            self.cells[idx] = self._make_cell(idx)
        return self.cells[idx].value

    def __setitem__(self, idx: int, value: T):
        if idx >= len(self):
            raise IndexError("sequence index out of range")
        if idx not in self.cells:
            self.cells[idx] = JournalableCell(self, idx, value)
        else:
            self.cells[idx].value = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class StaticStorageArray(StorageSequence[T]):
    def __init__(self, length: int, value_type: VyperType):
        super().__init__(value_type)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def append(self, value: T):
        raise AttributeError("'StaticStorageArray' object has no attribute 'append'")

    def pop(self) -> T:
        raise AttributeError("'StaticStorageArray' object has no attribute 'pop'")


class DynamicStorageArray(StorageSequence[T]):
    def __init__(self, value_type: VyperType, max_length: Optional[int] = None):
        super().__init__(value_type)
        self.max_length = max_length
        self._length = 0  # Track current length separately

    def __len__(self) -> int:
        return self._length

    def append(self, value: T):
        if self.max_length and len(self) >= self.max_length:
            raise ValueError(f"Cannot exceed maximum length {self.max_length}")

        Journal().record(JournalEntryType.STORAGE, self, "length", self._length)

        idx = self._length
        self.cells[idx] = JournalableCell(self, idx, value)
        self._length += 1

    def pop(self) -> T:
        if not self._length:
            raise IndexError("pop from empty array")

        Journal().record(JournalEntryType.STORAGE, self, "length", self._length)

        idx = self._length - 1
        self._length -= 1

        cell = self.cells.pop(idx, None)
        if cell is None:
            # If cell was never allocated, return default
            return VyperEvaluator.default_value(self.value_type)
        return cell.value


class StorageMap(dict):
    def __init__(self, key_type: VyperType, value_type: VyperType):
        super().__init__()
        self.cells: Dict[Any, JournalableCell] = {}
        self.value_type = value_type
        self.key_type = key_type

    def __getitem__(self, key):
        if key not in self.cells:
            default = VyperEvaluator.default_value(self.value_type)
            self.cells[key] = JournalableCell(self, key, default)
            super().__setitem__(key, default)
        return self.cells[key].value

    def __setitem__(self, key, value):
        if key not in self.cells:
            self.cells[key] = JournalableCell(self, key, value)
        else:
            self.cells[key].value = value
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if key in self.cells:
            Journal().record(JournalEntryType.STORAGE, self, key, self.cells[key].value)
            del self.cells[key]
            super().__delitem__(key)


class StorageStruct(Struct):
    def __init__(self, typ: StructT, kws: Dict[str, Any]):
        super().__init__(typ, {})  # Initialize empty first
        self.cells = {
            key: JournalableCell(self, key, value) for key, value in kws.items()
        }
        # Initialize the underlying dict with values
        super().update({k: cell.value for k, cell in self.cells.items()})

    def __getitem__(self, key: str):
        return self.cells[key].value

    def __setitem__(self, key: str, value: Any):
        self.cells[key].value = value
        super().__setitem__(key, value)


def make_storage_type(typ: VyperType, initial_value: Any = None) -> Any:
    """Factory function to create appropriate storage type based on VyperType"""
    if isinstance(typ, _SequenceT):
        if typ.count is None:
            # Dynamic array
            return DynamicStorageArray(typ.value_type)
        else:
            # Static array with fixed size
            return StaticStorageArray(typ.count, typ.value_type)
    elif isinstance(typ, HashMapT):
        return StorageMap(typ.key_type, typ.value_type)
    elif isinstance(typ, StructT):
        kws = {k: VyperEvaluator.default_value(v) for k, v in typ.members.items()}
        if initial_value:
            kws.update(initial_value)
        return StorageStruct(typ, kws)
    else:
        # For primitive types, just use JournalableCell directly
        return JournalableCell(
            None, None, initial_value or VyperEvaluator.default_value(typ)
        )

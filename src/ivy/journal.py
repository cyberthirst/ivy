from typing import Any
from enum import Enum

from vyper.semantics.data_locations import DataLocation

from ivy.exceptions import StaticCallViolation


class JournalEntryType(Enum):
    STORAGE = 1
    BALANCE = 2
    ACCOUNT_CREATION = 3
    ACCOUNT_DESTRUCTION = 4
    # Add more types as needed


class JournalEntry:
    def __init__(
        self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any
    ):
        self.entry_type = entry_type
        self.obj = obj
        self.key = key
        self.old_value = old_value


class Journal:
    _instance = None
    _is_static = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Journal, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    @classmethod
    def journalable_loc(cls, location: DataLocation):
        return location in (DataLocation.STORAGE, DataLocation.TRANSIENT)

    def initialize(self):
        self.recorded_entries: list[dict[tuple[int, Any, Any], JournalEntry]] = []
        self._is_static = []

    def record(self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any):
        # this is a hacky way to check whether we're executing a tx
        # evm state can be set through setters
        # exposed in the `Env` class (e.g. `set_balance`), we don't want
        # to journal those
        if len(self.recorded_entries) == 0:
            return
        if self._is_static[-1]:
            # journal is aware of all state changes
            # so we check static violation here
            raise StaticCallViolation()
        entry_key = (id(obj), key, entry_type.value)
        if entry_key not in self.recorded_entries[-1]:
            entry = JournalEntry(entry_type, obj, key, old_value)
            self.recorded_entries[-1][entry_key] = entry

    def begin_call(self, is_static: bool):
        self.recorded_entries.append({})
        self._is_static.append(is_static)

    def finalize_call(self, is_error):
        if is_error:
            self._rollback()
        else:
            self._commit()
        self._is_static.pop()

    def _commit(self):
        if len(self.recorded_entries) > 1:
            committed_records = self.recorded_entries.pop()
            self.recorded_entries[-1].update(committed_records)
        else:
            assert len(self.recorded_entries) == 1
            self.recorded_entries.pop()

    def _rollback(self):
        assert len(self.recorded_entries) > 0
        if self.recorded_entries:
            entries_to_rollback = self.recorded_entries.pop()
            for entry in reversed(entries_to_rollback.values()):
                self._apply_rollback(entry)

    def _apply_rollback(self, entry: JournalEntry):
        # if entry.entry_type == JournalEntryType.ACCOUNT_CREATION:
        #    del entry.obj[entry.key]
        # elif entry.entry_type == JournalEntryType.ACCOUNT_DESTRUCTION:
        #    entry.obj[entry.key] = entry.old_value
        if entry.entry_type == JournalEntryType.BALANCE:
            entry.obj.balance = entry.old_value
        elif entry.entry_type == JournalEntryType.STORAGE:
            entry.obj[entry.key] = entry.old_value
        else:
            assert False, f"unreachable: {entry.entry_type}"

    def reset(self):
        self.initialize()


class JournalableCell:
    def __init__(self, container: Any, key: Any, initial_value: Any):
        self.container = container
        self.key = key
        self._value = initial_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        Journal().record(
            JournalEntryType.STORAGE, self.container, self.key, self._value
        )
        self._value = new_value

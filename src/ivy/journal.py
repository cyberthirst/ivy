from typing import Any
from enum import Enum


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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Journal, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.recorded_entries: list[dict[tuple[int, Any, Any], JournalEntry]] = [{}]

    def record(self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any):
        entry_key = (id(obj), key, entry_type.value)
        if entry_key not in self.recorded_entries[-1]:
            entry = JournalEntry(entry_type, obj, key, old_value)
            self.recorded_entries[-1][entry_key] = entry

    def begin_call(self):
        self.recorded_entries.append({})

    def finalize_call(self, is_error):
        if is_error:
            self.rollback()
        else:
            self.commit()

    def commit(self):
        if len(self.recorded_entries) > 1:
            committed_records = self.recorded_entries.pop()
            self.recorded_entries[-1].update(committed_records)

    def rollback(self):
        if self.recorded_entries:
            entries_to_rollback = self.recorded_entries.pop()
            for entry in reversed(entries_to_rollback.values()):
                self.apply_rollback(entry)

    def apply_rollback(self, entry: JournalEntry):
        if False:
            pass
        # if entry.entry_type == JournalEntryType.ACCOUNT_CREATION:
        #    del entry.obj[entry.key]
        # elif entry.entry_type == JournalEntryType.ACCOUNT_DESTRUCTION:
        #    entry.obj[entry.key] = entry.old_value
        # elif entry.entry_type == JournalEntryType.BALANCE:
        #    entry.obj.balance = entry.old_value
        else:
            entry.obj[entry.key] = entry.old_value

    def reset(self):
        self.initialize()

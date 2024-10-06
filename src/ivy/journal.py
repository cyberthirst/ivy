from typing import Any, Dict, List, Tuple
from enum import Enum
import contextlib


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
    def __init__(self):
        self.stack: List[List[JournalEntry]] = [[]]
        self.recorded_entries: List[Dict[Tuple[int, Any, Any], bool]] = [{}]

    def record(self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any):
        entry_key = (id(obj), key, entry_type.value)
        if entry_key not in self.recorded_entries[-1]:
            entry = JournalEntry(entry_type, obj, key, old_value)
            self.stack[-1].append(entry)
            self.recorded_entries[-1][entry_key] = True

    @contextlib.contextmanager
    def nested_call(self):
        self.stack.append([])
        self.recorded_entries.append({})
        try:
            yield
        except Exception as e:
            self.rollback()
            raise e
        else:
            self.commit()

    def commit(self):
        if len(self.stack) > 1:
            committed_entries = self.stack.pop()
            self.stack[-1].extend(committed_entries)

            committed_records = self.recorded_entries.pop()
            self.recorded_entries[-1].update(committed_records)

    def rollback(self):
        self.stack.pop()
        self.recorded_entries.pop()

    def apply_rollback(self, entry: JournalEntry):
        if entry.entry_type == JournalEntryType.ACCOUNT_CREATION:
            del entry.obj[entry.key]
        elif entry.entry_type == JournalEntryType.ACCOUNT_DESTRUCTION:
            entry.obj[entry.key] = entry.old_value
        else:
            setattr(entry.obj, entry.key, entry.old_value)

from typing import Any, NamedTuple, Optional
from enum import Enum, auto

from vyper.semantics.data_locations import DataLocation

from ivy.exceptions import StaticCallViolation


class JournalEntryType(Enum):
    """Types of state changes that can be recorded in the journal."""

    STORAGE = auto()
    BALANCE = auto()
    ACCOUNT_CREATION = auto()
    ACCOUNT_DESTRUCTION = auto()
    NONCE = auto()
    CODE = auto()
    TRANSIENT_STORAGE = auto()
    ARRAY_LENGTH = auto()


class JournalEntry:
    """
    Represents a single state change in the EVM that may need to be reverted.

    Attributes:
        entry_type: The type of state change (storage, balance, etc.)
        obj: The object being modified
        key: The key or identifier for the specific value being changed
        old_value: The original value before modification
    """

    def __init__(
        self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any
    ):
        self.entry_type = entry_type
        self.obj = obj
        self.key = key
        self.old_value = old_value


JournalKey = tuple[int, Any, int]  # (obj_id, key, entry_type_val)


class JournalFrame(NamedTuple):
    entries: dict[JournalKey, JournalEntry]
    is_static: bool


class Journal:
    """
    Records and manages state changes during EVM execution.

    The journal provides transaction atomicity by tracking all state changes
    and allowing them to be rolled back if execution fails. It also enforces
    static call constraints.
    """

    _instance = None
    _frame_stack: list[JournalFrame] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Journal, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    @classmethod
    def journalable_loc(cls, location: DataLocation) -> bool:
        return location in (DataLocation.STORAGE, DataLocation.TRANSIENT)

    @property
    def current_frame(self) -> Optional[JournalFrame]:
        return self._frame_stack[-1] if self._frame_stack else None

    @property
    def is_active(self) -> bool:
        return bool(self._frame_stack)

    @property
    def in_static_context(self) -> bool:
        return self.current_frame.is_static if self.current_frame else False

    def initialize(self):
        self._frame_stack = []

    def record(
        self, entry_type: JournalEntryType, obj: Any, key: Any, old_value: Any
    ) -> None:
        # Skip recording if not in an active transaction
        # This might happen e.g. when using convenience setters (like `set_balance`) from `Env`
        if not self.is_active:
            return

        # Check for static call violations
        if self.in_static_context:
            raise StaticCallViolation("State modification in static context")

        # Only record the first change to each state element
        entry_key = (id(obj), key, entry_type.value)
        if entry_key not in self.current_frame.entries:
            entry = JournalEntry(entry_type, obj, key, old_value)
            self.current_frame.entries[entry_key] = entry

    def begin_call(self, is_static: bool) -> None:
        # If parent frame is static, child must also be static
        if self._frame_stack and self.current_frame.is_static:
            is_static = True

        self._frame_stack.append(JournalFrame(entries={}, is_static=is_static))

    def finalize_call(self, is_error: bool) -> None:
        if not self._frame_stack:
            raise IndexError("finalize_call called with no active journal frames")

        if is_error:
            self._rollback()
        else:
            self._commit()

    def _commit(self) -> None:
        """Commit changes from the current frame, merging them with the parent frame if it exists."""
        if not self._frame_stack:
            return

        frame = self._frame_stack.pop()

        # If this isn't the root frame, merge changes upward
        # Only add entries that parent doesn't already have - this preserves
        # the earliest old_value for proper rollback on nested calls
        if self._frame_stack:
            for key, entry in frame.entries.items():
                if key not in self.current_frame.entries:
                    self.current_frame.entries[key] = entry

    def _rollback(self) -> None:
        """Revert all changes in the current frame."""
        if not self._frame_stack:
            return

        frame = self._frame_stack.pop()

        for entry in reversed(list(frame.entries.values())):
            self._apply_rollback(entry)

    def _apply_rollback(self, entry: JournalEntry) -> None:
        if entry.entry_type == JournalEntryType.BALANCE:
            entry.obj._balance = entry.old_value
        elif entry.entry_type == JournalEntryType.STORAGE:
            storage = getattr(entry.obj, "_values", entry.obj)
            if entry.old_value is None:
                # Key didn't exist before (e.g., from append), remove it
                storage.pop(entry.key, None)
            else:
                storage[entry.key] = entry.old_value
        elif entry.entry_type == JournalEntryType.TRANSIENT_STORAGE:
            if entry.old_value is None:
                # Key didn't exist before, remove it
                entry.obj.pop(entry.key, None)
            else:
                entry.obj[entry.key] = entry.old_value
        elif entry.entry_type == JournalEntryType.NONCE:
            entry.obj.nonce = entry.old_value
        elif entry.entry_type == JournalEntryType.CODE:
            entry.obj.contract_data = entry.old_value
        elif entry.entry_type == JournalEntryType.ACCOUNT_CREATION:
            if entry.key in entry.obj:
                del entry.obj[entry.key]
        elif entry.entry_type == JournalEntryType.ACCOUNT_DESTRUCTION:
            entry.obj[entry.key] = entry.old_value
        elif entry.entry_type == JournalEntryType.ARRAY_LENGTH:
            entry.obj.length = entry.old_value
        else:
            raise ValueError(f"Unknown journal entry type: {entry.entry_type}")

    def reset(self) -> None:
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

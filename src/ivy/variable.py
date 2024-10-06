from vyper.semantics.types import VyperType

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType


class GlobalVariable:
    name: str
    typ: VyperType
    location: dict  # TODO can we make this more specific?
    journal: Journal

    def __init__(self, name: str, typ: VyperType, location: dict, journal: Journal):
        self.typ = typ
        self.location = location
        self.name = name
        self.location[self.name] = VyperEvaluator.default_value(typ)
        self.journal = journal

    @property
    def value(self):
        return self.location[self.name]

    @value.setter
    def value(self, new_value):
        old_value = self.location.get(self.name, None)
        self.journal.record(
            JournalEntryType.STORAGE, self.location, self.name, old_value
        )
        self.location[self.name] = new_value

from vyper.semantics.types import VyperType

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType


class GlobalVariable:
    name: str
    typ: VyperType
    location: dict  # TODO can we make this more specific?

    def __init__(self, name: str, typ: VyperType, location: dict):
        self.typ = typ
        self.location = location
        self.name = name
        self.location[self.name] = VyperEvaluator.default_value(typ)

    @property
    def value(self):
        if self.name not in self.location:
            self.location[self.name] = VyperEvaluator.default_value(self.typ)
        return self.location[self.name]

    @value.setter
    def value(self, new_value):
        self.location[self.name] = new_value

    def record(self):
        old_value = self.location.get(self.name, None)
        Journal().record(JournalEntryType.STORAGE, self.location, self.name, old_value)

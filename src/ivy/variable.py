from vyper.semantics.types import VyperType

from ivy.evaluator import VyperEvaluator


class GlobalVariable:
    # TODO add a reference to execution journal
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
        return self.location[self.name]

    @value.setter
    def value(self, new_value):
        # TODO register old value in execution journal
        self.location[self.name] = new_value

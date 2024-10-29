from urllib.response import addbase

from vyper.semantics.analysis.base import VarInfo
from vyper.semantics.types import VyperType, BoolT

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType


class GlobalVariable:
    address: int
    typ: VyperType
    location: dict  # TODO can we make this more specific?

    def __init__(self, address: int, typ: VyperType, location: dict):
        self.typ = typ
        self.location = location
        self.address = address
        self.location[self.address] = VyperEvaluator.default_value(typ)

    @property
    def value(self):
        if self.address not in self.location:
            self.location[self.address] = VyperEvaluator.default_value(self.typ)
        return self.location[self.address]

    @value.setter
    def value(self, new_value):
        self.location[self.address] = new_value

    def record(self):
        old_value = self.location.get(self.address, None)
        Journal().record(
            JournalEntryType.STORAGE, self.location, self.address, old_value
        )


class GlobalVariables:
    def __init__(self):
        self.variables = {}
        self.reentrant_key_address = None

    def new_variable(self, var: VarInfo, typ: VyperType, location: dict):
        address = var.position
        assert address not in self.variables
        variable = GlobalVariable(address, typ, location)
        self.variables[address] = variable

    def __setitem__(self, key: VarInfo, value):
        address = key.position
        self.variables[address] = value

    def __getitem__(self, key: VarInfo):
        address = key.position
        return self.variables[address]

    def allocate_reentrant_key(self, address: int, location):
        assert self.reentrant_key_address is None
        self.reentrant_key_address = address
        self.variables[address] = GlobalVariable(address, BoolT(), location)

    def set_reentrant_key(self, value: bool):
        address = self.reentrant_key_address
        self.variables[address] = value

    def get_reentrant_key(self):
        address = self.reentrant_key_address
        return self.variables[address]

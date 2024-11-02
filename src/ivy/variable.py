from typing import Optional

from vyper.semantics.analysis.base import VarInfo
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import VyperType, BoolT

from ivy.evaluator import VyperEvaluator
from ivy.journal import Journal, JournalEntryType


class GlobalVariable:
    address: int
    typ: VyperType
    location: dict  # TODO can we make this more specific?
    varinfo: Optional[VarInfo]

    def __init__(
        self,
        address: int,
        typ: VyperType,
        location: dict,
        varinfo: Optional[VarInfo] = None,
    ):
        self.typ = typ
        self.location = location
        self.address = address
        self.location[self.address] = VyperEvaluator.default_value(typ)
        self.varinfo = varinfo

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

    def _get_address(self, var: VarInfo):
        return (var.position, var.location)

    def new_variable(self, var: VarInfo, location: dict):
        address = self._get_address(var)
        assert address not in self.variables
        variable = GlobalVariable(var.position, var.typ, location, var)
        self.variables[address] = variable

    def __setitem__(self, key: VarInfo, value):
        address = self._get_address(key)
        self.variables[address] = value

    def __getitem__(self, key: VarInfo):
        address = self._get_address(key)
        return self.variables[address]

    def allocate_reentrant_key(self, position: int, location):
        assert self.reentrant_key_address is None
        address = (position, DataLocation.TRANSIENT)
        self.reentrant_key_address = address
        assert address not in self.variables
        self.variables[address] = GlobalVariable(position, BoolT(), location)

    def set_reentrant_key(self, value: bool):
        address = self.reentrant_key_address
        self.variables[address].value = value

    def get_reentrant_key(self):
        address = self.reentrant_key_address
        return self.variables[address].value

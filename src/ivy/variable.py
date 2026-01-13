from typing import Optional

from vyper.semantics.analysis.base import VarInfo
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import VyperType, BoolT

from ivy.expr.default_values import get_default_value
from ivy.journal import Journal, JournalEntryType


class GlobalVariable:
    address: int
    typ: VyperType
    get_location: callable  # This will store the lambda
    varinfo: Optional[VarInfo]

    def __init__(
        self,
        address: int,
        typ: VyperType,
        get_location: callable,  # Now receives a function instead of direct location
        varinfo: Optional[VarInfo] = None,
        initial_value: Optional[bool] = None,
    ):
        self.typ = typ
        self.get_location = get_location
        self.address = address
        if initial_value is None:
            initial_value = get_default_value(typ)
        self.get_location()[self.address] = initial_value
        self.varinfo = varinfo

    @property
    def value(self):
        location = self.get_location()
        if self.address not in location:
            location[self.address] = get_default_value(self.typ)
        return location[self.address]

    @value.setter
    def value(self, new_value):
        location = self.get_location()
        if self.varinfo and Journal.journalable_loc(self.varinfo.location):
            old_value = location.get(self.address, None)
            entry_type = (
                JournalEntryType.TRANSIENT_STORAGE
                if self.varinfo.location == DataLocation.TRANSIENT
                else JournalEntryType.STORAGE
            )
            Journal().record(entry_type, location, self.address, old_value)

        location[self.address] = new_value


class GlobalVariables:
    def __init__(self):
        self.variables = {}
        self.reentrant_key_address = None
        self.adrr_to_name = {}
        self.positions = {}  # VarInfo -> int

    def _get_address(self, var: VarInfo):
        return (self.positions[var], var.location)

    def new_variable(
        self,
        var: VarInfo,
        get_location: callable,
        position: int,
        initial_value=None,
        name=Optional[str],
    ):
        # Store position for this varinfo
        self.positions[var] = position
        address = (position, var.location)
        assert address not in self.variables
        variable = GlobalVariable(position, var.typ, get_location, var, initial_value)
        self.variables[address] = variable
        if name:
            self.adrr_to_name[address] = name

    def __setitem__(self, key: VarInfo, value):
        address = self._get_address(key)
        self.variables[address] = value

    def __getitem__(self, key: VarInfo):
        address = self._get_address(key)
        res = self.variables[address]
        assert res is not None
        return res

    def allocate_reentrant_key(self, position: int, get_location):
        assert self.reentrant_key_address is None
        address = (position, DataLocation.TRANSIENT)
        self.reentrant_key_address = address
        assert address not in self.variables
        # Create a VarInfo for the reentrant key so it gets journaled properly
        from vyper.semantics.analysis.base import Modifiability

        varinfo = VarInfo(
            typ=BoolT(),
            location=DataLocation.TRANSIENT,
            modifiability=Modifiability.MODIFIABLE,
            is_public=False,
            decl_node=None,
        )
        # Store position in our dict (not on varinfo) for consistency
        self.positions[varinfo] = position
        self.variables[address] = GlobalVariable(
            position, BoolT(), get_location, varinfo
        )

    def set_reentrant_key(self, value: bool):
        address = self.reentrant_key_address
        self.variables[address].value = value

    def get_reentrant_key(self):
        address = self.reentrant_key_address
        return self.variables[address].value

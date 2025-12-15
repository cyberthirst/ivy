import copy
import random
from typing import List

from vyper.ast import nodes as ast
from vyper.semantics.types import InterfaceT
from vyper.semantics.types.function import ContractFunctionT, StateMutability


class InterfaceRegistry:
    """Registry for generating interfaces for external calls."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counter = 0
        self._interfaces: List[ast.InterfaceDef] = []

    def reset(self):
        self.counter = 0
        self._interfaces.clear()

    def _mutability_name(self, mutability: StateMutability) -> str:
        if mutability == StateMutability.PURE:
            return "pure"
        elif mutability == StateMutability.VIEW:
            return "view"
        elif mutability == StateMutability.PAYABLE:
            return "payable"
        else:
            return "nonpayable"

    def _build_interface_func_def(self, func: ContractFunctionT) -> ast.FunctionDef:
        """Build interface function signature by copying ast_def and modifying body."""
        assert func.ast_def is not None, "External function must have ast_def"

        # Deep copy the function definition
        func_def: ast.FunctionDef = copy.deepcopy(func.ast_def)

        # Clear decorators (interface functions don't have them)
        func_def.decorator_list = []

        # Replace body with just mutability indicator
        mut_name = self._mutability_name(func.mutability)
        func_def.body = [ast.Expr(value=ast.Name(id=mut_name))]

        return func_def

    def create_interface(self, func: ContractFunctionT) -> tuple[str, InterfaceT]:
        """Create a fresh interface for the given function."""
        iface_name = f"IGen{self.counter}"
        self.counter += 1

        func_def = self._build_interface_func_def(func)
        iface_def = ast.InterfaceDef(name=iface_name, body=[func_def])

        iface_type = InterfaceT(
            _id=iface_name,
            decl_node=iface_def,
            functions={func.name: func},
            events={},
            structs={},
            flags={},
        )

        self._interfaces.append(iface_def)
        return iface_name, iface_type

    def get_interface_defs(self) -> List[ast.InterfaceDef]:
        return self._interfaces

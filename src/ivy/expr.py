from abc import abstractmethod
from typing import Optional

from vyper.ast import nodes as ast
from vyper.semantics.types import TYPE_T, InterfaceT, StructT, SelfT
from vyper.semantics.types.module import ModuleT

from ivy.evaluator import VyperEvaluator
from ivy.visitor import BaseVisitor
from ivy.types import Address

ENVIRONMENT_VARIABLES = {"block", "msg", "tx", "chain"}
ADDRESS_VARIABLES = {
    "address",
    "balance",
    "codesize",
    "is_contract",
    "codehash",
    "code",
}


class ExprVisitor(BaseVisitor):
    evaluator: VyperEvaluator

    @abstractmethod
    def generic_call_handler(
        self,
        func,
        args,
        kws,
        typs,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        pass

    @property
    @abstractmethod
    def current_address(self):
        pass

    @abstractmethod
    def _handle_env_variable(self, node: ast.Attribute):
        pass

    @abstractmethod
    def _handle_address_variable(self, node: ast.Attribute):
        pass

    def visit_Int(self, node: ast.Int):
        # literals are validated in Vyper
        return node.value

    def visit_Decimal(self, node: ast.Decimal):
        raise NotImplementedError("Decimal not implemented")

    def visit_Hex(self, node: ast.Hex):
        # literals are validated in Vyper
        return int(node.value, 16)

    def visit_Str(self, node: ast.Str):
        # literals are validated in Vyper
        return node.value

    def visit_Bytes(self, node: ast.Bytes):
        # literals are validated in Vyper
        return node.value

    def visit_NameConstant(self, node: ast.NameConstant):
        return node.value

    def visit_Name(self, node: ast.Name):
        if node.id == "self":
            return self.current_address
        return self.get_variable(node.id, node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in ADDRESS_VARIABLES:
            return self._handle_address_variable(node)
        elif (
            isinstance(node.value, ast.Name) and node.value.id in ENVIRONMENT_VARIABLES
        ):
            return self._handle_env_variable(node)
        typ = node.value._metadata["type"]
        if isinstance(typ, (SelfT, ModuleT)):
            return self.get_variable(node.attr, node)
        else:
            assert isinstance(typ, StructT)
            obj = self.visit(node.value)
            return obj[node.attr]

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return value[slice]

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.evaluator.eval_binop(node, left, right)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.evaluator.eval_compare(node, left, right)

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        return self.evaluator.eval_boolop(node, values)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        return self.evaluator.eval_unaryop(node, operand)

    def visit_List(self, node: ast.List):
        return [self.visit(elem) for elem in node.elements]

    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(elem) for elem in node.elements)

    def visit_IfExp(self, node: ast.IfExp):
        test = self.visit(node.test)
        if test:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_ExtCall(self, node: ast.ExtCall):
        return self._visit_external_call(node, is_static=False)

    def visit_StaticCall(self, node: ast.StaticCall):
        return self._visit_external_call(node, is_static=True)

    def _visit_external_call(self, node, is_static: bool):
        call_node = node.value
        # `func` always is an `Attribute` node so to get the target we need to visit the `value` of the `Attribute`
        assert isinstance(call_node.func, ast.Attribute)
        typ = call_node.func.value._metadata["type"]
        assert isinstance(typ, InterfaceT)
        address = self.visit(call_node.func.value)
        return self._visit_generic_call(call_node, target=address, is_static=is_static)

    def visit_Call(self, node: ast.Call):
        return self._visit_generic_call(node)

    def _visit_generic_call(
        self,
        node,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        assert isinstance(node, ast.Call)
        args = ()
        typs = ()
        for arg in node.args:
            typ = arg._metadata["type"]
            if isinstance(typ, TYPE_T):
                args += (typ.typedef,)
                typs += (typ,)
            else:
                args += (self.visit(arg),)
                typs += (typ,)
        kws = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return self.generic_call_handler(node, args, kws, typs, target, is_static)

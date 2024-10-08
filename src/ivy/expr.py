from abc import abstractmethod
from typing import Optional

from vyper.ast import nodes as ast
from vyper.semantics.types import TYPE_T

from ivy.visitor import BaseVisitor
from titanoboa.boa.util.abi import Address


class ExprVisitor(BaseVisitor):
    def visit_Int(self, node: ast.Int):
        return node.value

    def visit_Decimal(self, node: ast.Decimal):
        raise NotImplementedError("Decimal not implemented")

    def visit_Hex(self, node: ast.Hex):
        return int(node.value, 16)

    def visit_Str(self, node: ast.Str):
        return node.value

    def visit_Bytes(self, node: ast.Bytes):
        return node.value

    def visit_NameConstant(self, node: ast.NameConstant):
        return node.value

    def visit_Name(self, node: ast.Name):
        if node.id == "self":
            return self.current_address
        return self.get_variable(node.id)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            try:
                return self.get_variable(node.attr)
            except KeyError:
                pass
            raise NotImplementedError(
                f"Getting value from {type(node)} not implemented"
            )
        else:
            obj = self.visit(node.value)
            return getattr(obj, node.attr)

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
        op = node.op.__class__.__name__
        return self.evaluator.eval_boolop(op, values)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op = node.op.__class__.__name__
        return self.evaluator.eval_unaryop(op, operand)

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
        # TODO properly handle target - currently we only support Interface(expr).method()
        target = self.visit(node.value.func.value.args[0])
        return self._visit_generic_call(node.value, target=target, is_static=is_static)

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

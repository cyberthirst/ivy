from abc import ABC, abstractmethod

from vyper.ast import nodes as ast

from ivy.visitor import BaseVisitor

class ExprVisitor(BaseVisitor):

    def generic_visit(self, node: ast.VyperNode):
        raise Exception(f"No visit method for {type(node).__name__}")

    def visit_Int(self, node: ast.Int):
        return node.value

    def visit_Decimal(self, node: ast.Decimal):
        return node.value * self.interpreter.DECIMAL_DIVISOR

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
            return self.interpreter.contract_address
        return self.get_variable(node.id)

    def visit_Attribute(self, node: ast.Attribute):
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return value[slice]

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.__class__.__name__
        return self.evaluator.eval_binop(op, left, right)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.__class__.__name__
        return self.evaluator.eval_compare(op, left, right)

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        op = node.op.__class__.__name__
        return self.evaluator.eval_boolop(op, values)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op = node.op.__class__.__name__
        return self.evaluator.eval_unaryop(op, operand)

    def visit_Call(self, node: ast.Call):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return self.handle_call(func, args)

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

    # Additional methods for handling external calls, static calls, etc.
    def visit_ExtCall(self, node: ast.ExtCall):
        return self.handle_external_call(node)

    def visit_StaticCall(self, node: ast.StaticCall):
        return self.handle_static_call(node)

    @abstractmethod
    def construct_value(self, node: ast.VyperNode):
        pass

    @abstractmethod
    def handle_call(self, func, args):
        pass

    @abstractmethod
    def handle_external_call(self, node):
        pass

    @abstractmethod
    def handle_static_call(self, node):
        pass
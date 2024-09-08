from abc import ABC, abstractmethod

from ivy.visitor import BaseVisitor

class ExprVisitor(BaseVisitor):

    def generic_visit(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    def visit_Int(self, node):
        return node.value

    def visit_Decimal(self, node):
        return node.value * self.interpreter.DECIMAL_DIVISOR

    def visit_Hex(self, node):
        return int(node.value, 16)

    def visit_Str(self, node):
        return node.value

    def visit_Bytes(self, node):
        return node.value

    def visit_NameConstant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id == "self":
            return self.interpreter.contract_address
        return self.get_variable(node.id)

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return value[slice]

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.__class__.__name__
        return self.handle_binop(op, left, right)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.__class__.__name__
        return self.handle_compare(op, left, right)

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        op = node.op.__class__.__name__
        return self.handle_boolop(op, values)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op = node.op.__class__.__name__
        return self.handle_unaryop(op, operand)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return self.handle_call(func, args)

    def visit_List(self, node):
        return [self.visit(elem) for elem in node.elements]

    def visit_Tuple(self, node):
        return tuple(self.visit(elem) for elem in node.elements)

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        if test:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    # Additional methods for handling external calls, static calls, etc.
    def visit_ExtCall(self, node):
        return self.handle_external_call(node)

    def visit_StaticCall(self, node):
        return self.handle_static_call(node)

    @abstractmethod
    def handle_boolop(self, op, values):
        pass

    @abstractmethod
    def handle_unaryop(self, op, operand):
        pass

    @abstractmethod
    def handle_compare(self, op, left, right):
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
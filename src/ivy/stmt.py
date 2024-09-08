from abc import ABC, abstractmethod

from ivy.visitor import BaseVisitor


class StmtVisitor(BaseVisitor):

    def visit_Expr(self, node):
        # Evaluate the expression
        return self.visit(node.value)

    def visit_Pass(self, node):
        # Do nothing for pass statement
        return None

    def visit_Name(self, node):
        if node.id == "vdb":
            # Handle debugger
            return None
        raise Exception(f"Unsupported Name: {node.id}")

    def visit_AnnAssign(self, node):
        value = self.visit(node.value)
        self.set_variable(node.target.id, value)
        return None

    def visit_Assign(self, node):
        value = self.visit(node.value)
        target = self.visit(node.target)
        self.set_variable(target, value)
        return None

    def visit_If(self, node):
        condition = self.visit(node.test)
        if condition:
            return self.visit_body(node.body)
        elif node.orelse:
            return self.visit_body(node.orelse)
        return None

    def visit_Assert(self, node):
        condition = self.visit(node.test)
        if not condition:
            if node.msg:
                msg = self.visit(node.msg)
                raise AssertionError(msg)
            else:
                raise AssertionError()
        return None

    def visit_Raise(self, node):
        if node.exc:
            exc = self.visit(node.exc)
            raise exc
        else:
            raise Exception("Generic raise")

    def visit_For(self, node):
        # Implement for loop logic
        pass

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        right = self.visit(node.value)
        left = self.get_variable(target)
        new_value = self.handle_binop(node.op, left, right)
        self.set_variable(target, new_value)
        return None

    def visit_Continue(self, node):
        return "continue"

    def visit_Break(self, node):
        return "break"

    def visit_Return(self, node):
        if node.value:
            return self.visit(node.value)
        return None

    def visit_body(self, body):
        for stmt in body:
            result = self.visit(stmt)
            if (
                result == "continue"
                or result == "break"
                or isinstance(result, Exception)
            ):
                return result
        return None
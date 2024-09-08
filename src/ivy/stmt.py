from vyper.ast.nodes import VyperNode

from ivy.visitor import BaseVisitor

from vyper.ast import nodes as ast


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class StmtVisitor(BaseVisitor):

    def visit_Expr(self, node: ast.Expr):
        # Evaluate the expression
        return self.visit(node.value)

    def visit_Pass(self, node: ast.Pass):
        # Do nothing for pass statement
        return None

    def visit_Name(self, node: ast.Name):
        if node.id == "vdb":
            # Handle debugger
            return None
        raise Exception(f"Unsupported Name: {node.id}")

    def visit_AnnAssign(self, node: ast.AnnAssign):
        value = self.visit(node.value)
        self.set_variable(node.target.id, value)
        return None

    def visit_Assign(self, node: ast.Assign):
        value = self.visit(node.value)
        target = self.visit(node.target)
        self.set_variable(target, value)
        return None

    def visit_If(self, node: ast.If):
        condition = self.visit(node.test)
        if condition:
            return self.visit_body(node.body)
        elif node.orelse:
            return self.visit_body(node.orelse)
        return None

    def visit_Assert(self, node: ast.Assert):
        condition = self.visit(node.test)
        if not condition:
            if node.msg:
                msg = self.visit(node.msg)
                raise AssertionError(msg)
            else:
                raise AssertionError()
        return None

    def visit_Raise(self, node: ast.Raise):
        if node.exc:
            exc = self.visit(node.exc)
            raise exc
        else:
            raise Exception("Generic raise")

    def visit_For(self, node: ast.For):
        # Implement for loop logic
        pass

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit(node.target)
        right = self.visit(node.value)
        left = self.get_variable(target)
        new_value = self.evaluator.eval_binop(node.op, left, right)
        self.set_variable(target, new_value)
        return None

    def visit_Continue(self, node: ast.Continue):
        return "continue"

    def visit_Break(self, node: ast.Break):
        return "break"

    def visit_Return(self, node: ast.Return):
        if node.value:
            value = self.visit(node.value)
            raise ReturnException(value)
        raise ReturnException(None)

    def visit_body(self, body: list[VyperNode]):
        for stmt in body:
            result = self.visit(stmt)
            if (
                result == "continue"
                or result == "break"
                or isinstance(result, Exception)
            ):
                return result
        return None
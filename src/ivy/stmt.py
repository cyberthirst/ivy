from abc import abstractmethod

from vyper.ast.nodes import VyperNode
from vyper.ast import nodes as ast
from vyper.semantics.types import VyperType

from ivy.journal import Journal
from ivy.visitor import BaseVisitor


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class ContinueException(Exception):
    pass


class BreakException(Exception):
    pass


class StmtVisitor(BaseVisitor):
    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_Pass(self, node: ast.Pass):
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign):
        value = self.visit(node.value)
        typ = node.target._expr_info.typ
        self._new_variable(node.target.id, typ)
        self.set_variable(node.target.id, value)
        return None

    def visit_Assign(self, node: ast.Assign):
        value = self.visit(node.value)
        self._assign_target(node.target, value)

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
        iterable = self.visit(node.iter)

        self._push_scope()

        try:
            target_name = node.target.target
            target_typ = target_name._expr_info.typ
            self._new_variable(target_name.id, target_typ)
            for item in iterable:
                self._assign_target(target_name, item)

                try:
                    for stmt in node.body:
                        self.visit(stmt)
                except ContinueException:
                    continue
                except BreakException:
                    break
        finally:
            self._pop_scope()

        return None

    def visit_AugAssign(self, node: ast.AugAssign):
        target_value = self.visit(node.target)
        value = self.visit(node.value)
        new_value = self.evaluator.eval_binop(
            node, target_value, value, aug_assign=True
        )
        self._assign_target(node.target, new_value)
        return None

    def visit_Continue(self, node: ast.Continue):
        raise ContinueException

    def visit_Break(self, node: ast.Break):
        raise BreakException

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

    def _assign_target2(self, target, value):
        if isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise TypeError("Cannot unpack non-iterable to tuple")
            if len(target.elements) != len(value):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elements, value):
                self._assign_target2(t, v)
        else:
            target = self.visit(target)
            if (varinfo := target._expr_info.var_info) is not None:
                if Journal.journalable_loc(varinfo.location):
                    # TODO record the value
                    pass

            target = value
            return

    def _assign_target(self, target, value):
        if isinstance(target, ast.Name):
            self.set_variable(target.id, value)
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise TypeError("Cannot unpack non-iterable to tuple")
            if len(target.elements) != len(value):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elements, value):
                self._assign_target(t, v)
        elif isinstance(target, ast.Subscript):
            container = self.visit(target.value)
            index = self.visit(target.slice)
            container[index] = value
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                self.set_variable(target.attr, value)
            else:
                # structs
                obj = self.visit(target.value)
                obj[target.attr] = value
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    @abstractmethod
    def _push_scope(self):
        pass

    @abstractmethod
    def _pop_scope(self):
        pass

    @abstractmethod
    def _new_variable(self, name: str, typ: VyperType, location: dict):
        pass

    @property
    @abstractmethod
    def memory(self):
        pass

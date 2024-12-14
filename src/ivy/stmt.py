from abc import abstractmethod

from vyper.ast.nodes import VyperNode
from vyper.ast import nodes as ast
from vyper.codegen.core import calculate_type_for_external_return
from vyper.semantics.types import VyperType
from vyper.utils import method_id

from ivy.abi import abi_encode
from ivy.exceptions import Assert, Raise, Invalid
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
        value = self.deep_copy_visit(node.value)
        typ = node.target._expr_info.typ
        self._new_local(node.target.id, typ)
        self.set_variable(node.target.id, value)
        return None

    def visit_Assign(self, node: ast.Assign):
        value = self.deep_copy_visit(node.value)
        self._assign_target(node.target, value)

        return None

    def visit_If(self, node: ast.If):
        condition = self.visit(node.test)
        if condition:
            return self.visit_body(node.body)
        elif node.orelse:
            return self.visit_body(node.orelse)
        return None

    def _revert_with_msg(self, msg_node, is_assert=False):
        msg_string = self.visit(msg_node)

        assert isinstance(msg_string, str)
        if msg_string == "UNREACHABLE":
            raise Invalid()

        # encode the msg and raise Revert
        error_method_id = method_id("Error(string)")
        typ = msg_node._metadata["type"]
        wrapped_typ = calculate_type_for_external_return(typ)
        wrapped_msg = (msg_string,)

        encoded = abi_encode(wrapped_typ, wrapped_msg)

        to_raise = Assert if is_assert else Raise

        raise to_raise(message=msg_string, data=error_method_id + encoded)

    def visit_Assert(self, node: ast.Assert):
        condition = self.visit(node.test)
        if not condition:
            if node.msg:
                self._revert_with_msg(node.msg, is_assert=True)
            else:
                raise Assert()
        return None

    def visit_Raise(self, node: ast.Raise):
        if node.exc:
            self._revert_with_msg(node.exc)
        else:
            raise Raise()

    def visit_For(self, node: ast.For):
        iterable = self.visit(node.iter)
        # Scope for the iterator variable (not stricly neccessary)
        self._push_scope()
        target_name = node.target.target
        target_typ = target_name._expr_info.typ
        self._new_local(target_name.id, target_typ)

        for item in iterable:
            # New scope for each iteration
            self._push_scope()
            self._assign_target(target_name, item)

            try:
                for stmt in node.body:
                    self.visit(stmt)
            except ContinueException:
                continue
            except BreakException:
                break
            finally:
                self._pop_scope()  # Pop iteration scope

        self._pop_scope()  # Pop iterator variable scope
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

    def visit_Log(self, node: ast.Log):
        assert isinstance(node.value, ast.Call)
        self.visit(node.value)

    def visit_Return(self, node: ast.Return):
        if node.value:
            value = self.deep_copy_visit(node.value)
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

    @abstractmethod
    def _push_scope(self):
        pass

    @abstractmethod
    def _pop_scope(self):
        pass

    @abstractmethod
    def _new_local(self, name: str, typ: VyperType):
        pass

    @property
    @abstractmethod
    def memory(self):
        pass

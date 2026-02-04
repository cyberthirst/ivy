from abc import abstractmethod
from contextlib import contextmanager

from vyper.ast.nodes import VyperNode
from vyper.ast import nodes as ast
from vyper.codegen.core import calculate_type_for_external_return
from vyper.semantics.types import VyperType
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.user import StructT
from vyper.semantics.types.primitives import SelfT
from vyper.utils import method_id

from ivy.abi import abi_encode
from ivy.exceptions import Assert, Raise, Invalid
from ivy.visitor import BaseVisitor
from ivy.expr.operators import get_operator_handler
from ivy.expr.clamper import box_value_from_node, box_value
from ivy.types import VyperString


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class ContinueException(Exception):
    pass


class BreakException(Exception):
    pass


class StmtVisitor(BaseVisitor):
    @contextmanager
    def _scoped(self):
        self._push_scope()
        try:
            yield
        finally:
            self._pop_scope()

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
        taken = bool(condition)
        self._on_branch(node, taken)
        if taken:
            with self._scoped():
                return self.visit_body(node.body)
        elif node.orelse:
            with self._scoped():
                return self.visit_body(node.orelse)
        return None

    def _revert_with_msg(self, msg_node, is_assert=False):
        # Check for UNREACHABLE before evaluating - it's a special AST node
        # without _expr_info metadata, so visit() would fail
        if isinstance(msg_node, ast.Name) and msg_node.id == "UNREACHABLE":
            raise Invalid()

        msg_value = self.visit(msg_node)
        if not isinstance(msg_value, VyperString):
            raise TypeError(f"Expected VyperString, got {type(msg_value)}")
        msg_bytes = msg_value
        msg_string = str(msg_value)
        assert isinstance(msg_string, str)

        # encode the msg and raise Revert
        error_method_id = method_id("Error(string)")
        typ = msg_node._metadata["type"]
        wrapped_typ = calculate_type_for_external_return(typ)
        wrapped_msg = (msg_bytes,)

        encoded = abi_encode(wrapped_typ, wrapped_msg)

        to_raise = Assert if is_assert else Raise

        raise to_raise(message=msg_string, data=error_method_id + encoded)

    def visit_Assert(self, node: ast.Assert):
        condition = self.visit(node.test)
        self._on_branch(node, bool(condition))
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

        iteration_count = 0
        try:
            for item in iterable:
                iteration_count += 1
                # New scope for each iteration
                self._push_scope()
                # Box the item with the target type (e.g., range yields plain ints)
                self._assign_target(target_name, box_value(item, target_typ))

                try:
                    for stmt in node.body:
                        self.visit(stmt)
                except ContinueException:
                    continue
                except BreakException:
                    break
                finally:
                    self._pop_scope()  # Pop iteration scope
        finally:
            self._on_loop(node, iteration_count)
            self._pop_scope()  # Pop iterator variable scope
        return None

    def visit_AugAssign(self, node: ast.AugAssign):
        target = node.target

        # Evaluate target once, caching location info for reuse
        container = index = obj = None
        if isinstance(target, ast.Subscript):
            container, index, target_val = self._eval_subscript_location(target)
        elif isinstance(target, ast.Attribute):
            typ = target.value._metadata["type"]
            if isinstance(typ, (SelfT, ModuleT)):
                target_val = self.visit(target)
            else:
                # Struct attribute - base might have side effects
                _, refresh = self._eval_with_refresh(target.value)
                obj = refresh()
                target_val = obj[target.attr]
        else:
            target_val = self.visit(target)

        # Compute new value
        rhs_val = self.visit(node.value)
        new_val = get_operator_handler(node.op)(target_val, rhs_val)
        new_val = box_value_from_node(target, new_val)

        # Assign with cached location info
        self._assign_target(target, new_val, _container=container, _index=index, _obj=obj)
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

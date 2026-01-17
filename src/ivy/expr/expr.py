from abc import abstractmethod
from typing import Optional, Any, Callable

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    TYPE_T,
    InterfaceT,
    StructT,
    SelfT,
    FlagT,
    SArrayT,
    DArrayT,
    AddressT,
    BytesM_T,
)
from vyper.semantics.types.module import ModuleT
from vyper.semantics.namespace import Namespace
from vyper.semantics.types.utils import type_from_annotation

from ivy.visitor import BaseVisitor
from ivy.types import (
    Address,
    Flag,
    StaticArray,
    DynamicArray,
    VyperDecimal,
    VyperBool,
    VyperBytesM,
    Tuple as IvyTuple,
)
from ivy.expr.operators import get_operator_handler
from ivy.expr.clamper import box_value_from_node

ENVIRONMENT_VARIABLES = {"block", "msg", "tx", "chain"}
ADDRESS_VARIABLES = {
    "address",
    "balance",
    "codesize",
    "is_contract",
    "codehash",
    "code",
}

NAMESPACE = Namespace()


class ExprVisitor(BaseVisitor):
    @abstractmethod
    def generic_call_handler(
        self,
        func,
        args,
        kws,
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
        return box_value_from_node(node, node.value)

    def visit_Decimal(self, node: ast.Decimal):
        return VyperDecimal(node.value)

    def visit_Hex(self, node: ast.Hex):
        typ = node._metadata["type"]

        val = node.value

        if isinstance(typ, AddressT):
            return Address(val)
        assert isinstance(typ, BytesM_T)

        bytes_val = bytes.fromhex(val[2:])
        return VyperBytesM(bytes_val[: typ.m], typ)

    def visit_Str(self, node: ast.Str):
        return box_value_from_node(node, node.value)

    def visit_Bytes(self, node: ast.Bytes):
        return box_value_from_node(node, node.value)

    def visit_NameConstant(self, node: ast.NameConstant):
        # Box True/False to VyperBool, None passes through as-is
        return box_value_from_node(node, node.value)

    def visit_Name(self, node: ast.Name):
        if node.id == "self":
            return self.current_address
        if node.id in NAMESPACE:
            ret = type_from_annotation(node)
            return ret
        return self.get_variable(node.id, node)

    def _resolve_attribute(self, node: ast.Attribute, base_val=None):
        """
        Return the value of *node*.
        `base_val` is an optional *already evaluated* base object.
        Only the branches that actually need the runtime object will
        evaluate `self.visit(node.value)` (or use base_val if provided).
        """
        attr = node.attr

        # address helpers  (x.address, x.balance ..)
        if attr in ADDRESS_VARIABLES:
            return self._handle_address_variable(node)

        # environment variables (block.*, msg.* ..)
        if isinstance(node.value, ast.Name) and node.value.id in ENVIRONMENT_VARIABLES:
            return self._handle_env_variable(node)

        typ = node.value._metadata["type"]
        if hasattr(typ, "typedef"):
            typ = typ.typedef

        # self.x   or   MyModule.x
        if isinstance(typ, (SelfT, ModuleT)):
            return self.get_variable(attr, node)

        # struct.foo
        if isinstance(typ, StructT):
            if base_val is None:
                base_val = self.visit(node.value)  # lazy
            return base_val[attr]

        # MyFlag.BLUE
        if isinstance(typ, FlagT):
            return Flag(typ, attr)

        if base_val is None:
            base_val = self.visit(node.value)  # lazy
        return getattr(base_val, attr)

    def visit_Attribute(self, node: ast.Attribute):
        return self._resolve_attribute(node)

    def _eval_with_refresh(self, node: ast.VyperNode) -> tuple[Any, Callable[[], Any]]:
        """
        Evaluate *node* and return a tuple
            (value, refresh_callable)
        so that refresh_callable() will give the **current** value again
        *without* re‑executing any of the side‑effectful code that
        might have been inside the original expression.
        """

        if isinstance(node, ast.Attribute):
            typ = node.value._metadata["type"]
            if hasattr(typ, "typedef"):
                typ = typ.typedef

            if isinstance(typ, (SelfT, ModuleT)):
                result = self._resolve_attribute(node, base_val=None)

                def attribute_refresher_1(n=node, self_obj=self):
                    return self_obj._resolve_attribute(n, base_val=None)

                refresher = attribute_refresher_1
                return result, refresher

            base_val, base_refresh = self._eval_with_refresh(node.value)
            result = self._resolve_attribute(node, base_val)

            def attribute_refresher_2(br=base_refresh, n=node, self_obj=self):
                return self_obj._resolve_attribute(n, br())

            refresher = attribute_refresher_2
            return result, refresher

        if isinstance(node, ast.Subscript):
            base_val, base_refresh = self._eval_with_refresh(node.value)

            idx = self.visit(node.slice)

            def subscript_refresher(br=base_refresh, i=idx):
                return br()[i]

            return base_val[idx], subscript_refresher

        value = self.visit(node)

        def value_refresher(v=value):
            return v

        return value, value_refresher

    def _eval_subscript_location(self, node: ast.Subscript):
        """
        Evaluate a subscript, returning (container, index, current_value).
        Refreshes container after index evaluation to handle side effects.
        """
        _, refresh = self._eval_with_refresh(node.value)
        index = self.visit(node.slice)
        container = refresh()
        return container, index, container[index]

    def visit_Subscript(self, node: ast.Subscript):
        _, _, value = self._eval_subscript_location(node)
        return value

    def _eval_op(self, node, *args):
        handler = get_operator_handler(node.op)
        if isinstance(node.op, (ast.LShift, ast.Invert)):
            typ = node._metadata["type"]
            res = handler(*args, typ=typ)
        else:
            res = handler(*args)
        return box_value_from_node(node, res)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._eval_op(node, left, right)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._eval_op(node, left, right)

    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.Or):
            short_circuit_value = True
            op = "or"
        else:
            assert isinstance(node.op, ast.And)
            short_circuit_value = False
            op = "and"

        evaluated_count = 0
        for val in node.values:
            result = self.visit(val)
            evaluated_count += 1

            if result == short_circuit_value:
                self._on_boolop(node, op, evaluated_count, bool(result))
                return result

        assert isinstance(result, (bool, VyperBool))
        self._on_boolop(node, op, evaluated_count, bool(result))
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        return self._eval_op(node, operand)

    def visit_List(self, node: ast.List):
        idx = 0
        values = {}
        for elem in node.elements:
            values[idx] = self.visit(elem)
            idx += 1
        typ = node._metadata["type"]
        if isinstance(typ, SArrayT):
            return StaticArray(typ, values)
        assert isinstance(typ, DArrayT)
        return DynamicArray(typ, values)

    def visit_Tuple(self, node: ast.Tuple):
        typ = node._metadata["type"]
        t = IvyTuple(typ)
        for i, elem in enumerate(node.elements):
            t[i] = self.deep_copy_visit(elem)
        return t

    def visit_IfExp(self, node: ast.IfExp):
        test = self.visit(node.test)
        taken = bool(test)
        self._on_branch(node, taken)
        if taken:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_HexBytes(self, node: ast.HexBytes):
        return box_value_from_node(node, node.value)

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
        for arg in node.args:
            typ = arg._metadata["type"]
            if isinstance(typ, TYPE_T):
                # Type arguments (like in convert(x, uint256)) pass the type directly
                args += (typ.typedef,)
            else:
                # Value arguments are evaluated and boxed with their types
                args += (self.deep_copy_visit(arg),)
        kws = {}
        for kw in node.keywords:
            if kw.arg == "default_return_value":
                # defer evaluation of default_return_value until the call is made
                # and evaluate if the len(returndata) == 0
                kws[kw.arg] = kw.value
                continue
            kw_typ = kw.value._metadata["type"]
            if isinstance(kw_typ, TYPE_T):
                kws[kw.arg] = kw_typ.typedef
            else:
                kws[kw.arg] = self.deep_copy_visit(kw.value)
        return self.generic_call_handler(node, args, kws, target, is_static)

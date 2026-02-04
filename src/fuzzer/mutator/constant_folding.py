from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional, Literal

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    BytesT,
    DecimalT,
    DArrayT,
    IntegerT,
    SArrayT,
    StringT,
    StructT,
    TupleT,
    VyperType,
)

from vyper.utils import keccak256

from ivy.expr.clamper import box_value_from_node
from ivy.expr.expr import ExprVisitor
from ivy.types import (
    Address,
    DynamicArray,
    Flag,
    StaticArray,
    Struct,
    Tuple as IvyTuple,
    VyperBool,
    VyperBytes,
    VyperBytesM,
    VyperDecimal,
    VyperInt,
    VyperString,
)

from fuzzer.mutator import ast_builder


class ConstEvalError(Exception):
    pass


class ConstEvalNonConstant(ConstEvalError):
    pass


FoldStatus = Literal["value", "non_constant", "invalid_constant"]


def _bytes_value(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, VyperString):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (VyperBytes, VyperBytesM)):
        return bytes(value)
    raise ConstEvalError(f"unsupported bytes-like value: {type(value).__name__}")


def _unbox_value(value: Any, typ: VyperType) -> Any:
    if isinstance(value, VyperBool):
        return bool(value)
    if isinstance(value, VyperInt):
        return int(value)
    if isinstance(value, VyperBytesM):
        return bytes(value)
    if isinstance(value, VyperBytes):
        return bytes(value)
    if isinstance(value, VyperString):
        return str(value)
    if isinstance(value, VyperDecimal):
        return Decimal(str(value))
    if isinstance(value, Address):
        return str(value)
    if isinstance(value, Flag):
        return value.value
    if isinstance(value, (StaticArray, DynamicArray)):
        assert isinstance(typ, (SArrayT, DArrayT))
        return [_unbox_value(v, typ.value_type) for v in value]
    if isinstance(value, Struct):
        assert isinstance(typ, StructT)
        return {
            name: _unbox_value(value[name], typ.members[name]) for name in typ.members
        }
    if isinstance(value, IvyTuple):
        assert isinstance(typ, TupleT)
        return tuple(
            _unbox_value(value[i], member_t)
            for i, member_t in enumerate(typ.member_types)
        )
    if isinstance(typ, StringT) and isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="surrogateescape")
    if isinstance(typ, BytesT) and isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(typ, TupleT) and isinstance(value, tuple):
        return tuple(
            _unbox_value(member_v, member_t)
            for member_v, member_t in zip(value, typ.member_types)
        )
    if isinstance(typ, (SArrayT, DArrayT)) and isinstance(value, list):
        return [_unbox_value(v, typ.value_type) for v in value]
    return value


def _builtin_min_max(call: ast.Call, args: tuple[Any, ...], is_min: bool) -> Any:
    if len(args) != 2:
        raise ConstEvalError("min/max require two args")
    result = min(args) if is_min else max(args)
    return box_value_from_node(call, result)


def _builtin_abs(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1:
        raise ConstEvalError("abs expects one arg")
    arg = args[0]
    typ = call._metadata.get("type")
    if not isinstance(typ, IntegerT) or not typ.is_signed or typ.bits != 256:
        raise ConstEvalError("abs expects int256")
    if int(arg) == typ.ast_bounds[0]:
        raise ConstEvalError("abs overflow for minimum signed int")
    return box_value_from_node(call, abs(arg))


def _builtin_floor(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1 or not isinstance(args[0], VyperDecimal):
        raise ConstEvalError("floor expects one decimal arg")
    arg = args[0]
    result = arg.value // arg.SCALING_FACTOR
    return box_value_from_node(call, result)


def _builtin_ceil(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1 or not isinstance(args[0], VyperDecimal):
        raise ConstEvalError("ceil expects one decimal arg")
    arg = args[0]
    result = -((-arg.value) // arg.SCALING_FACTOR)
    return box_value_from_node(call, result)


def _builtin_len(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1:
        raise ConstEvalError("len expects one arg")
    try:
        result = len(args[0])
    except Exception as exc:
        raise ConstEvalError("len unsupported for value") from exc
    return box_value_from_node(call, result)


def _builtin_keccak256(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1:
        raise ConstEvalError("keccak256 expects one arg")
    data = _bytes_value(args[0])
    return box_value_from_node(call, keccak256(data))


def _builtin_method_id(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 1:
        raise ConstEvalError("method_id expects one arg")
    method = args[0]
    if not isinstance(method, VyperString):
        raise ConstEvalError("method_id expects string")
    return box_value_from_node(call, keccak256(bytes(method))[:4])


def _builtin_min_max_value(call: ast.Call, args: tuple[Any, ...], is_max: bool) -> Any:
    if len(args) != 1:
        raise ConstEvalError("min_value/max_value expect one arg")
    typ = args[0]
    if isinstance(typ, DecimalT):
        result = VyperDecimal.max() if is_max else VyperDecimal.min()
    elif isinstance(typ, IntegerT):
        result = typ.int_bounds[1] if is_max else typ.int_bounds[0]
    else:
        raise ConstEvalError("min_value/max_value unsupported type")
    return box_value_from_node(call, result)


def _builtin_shift(call: ast.Call, args: tuple[Any, ...]) -> Any:
    if len(args) != 2:
        raise ConstEvalError("shift expects two args")
    value = args[0]
    if not isinstance(value, VyperInt):
        raise ConstEvalError("shift expects int value")
    shift = int(args[1])

    if shift >= 256:
        result = 0
    elif shift >= 0:
        result = (int(value) << shift) % 2**256
    else:
        if -shift >= 256:
            result = -1 if value.typ.is_signed and int(value) < 0 else 0
        else:
            result = int(value) >> (-shift)

    if value.typ.is_signed and result >= 2**255:
        result -= 2**256

    return box_value_from_node(call, result)


_BUILTIN_HANDLERS = {
    "min": lambda call, args: _builtin_min_max(call, args, True),
    "max": lambda call, args: _builtin_min_max(call, args, False),
    "abs": _builtin_abs,
    "floor": _builtin_floor,
    "ceil": _builtin_ceil,
    "len": _builtin_len,
    "keccak256": _builtin_keccak256,
    "method_id": _builtin_method_id,
    "min_value": lambda call, args: _builtin_min_max_value(call, args, False),
    "max_value": lambda call, args: _builtin_min_max_value(call, args, True),
    "shift": _builtin_shift,
}


class ConstExprEvaluator(ExprVisitor):
    def __init__(self, constants: dict[str, Any]):
        self._constants = constants

    def generic_call_handler(
        self,
        call,
        args,
        kws,
        target=None,
        is_static=None,
    ):
        if any(isinstance(value, ast.VyperNode) for value in kws.values()):
            raise ConstEvalNonConstant("unsupported keyword arg value")
        if isinstance(call.func, ast.Attribute):
            raise ConstEvalNonConstant("method calls are not constant")
        if isinstance(call.func, ast.Name):
            handler = _BUILTIN_HANDLERS.get(call.func.id)
            if handler is not None:
                return handler(call, args)

        typ = call._metadata.get("type")
        if isinstance(typ, StructT):
            if args:
                raise ConstEvalError("struct constructors require keywords")
            return Struct(typ, kws)

        raise ConstEvalNonConstant("unsupported call in const eval")

    @property
    def current_address(self):
        raise ConstEvalNonConstant("self not allowed in const eval")

    def _handle_env_variable(self, node: ast.Attribute):
        raise ConstEvalNonConstant("env variables are not constant")

    def _handle_address_variable(self, node: ast.Attribute):
        raise ConstEvalNonConstant("address variables are not constant")

    def set_variable(self, name: str, value, node: Optional[ast.VyperNode] = None):
        raise ConstEvalNonConstant("const eval does not support assignment")

    def get_variable(self, name: str, node: Optional[ast.VyperNode] = None):
        if name in self._constants:
            return self._constants[name]
        raise ConstEvalNonConstant(f"unknown constant: {name}")


def evaluate_constant_expression(
    node: ast.VyperNode,
    constants: dict[str, Any],
) -> Any:
    evaluator = ConstExprEvaluator(constants)
    return evaluator.visit(node)


def fold_constant_expression(
    node: ast.VyperNode,
    constants: dict[str, Any],
) -> Optional[ast.VyperNode]:
    status, folded = fold_constant_expression_status(node, constants)
    if status != "value":
        return None
    return folded


def fold_constant_expression_status(
    node: ast.VyperNode,
    constants: dict[str, Any],
) -> tuple[FoldStatus, Optional[ast.VyperNode]]:
    typ = node._metadata.get("type")
    if typ is None:
        return "non_constant", None
    try:
        value = evaluate_constant_expression(node, constants)
    except ConstEvalNonConstant:
        return "non_constant", None
    except ConstEvalError:
        return "invalid_constant", None
    except Exception:
        return "invalid_constant", None
    if value is None:
        return "invalid_constant", None
    try:
        unboxed = _unbox_value(value, typ)
        return "value", ast_builder.literal(unboxed, typ)
    except Exception:
        return "invalid_constant", None


def constant_folds_to_zero(
    node: ast.VyperNode,
    constants: dict[str, Any],
) -> bool:
    folded = fold_constant_expression(node, constants)
    if folded is None:
        return False
    if isinstance(folded, ast.Int):
        return folded.value == 0
    if isinstance(folded, ast.Decimal):
        return folded.value == 0
    return False

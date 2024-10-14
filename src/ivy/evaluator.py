from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict

from vyper.ast import nodes as ast
from vyper.semantics.types import BoolT, InterfaceT
from vyper.semantics.types.primitives import IntegerT
from vyper.semantics.types.subscriptable import _SequenceT, HashMapT
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.user import StructT

from ivy.types import Address


class BaseEvaluator(ABC):
    @abstractmethod
    def eval_boolop(self, op, values):
        pass

    @abstractmethod
    def eval_unaryop(self, op, operand):
        pass

    @abstractmethod
    def eval_binop(self, op, left, right, aug_assign=False):
        pass

    @abstractmethod
    def eval_compare(self, op, left, right):
        pass

    @abstractmethod
    def default_value(self, typ):
        pass


class VyperEvaluator(BaseEvaluator):
    @staticmethod
    def validate_integer(value, typ):
        # For now, just return True
        return True

    @staticmethod
    def validate_bool(value, typ):
        return True

    @staticmethod
    def validate_sequence_len(value, typ):
        if len(value) > typ.length:
            raise ValueError(
                f"Invalid length for {typ}: expected at most {typ.count}, got {len(value)}"
            )

    @staticmethod
    def validate_bytes(value, typ):
        VyperEvaluator.validate_sequence_len(value, typ)

    @staticmethod
    def validate_string(value, typ):
        VyperEvaluator.validate_sequence_len(value, typ)

    @staticmethod
    def validate_sequence(value, typ):
        VyperEvaluator.validate_sequence_len(value, typ)
        for item in value:
            VyperEvaluator.validate_value(item, typ.value_type)

    @staticmethod
    def validate_struct(value, typ):
        pass

    @staticmethod
    def validate_hashmap(value, typ):
        pass

    @staticmethod
    def validate_interface(value, typ):
        pass

    type_validators = {
        "IntegerT": validate_integer,
        "BoolT": validate_bool,
        "BytesT": validate_bytes,
        "StringT": validate_string,
        "_SequenceT": validate_sequence,
        "StructT": validate_struct,
        "HashMapT": validate_hashmap,
        "InterfaceT": validate_interface,
        "SArrayT": validate_sequence,
        "DynArrayT": validate_sequence,
    }

    @classmethod
    def validate_value(cls, node, value):
        typ = node._metadata["type"]
        typ_name = typ.__class__.__name__
        cls.type_validators[typ_name](value, typ)

    @classmethod
    def eval_boolop(cls, op, values):
        eval = op.op._op
        res = eval(values)
        cls.validate_value(op, res)
        return res

    @classmethod
    def eval_unaryop(cls, op, operand):
        eval = op.op._op
        res = eval(operand)
        cls.validate_value(op, res)
        return res

    # aug_assign node is not annotated with a type, so we take the type from the target
    # alternatively we could fetch the type at the call site and have it passed in
    @classmethod
    def eval_binop(cls, op: ast.BinOp, left: Any, right: Any, aug_assign=False):
        eval = op.op._op
        res = eval(left, right)
        if aug_assign:
            cls.validate_value(op.target, res)
        else:
            cls.validate_value(op, res)
        return res

    @classmethod
    def eval_compare(cls, op: ast.Compare, left, right):
        eval = op.op._op
        res = eval(left, right)
        cls.validate_value(op, res)
        return res

    # rewrite to smth like dict for const-time dispatch
    @classmethod
    def default_value(cls, typ):
        if isinstance(typ, IntegerT):
            return 0
        if isinstance(typ, _SequenceT):
            return []
        if isinstance(typ, BytesT):
            return b""
        if isinstance(typ, StringT):
            return ""
        if isinstance(typ, StructT):
            kws = {k: cls.default_value(v) for k, v in typ.members.items()}
            return cls.construct_struct(typ.name, kws)
        if isinstance(typ, HashMapT):
            return defaultdict(lambda: cls.default_value(typ.value_type))
        if isinstance(typ, BoolT):
            return False
        if isinstance(typ, Address) or isinstance(typ, InterfaceT):
            return Address(0)
        return None

        raise NotImplementedError(f"Default value for {typ} not implemented")

    @classmethod
    def construct_struct(cls, name, kws):
        StructType = type(name, (object,), kws)
        return StructType()

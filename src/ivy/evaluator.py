from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict

from vyper.ast import nodes as ast
from vyper.semantics.types import BoolT, InterfaceT
from vyper.semantics.types.primitives import IntegerT
from vyper.semantics.types.subscriptable import _SequenceT, HashMapT
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.user import StructT

from titanoboa.boa.util.abi import Address


class BaseEvaluator(ABC):
    @abstractmethod
    def eval_boolop(self, op, values):
        pass

    @abstractmethod
    def eval_unaryop(self, op, operand):
        pass

    @abstractmethod
    def eval_binop(self, op, left, right):
        pass

    @abstractmethod
    def eval_compare(self, op, left, right):
        pass

    @abstractmethod
    def default_value(self, typ):
        pass


class VyperEvaluator(BaseEvaluator):
    @classmethod
    def eval_boolop(cls, op, values):
        eval = op.op._op
        res = eval(values)
        return res

    @classmethod
    def eval_unaryop(cls, op, operand):
        eval = op.op._op
        res = eval(operand)
        return res

    @classmethod
    def eval_binop(cls, op: ast.BinOp, left: Any, right: Any):
        print(f"eval_binop: {op.op._op}, {left}, {right}")
        eval = op.op._op
        res = eval(left, right)
        return res

    @classmethod
    def eval_compare(cls, op: ast.Compare, left, right):
        eval = op.op._op
        res = eval(left, right)
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
            # return Address(0)
            return None
        return None

        raise NotImplementedError(f"Default value for {typ} not implemented")

    @classmethod
    def construct_struct(cls, name, kws):
        StructType = type(name, (object,), kws)
        return StructType()

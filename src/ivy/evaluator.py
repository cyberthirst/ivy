from abc import ABC, abstractmethod
from typing import Any

from vyper.ast import nodes as ast


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


class VyperEvaluator(BaseEvaluator):
    def eval_boolop(self, op, values):
        eval = op.op._op
        res = eval(values)
        return res

    def eval_unaryop(self, op, operand):
        eval = op.op._op
        res = eval(operand)
        return res

    def eval_binop(self, op: ast.BinOp, left: Any, right: Any):
        eval = op.op._op
        res = eval(left, right)
        return res

    def eval_compare(self, op: ast.Compare, left, right):
        eval = op.op._op
        res = eval(left, right)
        return res

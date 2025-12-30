from abc import ABC, abstractmethod
from typing import Optional
import copy

from vyper.ast import nodes as ast

from ivy.types import _Container


class BaseVisitor(ABC):
    def visit(self, node):
        self._on_node(node)
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    def _on_node(self, node) -> None:
        pass

    def _on_branch(self, node, taken: bool) -> None:
        pass

    def _on_boolop(self, node, op: str, evaluated_count: int, result: bool) -> None:
        pass

    def _on_loop(self, node, iteration_count: int) -> None:
        pass

    @abstractmethod
    def set_variable(self, name: str, value, node: Optional[ast.VyperNode] = None):
        pass

    @abstractmethod
    def get_variable(self, name: str, node: Optional[ast.VyperNode] = None):
        pass

    # pass by value
    def deep_copy_visit(self, node):
        ret = self.visit(node)
        if not isinstance(ret, (_Container, tuple)):
            return ret
        # TODO can be further optimized
        # - can avoid deepcopy in ext func return values, exprs which don't
        # retrieve variables, when consumer doesn't modify the value (copy on write), etc
        return copy.deepcopy(ret)


class BaseClassVisitor(ABC):
    @classmethod
    def visit(cls, node, *args):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(cls, method_name, cls.generic_visit_class)
        return visitor(node, *args)

    def generic(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    @classmethod
    def generic_visit_class(cls, node):
        raise Exception(f"No class visit method for {type(node).__name__}")

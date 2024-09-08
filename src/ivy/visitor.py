from abc import ABC, abstractmethod

from ivy.evaluator import BaseEvaluator


class BaseVisitor(ABC):
    evaluator: BaseEvaluator

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    @abstractmethod
    def set_variable(self, name, value):
        pass

    @abstractmethod
    def get_variable(self, name):
        pass

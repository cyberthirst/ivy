from abc import ABC, abstractmethod


class BaseVisitor(ABC):
    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    @abstractmethod
    def set_variable(self, name: str, value):
        pass

    @abstractmethod
    def get_variable(self, name: str):
        pass

    @abstractmethod
    def _assign_target(self, target, value):
        pass


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

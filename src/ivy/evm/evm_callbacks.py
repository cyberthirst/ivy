from abc import ABC, abstractmethod
from typing import Any

from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import ContractFunctionT


class EVMCallbacks(ABC):
    @abstractmethod
    def allocate_storage(self, module_t: ModuleT) -> None:
        pass

    @abstractmethod
    def execute_init_function(self, func_t: ContractFunctionT) -> Any:
        pass

    @abstractmethod
    def dispatch(self):
        pass

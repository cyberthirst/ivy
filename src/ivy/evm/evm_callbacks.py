from abc import ABC, abstractmethod
from typing import Any

from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import ContractFunctionT


class EVMCallbacks(ABC):
    @abstractmethod
    def allocate_variables(self, module_t: ModuleT) -> None:
        pass

    @abstractmethod
    def execute_init_function(self, func_t: ContractFunctionT) -> Any:
        pass

    @abstractmethod
    def dispatch(self):
        pass

    @abstractmethod
    def on_state_committed(self) -> None:
        pass

    def check_call_timeout(self) -> None:
        """Check if the call timeout has been exceeded. No-op by default."""
        pass

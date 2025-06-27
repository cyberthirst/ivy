"""
Adapter to make Ivy's Env work with the ExecutionEnvironment protocol.
"""

from typing import Any, Dict, List, Optional
from ivy.frontend.env import Env
from ivy.frontend.loader import loads_from_solc_json
from ivy.types import Address


class IvyEnvAdapter:
    """Adapter that makes Ivy's Env conform to ExecutionEnvironment protocol."""

    def __init__(self, env: Optional[Env] = None):
        self.env = env or Env()
        self._original_eoa = None

    def deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        constructor_args: Optional[Dict[str, Any]] = None,
        encoded_constructor_args: Optional[bytes] = None,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)

        try:
            deployment_kwargs = {
                "env": self.env,
                "value": value,
            }

            if constructor_args is not None:
                deployment_kwargs["constructor_args"] = constructor_args
            elif encoded_constructor_args is not None:
                deployment_kwargs["encoded_constructor_args"] = encoded_constructor_args

            # Deploy using loads_from_solc_json
            contract = loads_from_solc_json(solc_json, **deployment_kwargs)
            return contract

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def call_contract_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method with python args in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)

        try:
            # Get the method
            method = getattr(contract, method_name)

            # Call with args and kwargs
            result = method(*args, **kwargs)
            return result

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call with calldata in Ivy."""
        # Set sender as eoa if specified
        if sender:
            self._original_eoa = self.env.eoa
            self.env.eoa = Address(sender)

        try:
            result = self.env.message_call(
                to_address=to_address,
                data=data,
                value=value,
            )
            return result

        finally:
            # Restore original eoa
            if self._original_eoa is not None:
                self.env.eoa = self._original_eoa
                self._original_eoa = None

    def set_balance(self, address: str, value: int) -> None:
        """Set balance of an address in Ivy."""
        self.env.set_balance(Address(address), value)

    def get_balance(self, address: str) -> int:
        """Get balance of an address in Ivy."""
        return self.env.get_balance(Address(address))

    def clear_transient_storage(self) -> None:
        """Clear transient storage in Ivy."""
        self.env.clear_transient_storage()

    def get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get storage dump from a contract in Ivy."""
        if hasattr(contract, "storage_dump"):
            return contract.storage_dump()
        return None

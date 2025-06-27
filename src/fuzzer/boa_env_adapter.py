"""
Adapter to make Boa work with the ExecutionEnvironment protocol.
"""

from typing import Any, Dict, List, Optional
import boa
from boa.contracts.vyper.vyper_contract import VyperContract


class BoaEnvAdapter:
    """Adapter that makes Boa conform to ExecutionEnvironment protocol."""

    def __init__(self):
        # Initialize boa if needed
        pass

    def deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        constructor_args: Optional[Dict[str, Any]] = None,
        encoded_constructor_args: Optional[bytes] = None,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source in Boa."""
        # Set sender if specified
        if sender:
            boa.env.set_balance(sender, self.get_balance(sender) + value + 10**18)
            boa.env.eoa = sender

        try:
            if constructor_args is not None:
                # Deploy with python args
                args = constructor_args.get("args", [])
                kwargs = constructor_args.get("kwargs", {})
                contract = boa.loads(source, *args, value=value, **kwargs)
            else:
                # Deploy with default constructor
                contract = boa.loads(source, value=value)

            return contract

        except Exception as e:
            raise e

    def call_contract_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method with python args in Boa."""
        # Set sender if specified
        if sender:
            # Ensure sender has balance
            self.set_balance(
                sender, self.get_balance(sender) + kwargs.get("value", 0) + 10**18
            )
            with boa.env.prank(sender):
                method = getattr(contract, method_name)
                result = method(*args, **kwargs)
        else:
            method = getattr(contract, method_name)
            result = method(*args, **kwargs)

        return result

    def message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call with calldata in Boa."""
        # In boa, we typically don't do raw message calls
        # This would require getting the contract instance and decoding the calldata
        # For now, this is a limitation when using calldata mode with boa
        raise NotImplementedError("Raw message calls not supported in Boa adapter")

    def set_balance(self, address: str, value: int) -> None:
        """Set balance of an address in Boa."""
        boa.env.set_balance(address, value)

    def get_balance(self, address: str) -> int:
        """Get balance of an address in Boa."""
        return boa.env.get_balance(address)

    def clear_transient_storage(self) -> None:
        """Clear transient storage in Boa."""
        # Boa might not have explicit transient storage clearing
        # This would depend on the boa version and features
        if hasattr(boa.env, "clear_transient_storage"):
            boa.env.clear_transient_storage()

    def get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get storage dump from a contract in Boa."""
        if isinstance(contract, VyperContract):
            # Get storage dump from boa contract
            # The exact method might vary based on boa version
            storage = {}
            if hasattr(contract, "_storage"):
                # Try to access internal storage representation
                for slot, value in contract._storage.items():
                    storage[hex(slot)] = hex(value)
            return storage
        return None

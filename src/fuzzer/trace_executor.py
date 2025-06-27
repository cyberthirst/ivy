"""
Trace execution utilities for both replay and fuzzing.

This module provides functions to execute different types of traces
(deployment, call, set_balance, clear_transient_storage) in a reusable way.
These functions are designed to work with different execution environments
(Ivy, Boa) through abstracted interfaces.
"""

from typing import Any, Dict, List, Optional, Union, Protocol
from pathlib import Path

from .export_utils import (
    DeploymentTrace,
    CallTrace,
    SetBalanceTrace,
    ClearTransientStorageTrace,
)
from .runner import DeploymentResult, CallResult


class ExecutionEnvironment(Protocol):
    """Protocol for execution environments (Ivy Env, Boa, etc.)"""

    def deploy_from_source(
        self,
        source: str,
        solc_json: Dict[str, Any],
        constructor_args: Optional[Dict[str, Any]] = None,
        encoded_constructor_args: Optional[bytes] = None,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> Any:
        """Deploy a contract from source"""
        ...

    def call_contract_method(
        self,
        contract: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> Any:
        """Call a contract method with python args"""
        ...

    def message_call(
        self,
        to_address: str,
        data: bytes,
        value: int = 0,
        sender: Optional[str] = None,
    ) -> bytes:
        """Low-level message call with calldata"""
        ...

    def set_balance(self, address: str, value: int) -> None:
        """Set balance of an address"""
        ...

    def get_balance(self, address: str) -> int:
        """Get balance of an address"""
        ...

    def clear_transient_storage(self) -> None:
        """Clear transient storage"""
        ...

    def get_storage_dump(self, contract: Any) -> Optional[Dict[str, Any]]:
        """Get storage dump from a contract"""
        ...


def prepare_deployment_source(
    trace: DeploymentTrace,
    mutated_source: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """
    Prepare source and solc_json for deployment.

    Returns:
        Tuple of (source_code, solc_json)
    """
    # Use mutated source if provided
    source_to_deploy = (
        mutated_source if mutated_source is not None else trace.source_code
    )
    if not source_to_deploy:
        raise ValueError("No source code available for deployment")

    # Always use the original solc_json - we don't need to modify it
    return source_to_deploy, trace.solc_json


def prepare_constructor_args(
    trace: DeploymentTrace,
    mutated_args: Optional[List[Any]] = None,
    mutated_kwargs: Optional[Dict[str, Any]] = None,
    use_python_args: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[bytes], int]:
    """
    Prepare constructor arguments for deployment.

    Returns:
        Tuple of (constructor_args, encoded_constructor_args, value)
    """
    # Determine value
    if mutated_kwargs is not None and "value" in mutated_kwargs:
        value = mutated_kwargs["value"]
    else:
        value = trace.value

    # Determine constructor args
    constructor_args = None
    encoded_constructor_args = None

    if use_python_args and trace.python_args is not None:
        # Use python_args (potentially mutated)
        if mutated_args is not None:
            constructor_args = {"args": mutated_args}
            if mutated_kwargs is not None:
                constructor_args["kwargs"] = {
                    k: v for k, v in mutated_kwargs.items() if k != "value"
                }
        else:
            constructor_args = trace.python_args
    elif trace.calldata:
        # Use encoded calldata
        encoded_constructor_args = bytes.fromhex(trace.calldata)

    return constructor_args, encoded_constructor_args, value


def execute_deployment(
    trace: DeploymentTrace,
    env: ExecutionEnvironment,
    deployed_contracts: Dict[str, Any],
    mutated_source: Optional[str] = None,
    mutated_args: Optional[List[Any]] = None,
    mutated_kwargs: Optional[Dict[str, Any]] = None,
    use_python_args: bool = False,
    collect_storage_dump: bool = False,
) -> DeploymentResult:
    """
    Execute a deployment trace with optional mutations.

    Args:
        trace: The deployment trace to execute
        env: The execution environment (Ivy Env, Boa, etc.)
        deployed_contracts: Dict to store deployed contracts
        mutated_source: Optional mutated source code
        mutated_args: Optional mutated constructor args
        mutated_kwargs: Optional mutated constructor kwargs
        use_python_args: Whether to use python_args instead of calldata

    Returns:
        DeploymentResult with success status and contract instance/error
    """
    # Only support source deployments
    if trace.deployment_type != "source":
        return DeploymentResult(
            success=False,
            error=Exception(f"Unsupported deployment type: {trace.deployment_type}"),
        )

    # Check if we have solc_json
    if not trace.solc_json:
        return DeploymentResult(
            success=False, error=Exception("solc_json is required for deployment")
        )

    # Ensure the deployer account exists with proper balance
    deployer_balance = env.get_balance(trace.deployer)
    if deployer_balance < trace.value:
        # Give the deployer enough balance to deploy
        env.set_balance(trace.deployer, trace.value + 10**18)  # Add 1 ETH extra

    deployment_succeeded = True
    contract = None
    error = None

    try:
        # Prepare source and constructor args
        source_to_deploy, solc_json = prepare_deployment_source(trace, mutated_source)
        constructor_args, encoded_constructor_args, value = prepare_constructor_args(
            trace, mutated_args, mutated_kwargs, use_python_args
        )

        # Deploy the contract
        contract = env.deploy_from_source(
            source=source_to_deploy,
            solc_json=solc_json,
            constructor_args=constructor_args,
            encoded_constructor_args=encoded_constructor_args,
            value=value,
            sender=trace.deployer,
        )

    except Exception as e:
        deployment_succeeded = False
        error = e

    if deployment_succeeded and contract:
        # Store deployed contract
        deployed_contracts[trace.deployed_address] = contract

        # Also store by actual address if different
        contract_address = str(getattr(contract, "address", contract))
        if contract_address != trace.deployed_address:
            deployed_contracts[contract_address] = contract

        # Get storage dump only if requested
        storage_dump = None
        if collect_storage_dump:
            storage_dump = env.get_storage_dump(contract)

        return DeploymentResult(
            success=True, address=contract, storage_dump=storage_dump
        )
    else:
        return DeploymentResult(success=False, error=error)


def execute_call(
    trace: CallTrace,
    env: ExecutionEnvironment,
    deployed_contracts: Dict[str, Any],
    use_python_args: bool = False,
    collect_storage_dump: bool = False,
) -> CallResult:
    """
    Execute a call trace.

    Args:
        trace: The call trace to execute
        env: The execution environment
        deployed_contracts: Dict of deployed contracts
        use_python_args: Whether to use python_args instead of calldata

    Returns:
        CallResult with success status and output/error
    """
    call_args = trace.call_args

    # Extract call parameters
    to_address = call_args["to"]
    sender = call_args["sender"]
    value = call_args["value"]

    # Ensure sender has enough balance
    sender_balance = env.get_balance(sender)
    if sender_balance < value:
        env.set_balance(sender, value + 10**18)

    call_succeeded = True
    output = None
    error = None
    storage_dump = None

    try:
        if (
            use_python_args
            and trace.python_args is not None
            and trace.function_name is not None
        ):
            # Use python_args - call contract method directly
            if to_address in deployed_contracts:
                contract = deployed_contracts[to_address]

                # Call the method with python args
                args = trace.python_args.get("args", [])
                kwargs = trace.python_args.get("kwargs", {})
                kwargs["value"] = value

                output = env.call_contract_method(
                    contract=contract,
                    method_name=trace.function_name,
                    args=args,
                    kwargs=kwargs,
                    sender=sender,
                )

                # Get storage dump after call only if requested
                if collect_storage_dump:
                    storage_dump = env.get_storage_dump(contract)
            else:
                raise Exception(
                    f"Contract at {to_address} not found for python_args call"
                )
        else:
            # Use calldata
            calldata = bytes.fromhex(call_args["calldata"])
            output = env.message_call(
                to_address=to_address,
                data=calldata,
                value=value,
                sender=sender,
            )

            # Try to get storage dump if contract is available and requested
            if collect_storage_dump and to_address in deployed_contracts:
                contract = deployed_contracts[to_address]
                storage_dump = env.get_storage_dump(contract)

    except Exception as e:
        call_succeeded = False
        error = e

    return CallResult(
        success=call_succeeded, output=output, error=error, storage_dump=storage_dump
    )


def execute_set_balance(trace: SetBalanceTrace, env: ExecutionEnvironment) -> None:
    """Execute a set_balance trace."""
    env.set_balance(trace.address, trace.value)


def execute_clear_transient_storage(
    trace: ClearTransientStorageTrace, env: ExecutionEnvironment
) -> None:
    """Execute a clear_transient_storage trace."""
    env.clear_transient_storage()

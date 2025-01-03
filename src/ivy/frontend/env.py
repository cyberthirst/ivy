import random
from typing import Any, Optional, TypeAlias, Union

from vyper import ast as vy_ast

from ivy.vyper_interpreter import VyperInterpreter
from ivy.types import Address
from ivy.evm.evm_state import StateAccess
from ivy.context import ExecutionOutput

# make mypy happy
_AddressType: TypeAlias = Address | str | bytes


class Env:
    _singleton = None
    _random = random.Random("ivy")

    interpreter: VyperInterpreter

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interpreter = VyperInterpreter()
        self.state: StateAccess = self.interpreter.state
        self._aliases = {}
        self.eoa = self.generate_address("eoa")
        self._accounts = []
        self._contracts = {}

    def clear_state(self):
        # TODO should we just clear the EVM state instead of instantiating the itp?
        self.interpreter = VyperInterpreter()
        self.state = self.interpreter.state
        self._aliases = {}
        self.eoa = self.generate_address("eoa")
        self._contracts = {}

    @classmethod
    def get_singleton(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def _get_sender(self, sender=None) -> Address:
        if sender is None:  # TODO add ctx manager to set this
            sender = self.eoa
        if self.eoa is None:
            raise ValueError(f"{self}.eoa not defined!")
        return Address(sender)

    def generate_address(self, alias: Optional[str] = None) -> _AddressType:
        t = Address(self._random.randbytes(20))
        if alias is not None:
            self.alias(t, alias)
        return t

    def alias(self, address, name):
        self._aliases[Address(address).canonical_address] = name

    def register_contract(self, address, obj):
        self._contracts[address.canonical_address] = obj

    def lookup_contract(self, address: _AddressType):
        if address == b"":
            return None
        return self._contracts.get(Address(address).canonical_address)

    def raw_call(
        self,
        to_address: _AddressType = Address(0),
        sender: Optional[_AddressType] = None,
        value: int = 0,
        calldata: Union[bytes, str] = b"",
        is_modifying: bool = True,
    ) -> Any:
        if isinstance(calldata, str):
            assert calldata.startswith("0x")
            calldata = bytes.fromhex(calldata[2:])

        ret = self.execute_code(to_address, sender, value, calldata, is_modifying)

        if ret.is_error:
            raise ret.error

        return ret.output

    # compatability alias for vyper env
    def message_call(self, to_address: _AddressType, data: bytes):
        return self.raw_call(to_address, calldata=data)

    def get_balance(self, address: _AddressType) -> int:
        return self.state.get_balance(address)

    def set_balance(self, address: _AddressType, value: int):
        self.state.set_balance(Address(address), value)

    def get_account(self, address: _AddressType):
        return self.state.get_account(Address(address))

    @property
    def accounts(self):
        if not self._accounts:
            for i in range(10):
                self._accounts.append(self.generate_address(f"account{i}"))

        return self._accounts

    @property
    def deployer(self):
        return self.eoa

    @property
    def timestamp(self):
        return self.state.env.time

    def deploy(
        self,
        module: vy_ast.Module,
        raw_args: bytes = None,
        sender: Optional[_AddressType] = None,
        value: int = 0,
    ) -> tuple[Address, ExecutionOutput]:
        sender = self._get_sender(sender)

        contract_address, execution_output = self.interpreter.execute(
            sender=sender,
            to=b"",
            module=module,
            value=value,
            calldata=raw_args,
        )

        return contract_address, execution_output

    def execute_code(
        self,
        to_address: _AddressType = Address(0),
        sender: Optional[_AddressType] = None,
        value: int = 0,
        calldata: bytes = b"",
        is_modifying: bool = True,
    ) -> ExecutionOutput:
        sender = self._get_sender(sender)

        to = Address(to_address)

        is_static = not is_modifying

        execution_output = self.interpreter.execute(
            sender=sender,
            to=to,
            value=value,
            calldata=calldata,
            is_static=is_static,
        )

        return execution_output

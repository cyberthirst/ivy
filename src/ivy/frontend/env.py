import random
from typing import Any, Optional, TypeAlias, Union

from vyper import ast as vy_ast

from ivy.vyper_interpreter import VyperInterpreter
from ivy.types import Address

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
        self._aliases = {}
        self.eoa = self.generate_address("eoa")
        self._accounts = []

    def clear_state(self):
        self.interpreter = VyperInterpreter()
        self._aliases = {}
        self.eoa = self.generate_address("eoa")

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

        return ret

    def get_balance(self, address: _AddressType) -> int:
        return self.interpreter.get_balance(Address(address))

    def set_balance(self, address: _AddressType, value: int):
        self.interpreter.set_balance(Address(address), value)

    @property
    def accounts(self):
        if not self._accounts:
            for i in range(10):
                self._accounts.append(self.generate_address(f"account{i}"))

        return self._accounts

    @property
    def deployer(self):
        return self.eoa

    def deploy(
        self,
        module: vy_ast.Module,
        raw_args: bytes = None,
        sender: Optional[_AddressType] = None,
        value: int = 0,
    ):
        sender = self._get_sender(sender)

        contract_address = self.interpreter.execute_tx(
            sender=sender,
            to=b"",
            module=module,
            value=value,
            calldata=raw_args,
        )

        return contract_address

    def execute_code(
        self,
        to_address: _AddressType = Address(0),
        sender: Optional[_AddressType] = None,
        value: int = 0,
        calldata: bytes = b"",
        is_modifying: bool = True,
    ) -> Any:
        sender = self._get_sender(sender)

        to = Address(to_address)

        is_static = not is_modifying

        ret = self.interpreter.execute_tx(
            sender=sender,
            to=to,
            value=value,
            calldata=calldata,
            is_static=is_static,
        )

        return ret

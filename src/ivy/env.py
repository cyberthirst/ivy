import random

from typing import Any, Optional, TypeAlias

from eth_account import Account
from eth_typing import Address as PYEVM_Address  # it's just bytes.
import eth.constants as constants

from vyper import ast as vy_ast

from titanoboa.boa.util.abi import Address

from ivy.interpreter import Interpreter, VyperEVM, EVM

# make mypy happy
_AddressType: TypeAlias = Address | str | bytes | PYEVM_Address


class Env:
    _singleton = None
    _random = random.Random("ivy")

    def __init__(
        self,
        accounts: dict[str, Account] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._accounts = accounts or {}
        self.evm: EVM = VyperEVM()
        self.interpreter = Interpreter(self.evm)
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

    def deploy(
        self,
        module: vy_ast.Module,
        *args: Any,
        raw_args: bytes = None,
        sender: Optional[_AddressType] = None,
        value: int = 0,
    ):
        sender = self._get_sender(sender)

        target_address = self.interpreter.generate_create_address(sender)

        origin = sender

        self.interpreter.deploy(
            sender,
            origin,
            target_address,
            module,
            value,
            *args,
            raw_args=raw_args,
        )

        return target_address

    def execute_code(
        self,
        func_name: str,
        *args: Any,
        to_address: _AddressType = constants.ZERO_ADDRESS,
        sender: Optional[_AddressType] = None,
        value: int = 0,
        raw_args: bytes = b"",
        is_modifying: bool = True,
    ) -> Any:
        sender = self._get_sender(sender)

        to = Address(to_address)

        code = self.interpreter.get_code(to)

        is_static = not is_modifying

        ret = self.interpreter.execute_code(
            sender,
            to,
            value,
            code,
            func_name,
            *args,
            raw_args=raw_args,
            is_static=is_static,
        )

        return ret

import random

from typing import Any, Optional, TypeAlias

from eth_account import Account
from eth_typing import Address as PYEVM_Address  # it's just bytes.
import eth.constants as constants

from vyper import ast as vy_ast

from titanoboa.boa.util.abi import Address

from evm import Interpreter
from vyper_contract import VyperDeployer

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
        self.eoa = self.generate_address("eoa")
        self.evm = Interpreter()
        self._aliases = {}
        self.deployer_class = VyperDeployer


    @classmethod
    def get_singleton(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton


    def _get_sender(self, sender=None) -> Address:
        if sender is None: # TODO add ctx manager to set this
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
            args: bytes = None,
            sender: Optional[_AddressType] = None,
            value: int = 0,
    ):
        sender = self._get_sender(sender)

        target_address = self.evm.generate_create_address(sender)

        origin = sender

        self.evm.deploy(
            sender=sender,
            origin=origin,
            target_address=target_address,
            value=value,
            module=module,
            args=args,
        )

        return target_address


    def execute_code(
        self,
        to_address: _AddressType = constants.ZERO_ADDRESS,
        sender: Optional[_AddressType] = None,
        value: int = 0,
        data: bytes = b"",
        is_modifying: bool = True,
    ) -> Any:

        sender = self._get_sender(sender)

        to = Address(to_address)

        code = self.evm.get_code(to)

        is_static = not is_modifying

        ret = self.evm.execute_code(
            sender=sender,
            to=to,
            value=value,
            code=code,
            data=data,
            is_static=is_static,
        )

        return ret
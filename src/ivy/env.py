from typing import Any, Optional, TypeAlias

from eth_account import Account
from eth_typing import Address as PYEVM_Address  # it's just bytes.
import eth.constants as constants

from titanoboa.boa.util.abi import Address
from titanoboa.boa.contracts.vyper.vyper_contract import IvyDeployer

from evm import Interpreter


# make mypy happy
_AddressType: TypeAlias = Address | str | bytes | PYEVM_Address


class Env:

    _singleton = None

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
        self.deployer_class = IvyDeployer


    @classmethod
    def get_singleton(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton


    def _get_sender(self, sender=None) -> Address:
        if sender is None:
            sender = self.eoa
        if self.eoa is None:
            raise ValueError(f"{self}.eoa not defined!")
        return Address(sender)

    def generate_address(self, alias: Optional[str] = None) -> _AddressType:
        t = Address(self._random.randbytes(20))
        if alias is not None:
            self.alias(t, alias)

    def alias(self, address, name):
        self._aliases[Address(address).canonical_address] = name

    # override
    def deploy(
            self,
            args: bytes = None,
            sender: Optional[_AddressType] = None,
            gas: Optional[int] = None,
            value: int = 0,
            bytecode: bytes = b"",
            start_pc: int = 0,  # TODO: This isn't used
            # override the target address:
            override_address: Optional[_AddressType] = None,
            # the calling vyper contract
            contract: Any = None,
    ):
        sender = self._get_sender(sender)

        if override_address is None:
            target_address = self.evm.generate_create_address(sender)
        else:
            target_address = Address(override_address)

        origin = sender  # XXX: consider making this parameterizable
        # TODO interpretet the constructor with the args
        #computation = self.evm.deploy_code(
        #    sender=sender,
        #    origin=origin,
        #    target_address=target_address,
        #    gas=gas,
        #    gas_price=self.get_gas_price(),
        #    value=value,
        #    bytecode=bytecode,
        #)
        #
        #if self._coverage_enabled:
        #    self._trace_computation(computation, contract)

        return target_address


    def execute_code(
        self,
        to_address: _AddressType = constants.ZERO_ADDRESS,
        sender: Optional[_AddressType] = None,
        gas: Optional[int] = None,
        value: int = 0,
        data: bytes = b"",
        override_bytecode: Optional[bytes] = None,
        ir_executor: Any = None,
        is_modifying: bool = True,
        start_pc: int = 0,
        fake_codesize: Optional[int] = None,
        contract: Any = None,  # the calling VyperContract
    ) -> Any:

        sender = self._get_sender(sender)

        to = Address(to_address)

        bytecode = override_bytecode
        if override_bytecode is None:
            bytecode = self.evm.get_code(to)

        is_static = not is_modifying
        ret = self.evm.execute_code(
            sender=sender,
            to=to,
            gas=gas,
            value=value,
            bytecode=bytecode,
            data=data,
            is_static=is_static,
            fake_codesize=fake_codesize,
            start_pc=start_pc,
            ir_executor=ir_executor,
            contract=contract,
        )

        return ret
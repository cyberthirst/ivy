from typing import Union

from eth_utils import to_canonical_address, to_checksum_address
from eth_typing.evm import Address as EthAddress

from vyper.semantics.types import FlagT, StructT

from ivy.utils import lrudict


# adapted from titanoboa: https://github.com/vyperlang/titanoboa/blob/bedd49e5a4c1e79a7d12c799e42a23a9dc449395/boa/util/abi.py#L20
# inherit from `str` so that users can compare with regular hex string
# addresses
class Address(str):
    # converting between checksum and canonical addresses is a hotspot;
    # this class contains both and caches recently seen conversions
    __slots__ = ("canonical_address",)
    _cache = lrudict(1024)

    canonical_address: EthAddress

    def __new__(cls, address):
        if isinstance(address, Address):
            return address

        if isinstance(address, bytes):
            if len(address) > 20:
                raise ValueError(f"Invalid address length: {len(address)} bytes.")
            address = address.rjust(20, b"\x00").hex()

        if isinstance(address, int):
            address = hex(address)[2:]
            address = address.rjust(40, "0")

        try:
            return cls._cache[address]
        except KeyError:
            pass

        checksum_address = to_checksum_address(address)
        self = super().__new__(cls, checksum_address)
        self.canonical_address = to_canonical_address(address)
        cls._cache[address] = self
        return self

    def __repr__(self):
        checksum_addr = super().__repr__()
        return f"Address({checksum_addr})"


class Struct(dict):
    def __init__(self, typ: StructT, kws):
        self.typ = typ
        super().__init__(kws)


class Flag:
    def __init__(self, typ: FlagT, source: Union[str, int]):
        self.typ = typ
        self.mask = (1 << len(typ._flag_members)) - 1
        if isinstance(source, str):
            if source not in typ._flag_members:
                raise AttributeError(f"'{typ.name}' flag has no member '{source}'")
            value = 2 ** typ._flag_members[source]
        else:
            assert source >> len(typ._flag_members) == 0
            value = source
        self.value = value & self.mask

    def __eq__(self, other):
        return self.value == other.value

    def __or__(self, other):
        return Flag(self.typ, self.value | other.value)

    def __and__(self, other):
        return Flag(self.typ, self.value & other.value)

    def __xor__(self, other):
        return Flag(self.typ, self.value ^ other.value)

    def __invert__(self):
        return Flag(self.typ, (~self.value) & self.mask)

    def __contains__(self, other):
        return (self.value & other.value) != 0

    def __hash__(self):
        return hash(self.value)

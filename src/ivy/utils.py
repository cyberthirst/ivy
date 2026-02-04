import re

import rlp
from eth_utils import keccak

from vyper.utils import method_id
from vyper.semantics.types.subscriptable import TupleT


def compute_call_abi_data(func_t, num_kwargs):
    sig_kwargs = func_t.keyword_args[:num_kwargs]
    sig_args = func_t.positional_args + sig_kwargs

    calldata_args_t = TupleT(list(arg.typ for arg in sig_args))

    sig = func_t.name + calldata_args_t.abi_type.selector_name()

    selector = method_id(sig)

    return selector, calldata_args_t


# lrudict from titanoboa: https://github.com/vyperlang/titanoboa/blob/bedd49e5a4c1e79a7d12c799e42a23a9dc449395/boa/util/lrudict.py
class lrudict(dict):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def __getitem__(self, k):
        val = super().__getitem__(k)
        del self[k]  # move to the front of the queue
        super().__setitem__(k, val)
        return val

    def __setitem__(self, k, val):
        if len(self) == self.n and k not in self:
            del self[next(iter(self))]
        super().__setitem__(k, val)

    # set based on a lambda
    def setdefault_lambda(self, k, fn):
        try:
            return self[k]
        except KeyError:
            self[k] = (ret := fn(k))
            return ret


# version detection from titanoboa: https://github.com/vyperlang/titanoboa/blob/bedd49e5a4c1e79a7d12c799e42a23a9dc449395/boa/contracts/vvm/vvm_contract.py#L9
VERSION_RE = re.compile(r"\s*#\s*(pragma\s+version|@version)\s+(\d+\.\d+\.\d+)")


def _detect_version(source_code: str):
    res = VERSION_RE.findall(source_code)
    if len(res) < 1:
        return None
    # TODO: handle len(res) > 1
    return res[0][1]


def compute_contract_address(address, nonce):
    computed_address = keccak(rlp.encode([address, nonce]))
    canonical_address = computed_address[-20:]
    return canonical_address


def compute_create2_address(sender: bytes, salt: bytes, init_code: bytes) -> bytes:
    """
    Compute CREATE2 contract address per EIP-1014.

    address = keccak256(0xff ++ sender ++ salt ++ keccak256(init_code))[-20:]
    """
    assert len(sender) == 20, f"sender must be 20 bytes, got {len(sender)}"
    assert len(salt) == 32, f"salt must be 32 bytes, got {len(salt)}"

    init_code_hash = keccak(init_code)
    preimage = b"\xff" + sender + salt + init_code_hash
    computed_address = keccak(preimage)
    return computed_address[-20:]


def calldata_slice(payload: bytes, start: int, length: int) -> bytes:
    """Slice calldata with zero-extension beyond the payload length."""
    if length == 0:
        return b""
    if start < 0:
        raise ValueError(f"OOB {start}")

    payload_len = len(payload)
    end = start + length

    if start >= payload_len:
        return b"\x00" * length
    if end > payload_len:
        available = payload[start:payload_len]
        padding = b"\x00" * (end - payload_len)
        return available + padding
    return payload[start:end]


def _trunc_div(n: int, d: int) -> int:
    """
    Integer division rounded **toward 0** for all sign combinations.
    Caller must ensure d != 0.
    """
    q = abs(n) // abs(d)  # non‑negative → already truncated
    return -q if (n < 0) ^ (d < 0) else q

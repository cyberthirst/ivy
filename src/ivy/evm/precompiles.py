import hashlib

import coincurve
from eth_utils import keccak

from ivy.types import Address
from ivy.utils import calldata_slice


SECP256K1N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def precompile_ecrecover(data: bytes) -> bytes:
    # Pad input to 128 bytes (right-pad with zeros per Yellow Paper)
    data = data.ljust(128, b"\x00")

    # Parse input fields
    msg_hash = data[0:32]
    v = int.from_bytes(data[32:64], "big")
    r = int.from_bytes(data[64:96], "big")
    s = int.from_bytes(data[96:128], "big")

    # Validate v, r, s per Yellow Paper
    if v not in (27, 28):
        return b""
    if not (0 < r < SECP256K1N):
        return b""
    if not (0 < s < SECP256K1N):
        return b""

    # Build recoverable signature: r(32) + s(32) + recovery_id(1)
    r_bytes = r.to_bytes(32, "big")
    s_bytes = s.to_bytes(32, "big")
    recovery_id = v - 27
    signature = r_bytes + s_bytes + bytes([recovery_id])

    try:
        pubkey = coincurve.PublicKey.from_signature_and_message(
            signature, msg_hash, hasher=None
        )
        # Get uncompressed format (65 bytes: 0x04 + x + y), strip the 0x04 prefix
        pubkey_bytes = pubkey.format(compressed=False)[1:]
    except Exception:
        return b""

    # Hash to address: keccak256(pubkey)[12:32], left-padded to 32 bytes
    address = keccak(pubkey_bytes)[12:32]
    return address.rjust(32, b"\x00")


def precompile_identity(data: bytes) -> bytes:
    return data


def precompile_sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def precompile_ripemd160(data: bytes) -> bytes:
    hash_bytes = hashlib.new("ripemd160", data).digest()
    return hash_bytes.rjust(32, b"\x00")


def precompile_modexp(data: bytes) -> bytes:
    base_len = int.from_bytes(calldata_slice(data, 0, 32), "big")
    exp_len = int.from_bytes(calldata_slice(data, 32, 32), "big")
    mod_len = int.from_bytes(calldata_slice(data, 64, 32), "big")

    if base_len == 0 and mod_len == 0:
        return b""

    base = int.from_bytes(calldata_slice(data, 96, base_len), "big")
    exp_start = 96 + base_len
    exp = int.from_bytes(calldata_slice(data, exp_start, exp_len), "big")
    mod_start = exp_start + exp_len
    modulus = int.from_bytes(calldata_slice(data, mod_start, mod_len), "big")

    if modulus == 0:
        return b"\x00" * mod_len

    return pow(base, exp, modulus).to_bytes(mod_len, "big")


PRECOMPILE_REGISTRY = {
    Address(0x1): precompile_ecrecover,
    Address(0x2): precompile_sha256,
    Address(0x3): precompile_ripemd160,
    Address(0x4): precompile_identity,
    Address(0x5): precompile_modexp,
}

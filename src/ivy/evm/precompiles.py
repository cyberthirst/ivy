import coincurve
from eth_utils import keccak

from ivy.types import Address


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


PRECOMPILE_REGISTRY = {
    Address(0x1): precompile_ecrecover,
    Address(0x4): precompile_identity,
}

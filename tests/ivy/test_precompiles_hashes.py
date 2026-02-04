import hashlib

from ivy.frontend.loader import loads


SHA256_ADDRESS = "0x0000000000000000000000000000000000000002"
RIPEMD160_ADDRESS = "0x0000000000000000000000000000000000000003"
MODEXP_ADDRESS = "0x0000000000000000000000000000000000000005"


def test_builtin_sha256_bytes_and_string():
    src = """
@external
def from_bytes(_value: Bytes[100]) -> bytes32:
    return sha256(_value)

@external
def from_string(_value: String[100]) -> bytes32:
    return sha256(_value)
    """
    c = loads(src)
    expected = hashlib.sha256(b"potato").digest()
    assert c.from_bytes(b"potato") == expected
    assert c.from_string("potato") == expected


def test_precompiles_sha256_ripemd160_modexp():
    src = f"""
@external
def sha256_pc(data: Bytes[256]) -> Bytes[32]:
    return raw_call({SHA256_ADDRESS}, data, max_outsize=32)

@external
def ripemd160_pc(data: Bytes[256]) -> Bytes[32]:
    return raw_call({RIPEMD160_ADDRESS}, data, max_outsize=32)

@external
def modexp_pc(data: Bytes[256]) -> Bytes[32]:
    return raw_call({MODEXP_ADDRESS}, data, max_outsize=32)

    """
    c = loads(src)

    sha_data = b"potato"
    assert c.sha256_pc(sha_data) == hashlib.sha256(sha_data).digest()

    ripemd_data = b"hello"
    ripemd = hashlib.new("ripemd160", ripemd_data).digest()
    assert c.ripemd160_pc(ripemd_data) == ripemd.rjust(32, b"\x00")

    base = 5
    exponent = 3
    modulus = 13
    base_len = 1
    exp_len = 1
    mod_len = 32
    modexp_data = (
        base_len.to_bytes(32, "big")
        + exp_len.to_bytes(32, "big")
        + mod_len.to_bytes(32, "big")
        + base.to_bytes(base_len, "big")
        + exponent.to_bytes(exp_len, "big")
        + modulus.to_bytes(mod_len, "big")
    )
    assert c.modexp_pc(modexp_data) == (8).to_bytes(mod_len, "big")

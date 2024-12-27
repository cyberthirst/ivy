from ivy.types import Address


def precompile_identity(data: bytes) -> bytes:
    return data


PRECOMPILE_REGISTRY = {
    Address(0x4): precompile_identity,
}

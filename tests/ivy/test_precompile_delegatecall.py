"""
Regression tests for delegatecall to precompiles.

Bug: Ivy was checking message.to instead of message.code_address for precompile
detection. For delegatecall, message.to is the caller's address (execution context),
not the precompile address. This caused delegatecall to precompiles to silently
fail and return empty bytes.

Fix: Check message.code_address which contains the actual target address being
delegated to (the precompile).
"""

from ivy.frontend.loader import loads


ECRECOVER_ADDRESS = "0x0000000000000000000000000000000000000001"
IDENTITY_ADDRESS = "0x0000000000000000000000000000000000000004"

# Test vector for ecrecover (verified with coincurve)
H = bytes.fromhex(
    "6c9c5e133b8aafb4d3c4a1f2e5e628f9da0c4e827b25bb3e1ec5bf0c7adc7055"
)
V = 28
R = 78616903610863619048312090996582975134846548711908124330623869860028714000524
S = 37668412685023658579322665379076573648491476554461132093879548014047408844408
EXPECTED_ADDR = bytes.fromhex("ef4ab0dac7136e1f44add67c6288a71629b2bb87")


def u256(x: int) -> bytes:
    return x.to_bytes(32, "big")


# =============================================================================
# Identity precompile (0x04) delegatecall tests
# =============================================================================


def test_identity_precompile_delegatecall():
    """Delegatecall to identity precompile should return input unchanged."""
    src = f"""
@external
def foo(input: Bytes[32]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32,
        is_delegate_call=True
    )
    return output
    """
    c = loads(src)
    input_data = b"Hello, World!"
    result = c.foo(input_data)
    assert result == input_data


def test_identity_precompile_delegatecall_larger_input():
    """Delegatecall to identity with larger input."""
    src = f"""
@external
def foo(input: Bytes[64]) -> Bytes[64]:
    output: Bytes[64] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=64,
        is_delegate_call=True
    )
    return output
    """
    c = loads(src)
    input_data = b"A" * 64
    result = c.foo(input_data)
    assert result == input_data


def test_identity_precompile_delegatecall_empty_input():
    """Delegatecall to identity with empty input."""
    src = f"""
@external
def foo() -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        b"",
        max_outsize=32,
        is_delegate_call=True
    )
    return output
    """
    c = loads(src)
    result = c.foo()
    assert result == b""


def test_identity_precompile_regular_call():
    """Sanity check: regular call to identity precompile still works."""
    src = f"""
@external
def foo(input: Bytes[32]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32
    )
    return output
    """
    c = loads(src)
    input_data = b"Hello, World!"
    result = c.foo(input_data)
    assert result == input_data


def test_identity_precompile_staticcall():
    """Staticcall to identity precompile should work."""
    src = f"""
@external
def foo(input: Bytes[32]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32,
        is_static_call=True
    )
    return output
    """
    c = loads(src)
    input_data = b"Hello, World!"
    result = c.foo(input_data)
    assert result == input_data


# =============================================================================
# Ecrecover precompile (0x01) delegatecall tests
# =============================================================================


def test_ecrecover_precompile_delegatecall():
    """Delegatecall to ecrecover precompile should recover address."""
    src = f"""
@external
def foo(data: Bytes[128]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {ECRECOVER_ADDRESS},
        data,
        max_outsize=32,
        is_delegate_call=True
    )
    return output
    """
    c = loads(src)
    calldata = H + u256(V) + u256(R) + u256(S)
    result = c.foo(calldata)
    expected = b"\x00" * 12 + EXPECTED_ADDR
    assert result == expected


def test_ecrecover_precompile_delegatecall_invalid_v():
    """Delegatecall to ecrecover with invalid v returns empty."""
    src = f"""
@external
def foo(data: Bytes[128]) -> Bytes[32]:
    success: bool = False
    result: Bytes[32] = b""
    success, result = raw_call(
        {ECRECOVER_ADDRESS},
        data,
        max_outsize=32,
        is_delegate_call=True,
        revert_on_failure=False
    )
    return result
    """
    c = loads(src)
    # v = 29 is invalid (must be 27 or 28)
    calldata = H + u256(29) + u256(R) + u256(S)
    result = c.foo(calldata)
    assert result == b""


def test_ecrecover_precompile_regular_call():
    """Sanity check: regular call to ecrecover precompile still works."""
    src = f"""
@external
def foo(data: Bytes[128]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {ECRECOVER_ADDRESS},
        data,
        max_outsize=32
    )
    return output
    """
    c = loads(src)
    calldata = H + u256(V) + u256(R) + u256(S)
    result = c.foo(calldata)
    expected = b"\x00" * 12 + EXPECTED_ADDR
    assert result == expected


# =============================================================================
# Context preservation tests
# =============================================================================


def test_delegatecall_precompile_preserves_storage_context():
    """
    Delegatecall to precompile should preserve storage context.

    This test verifies that even though we're delegatecalling to a precompile,
    the storage context (self) is still our contract, not the precompile address.
    """
    src = f"""
stored_result: public(Bytes[32])

@external
def foo(input: Bytes[32]) -> Bytes[32]:
    output: Bytes[32] = raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32,
        is_delegate_call=True
    )
    # Store result to prove we're in our own storage context
    self.stored_result = output
    return output
    """
    c = loads(src)
    input_data = b"test data"
    result = c.foo(input_data)
    assert result == input_data
    assert c.stored_result() == input_data


def test_delegatecall_to_contract_calling_precompile():
    """Contract A calls B via delegatecall, B calls precompile via regular call."""
    src_library = f"""
@external
def identity_via_call(input: Bytes[32]) -> Bytes[32]:
    return raw_call(
        {IDENTITY_ADDRESS},
        input,
        max_outsize=32
    )
    """

    src_caller = """
interface Library:
    def identity_via_call(input: Bytes[32]) -> Bytes[32]: nonpayable

@external
def foo(lib: address, input: Bytes[32]) -> Bytes[32]:
    # Call library function (not delegatecall to precompile directly)
    return extcall Library(lib).identity_via_call(input)
    """

    library = loads(src_library)
    caller = loads(src_caller)

    input_data = b"nested test"
    result = caller.foo(library, input_data)
    assert result == input_data

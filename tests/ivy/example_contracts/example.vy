struct S:
    a: uint256
    b: uint256

@external
def foo() -> uint256:
    s: S = S(a=1, b=2)
    return s.a
from ivy.frontend.loader import loads


def test_flag_deepcopy_in_struct_augassign():
    src = """
flag Bar:
    BAD
    BAK
    BAZ

struct Foo:
    a: uint256
    b: DynArray[Bar, 3]

@internal
def get_flag() -> Bar:
    return Bar.BAD

@external
def bar(x: Foo) -> Foo:
    y: Foo = x
    y.b[0] |= self.get_flag()
    return y
    """
    c = loads(src)

    result = c.bar([123, [2, 4, 1]])

    assert result == (123, [3, 4, 1])

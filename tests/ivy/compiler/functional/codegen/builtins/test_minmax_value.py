import pytest

from vyper.semantics.types import DecimalT, IntegerT


@pytest.mark.parametrize("typ", sorted(IntegerT.all() + (DecimalT(),)))
@pytest.mark.parametrize("op", ("min_value", "max_value"))
def test_minmax_value(get_contract, op, typ):
    code = f"""
@external
def foo() -> {typ}:
    return {op}({typ})
    """
    c = get_contract(code)

    lo, hi = typ.int_bounds
    if op == "min_value":
        assert c.foo() == lo
    elif op == "max_value":
        assert c.foo() == hi

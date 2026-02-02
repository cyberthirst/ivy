from vyper.semantics.data_locations import DataLocation

from fuzzer.storage_normalizer import normalize_storage_dump


def test_normalize_transient_storage_dump_defaults(get_contract):
    src = """
t: transient(uint256)
s: transient(String[8])
b: transient(Bytes[8])
arr: transient(uint256[2])

@external
def touch():
    assert self.t == 0
    assert len(self.s) == 0
    assert len(self.b) == 0
    assert self.arr[0] == 0
    """
    c = get_contract(src)
    c.touch()

    dump = c.transient_storage_dump()
    assert dump["t"] == 0
    assert dump["s"] == ""
    assert dump["b"] == b""
    assert dump["arr"] == [0, 0]

    normalized = normalize_storage_dump(dump, c, location=DataLocation.TRANSIENT)
    assert normalized == {}

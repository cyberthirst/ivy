# NOTICE: decoder taken from vyper: https://github.com/vyperlang/vyper/blob/master/tests/functional/builtins/codegen/abi_decode.py
# we follow the decoder almost 1:1, however for certain abi types we return the decoded value via a different type
# it's done for better compatbility with ivy's data model
#   - structs get decoded as tuples, ivy returns them as struct objects
#   - boolean get decoded as ints, ivy returns them as bools

from typing import TYPE_CHECKING, Iterable

from eth_utils import to_checksum_address

from vyper.abi_types import (
    ABI_Address,
    ABI_Bool,
    ABI_Bytes,
    ABI_BytesM,
    ABI_DynamicArray,
    ABI_GIntM,
    ABI_StaticArray,
    ABI_String,
    ABI_Tuple,
    ABIType,
)
from vyper.utils import int_bounds, unsigned_to_signed
from vyper.semantics.types import (
    VyperType,
    StructT,
    TupleT,
    SArrayT,
    DArrayT,
    FlagT,
    DecimalT,
)

from ivy.types import Flag, Struct, StaticArray, DynamicArray, VyperDecimal

if TYPE_CHECKING:
    from vyper.semantics.types import VyperType


class DecodeError(Exception):
    pass


def _strict_slice(payload, start, length, from_calldata=False):
    if start < 0:
        raise DecodeError(f"OOB {start}")

    end = start + length
    payload_len = len(payload)

    if from_calldata:
        # For calldata, zero-extend if reading beyond the payload
        if start >= payload_len:
            # Entirely out of bounds - return zeros
            return b"\x00" * length
        elif end > payload_len:
            # Partially out of bounds - return available data + zeros
            available = payload[start:payload_len]
            padding = b"\x00" * (end - payload_len)
            return available + padding
        else:
            # Within bounds
            return payload[start:end]
    else:
        # For regular payloads, enforce strict bounds
        if end > payload_len:
            raise DecodeError(f"OOB {start} + {length} (=={end}) > {payload_len}")
        return payload[start:end]


def _read_int(payload, ofst, from_calldata=False):
    return int.from_bytes(
        _strict_slice(payload, ofst, 32, from_calldata), byteorder="big"
    )


# TODO maybe split into 2 decoders - one which will decode into ivy
# one which will decode into vyper
# vyper abi_decode spec implementation
def abi_decode(
    typ: VyperType, payload: bytes, ivy_compat: bool = True, from_calldata: bool = False
):
    abi_t = typ.abi_type

    lo, hi = abi_t.static_size(), abi_t.size_bound()

    payload_len = len(payload)
    if from_calldata:
        if not (lo <= payload_len):
            raise DecodeError(f"bad calldata size {lo}>{payload_len}")
    else:
        if not (lo <= payload_len <= hi):
            raise DecodeError(f"bad payload size {lo}, {payload_len}, {hi}")

    return _decode_r(abi_t, typ, 0, payload, ivy_compat, from_calldata)


def _decode_r(
    abi_t: ABIType,
    typ: VyperType,
    current_offset: int,
    payload: bytes,
    ivy_compat: bool,
    from_calldata: bool,
):
    if isinstance(abi_t, ABI_Tuple):
        assert isinstance(typ, StructT) or isinstance(typ, TupleT)
        member_typs = typ.tuple_members()
        res = tuple(
            _decode_multi_r(
                abi_t.subtyps,
                member_typs,
                current_offset,
                payload,
                ivy_compat,
                from_calldata,
            )
        )
        if not ivy_compat:
            return res
        if isinstance(typ, StructT):
            kws = dict(zip(typ.tuple_keys(), res))
            return Struct(typ, kws)
        return res

    if isinstance(abi_t, ABI_StaticArray):
        assert isinstance(typ, SArrayT)
        n = abi_t.m_elems
        abi_subtyps = [abi_t.subtyp] * n
        subtyps = [typ.subtype] * n
        res = _decode_multi_r(
            abi_subtyps, subtyps, current_offset, payload, ivy_compat, from_calldata
        )
        if not ivy_compat:
            return res
        # TODO construct the dict in callee
        return StaticArray(typ, {i: res[i] for i in range(n)})

    if isinstance(abi_t, ABI_DynamicArray):
        assert isinstance(typ, DArrayT)
        bound = abi_t.elems_bound

        n = _read_int(payload, current_offset, from_calldata)
        if n > bound:
            raise DecodeError("Dynarray too large")

        # offsets in dynarray start from after the length word
        current_offset += 32
        abi_subtyps = [abi_t.subtyp] * n
        subtyps = [typ.subtype] * n
        res = _decode_multi_r(
            abi_subtyps, subtyps, current_offset, payload, ivy_compat, from_calldata
        )
        if not ivy_compat:
            return res
        # TODO construct the dict in callee
        return DynamicArray(typ, {i: res[i] for i in range(n)})

    # sanity check
    assert not abi_t.is_complex_type()

    if isinstance(abi_t, ABI_Bytes):
        bound = abi_t.bytes_bound
        length = _read_int(payload, current_offset, from_calldata)
        if length > bound:
            raise DecodeError("bytes too large")

        current_offset += 32  # size of length word
        ret = _strict_slice(payload, current_offset, length, from_calldata)

        # abi string doesn't actually define string decoder, so we
        # just bytecast the output
        if isinstance(abi_t, ABI_String):
            # match eth-stdlib, since that's what we check against
            ret = ret.decode(errors="surrogateescape")

        return ret

    # sanity check
    assert not abi_t.is_dynamic()

    if isinstance(abi_t, ABI_GIntM):
        ret = _read_int(payload, current_offset, from_calldata)

        # handle signedness
        if abi_t.signed:
            ret = unsigned_to_signed(ret, 256, strict=True)

        # bounds check
        lo, hi = int_bounds(signed=abi_t.signed, bits=abi_t.m_bits)
        if not (lo <= ret <= hi):
            u = "" if abi_t.signed else "u"
            raise DecodeError(f"invalid {u}int{abi_t.m_bits}")

        if isinstance(abi_t, ABI_Address):
            from ivy.types import Address
            return Address(to_checksum_address(ret.to_bytes(20, "big")))

        if isinstance(abi_t, ABI_Bool):
            if ret not in (0, 1):
                raise DecodeError("invalid bool")
            return True if ret == 1 else False

        if isinstance(typ, FlagT):
            bits = len(typ._flag_members)
            if ret >> bits > 0:
                raise DecodeError(f"flag value out of bounds {ret}")
            if not ivy_compat:
                return ret
            return Flag(typ, ret)

        if isinstance(typ, DecimalT):
            if not ivy_compat:
                return ret

            return VyperDecimal(ret, scaled=True)

        return ret

    if isinstance(abi_t, ABI_BytesM):
        ret = _strict_slice(payload, current_offset, 32, from_calldata)
        m = abi_t.m_bytes
        assert 1 <= m <= 32  # internal sanity check
        # BytesM is right-padded with zeroes
        if ret[m:] != b"\x00" * (32 - m):
            raise DecodeError(f"invalid bytes{m}")
        return ret[:m]

    raise RuntimeError("unreachable")


def _decode_multi_r(
    abi_typs: Iterable[ABIType],
    typs: Iterable[VyperType],
    outer_offset: int,
    payload: bytes,
    ivy_compat: bool,
    from_calldata: bool,
) -> list:
    ret = []
    static_ofst = outer_offset

    assert len(abi_typs) == len(typs)

    for abi_sub_t, sub_t in zip(abi_typs, typs):
        if abi_sub_t.is_dynamic():
            # "head" terminology from abi spec
            head = _read_int(payload, static_ofst, from_calldata)
            ofst = outer_offset + head
        else:
            ofst = static_ofst

        item = _decode_r(abi_sub_t, sub_t, ofst, payload, ivy_compat, from_calldata)

        ret.append(item)
        static_ofst += abi_sub_t.embedded_static_size()

    return ret

"""AST node construction utilities.

Provides functions for:
- Converting Python values to Vyper AST literal nodes
- Building composite AST nodes (calls, binops, comparisons, etc.)
"""

from typing import Optional

from vyper.ast import nodes as ast
from vyper.semantics.types import (
    VyperType,
    IntegerT,
    BoolT,
    AddressT,
    BytesM_T,
    BytesT,
    StringT,
    DecimalT,
    SArrayT,
    DArrayT,
    TupleT,
    StructT,
    FlagT,
)


# -----------------------------------------------------------------------------
# Value-to-AST conversion (for literals)
# -----------------------------------------------------------------------------


def _int_to_ast(value: int, typ: IntegerT) -> ast.Int:
    node = ast.Int(value=value)
    node._metadata["type"] = typ
    return node


def _bool_to_ast(value: bool, typ: BoolT) -> ast.NameConstant:
    node = ast.NameConstant(value=value)
    node._metadata["type"] = typ
    return node


def _address_to_ast(value: str, typ: AddressT) -> ast.Hex:
    if not value.startswith("0x"):
        value = f"0x{value}"
    node = ast.Hex(value=value)
    node._metadata["type"] = typ
    return node


def _bytesm_to_ast(value: bytes, typ: BytesM_T) -> ast.Hex:
    # Use ast.Hex (0x... format) for fixed-size bytesN types.
    # ast.HexBytes (x"..." format) only works for dynamic Bytes[N].
    node = ast.Hex(value=f"0x{value.hex()}")
    node._metadata["type"] = typ
    return node


def _bytes_to_ast(value: bytes, typ: BytesT) -> ast.Bytes:
    node = ast.Bytes(value=value)
    node._metadata["type"] = typ
    return node


def _string_to_ast(value: str, typ: StringT) -> ast.Str:
    node = ast.Str(value=value)
    node._metadata["type"] = typ
    return node


def _decimal_to_ast(value, typ: DecimalT) -> ast.Decimal:
    node = ast.Decimal(value=value)
    node._metadata["type"] = typ
    return node


def _array_to_ast(value: list, typ) -> ast.List:
    """Handle both SArrayT and DArrayT."""
    elements = [literal(v, typ.value_type) for v in value]
    node = ast.List(elements=elements)
    node._metadata["type"] = typ
    return node


def _tuple_to_ast(value: tuple, typ: TupleT) -> ast.Tuple:
    elements = [literal(v, t) for v, t in zip(value, typ.member_types)]
    node = ast.Tuple(elements=elements)
    node._metadata["type"] = typ
    return node


def _struct_to_ast(value: dict, typ: StructT) -> ast.Call:
    """Generate struct constructor call from dict value."""
    call_node = ast.Call(func=ast.Name(id=typ._id), args=[], keywords=[])

    for field_name, field_value in value.items():
        field_type = typ.members[field_name]
        field_expr = literal(field_value, field_type)
        keyword = ast.keyword(arg=field_name, value=field_expr)
        call_node.keywords.append(keyword)

    call_node._metadata["type"] = typ
    return call_node


def _flag_to_ast(value: int, typ: FlagT) -> ast.VyperNode:
    """Convert an integer bitmask to flag member AST (e.g., Roles.ADMIN | Roles.USER)."""
    flag_name = typ._id
    members = typ._flag_members  # dict: member_name -> index

    # Find which bits are set
    set_members = []
    for member_name, index in members.items():
        if value & (1 << index):
            set_members.append(member_name)

    # Handle empty flag (value == 0): use first member XOR'd with itself
    if not set_members:
        first_member = next(iter(members.keys()))
        member_node = ast.Attribute(value=ast.Name(id=flag_name), attr=first_member)
        member_node._metadata["type"] = typ
        node = ast.BinOp(left=member_node, op=ast.BitXor(), right=member_node)
        node._metadata["type"] = typ
        return node

    # Build AST nodes for each set member
    member_nodes = []
    for member_name in set_members:
        attr_node = ast.Attribute(value=ast.Name(id=flag_name), attr=member_name)
        attr_node._metadata["type"] = typ
        member_nodes.append(attr_node)

    # Single member: return directly
    if len(member_nodes) == 1:
        return member_nodes[0]

    # Multiple members: chain with BitOr
    result = member_nodes[0]
    for node in member_nodes[1:]:
        result = ast.BinOp(left=result, op=ast.BitOr(), right=node)
        result._metadata["type"] = typ
    return result


_LITERAL_BUILDERS = {
    IntegerT: _int_to_ast,
    BoolT: _bool_to_ast,
    AddressT: _address_to_ast,
    BytesM_T: _bytesm_to_ast,
    BytesT: _bytes_to_ast,
    StringT: _string_to_ast,
    DecimalT: _decimal_to_ast,
    SArrayT: _array_to_ast,
    DArrayT: _array_to_ast,
    TupleT: _tuple_to_ast,
    StructT: _struct_to_ast,
    FlagT: _flag_to_ast,
}


def literal(value, typ: VyperType) -> ast.VyperNode:
    """Convert a Python value to an AST literal node."""
    assert not isinstance(value, ast.VyperNode), "expected a Python value, not an AST node"
    builder = _LITERAL_BUILDERS.get(type(typ))
    if builder:
        return builder(value, typ)
    raise NotImplementedError(
        f"Literal conversion not implemented for {type(typ).__name__}"
    )


# -----------------------------------------------------------------------------
# Composite AST node builders
# -----------------------------------------------------------------------------


def uint256_literal(value: int) -> ast.Int:
    """Create a uint256 literal node."""
    return _int_to_ast(value, IntegerT(False, 256))


def builtin_call(
    name: str,
    args: list,
    ret_type: VyperType,
    builtins: Optional[dict] = None,
) -> ast.Call:
    """Create a builtin function call node."""
    fn = ast.Name(id=name)
    fn._metadata = {}
    if builtins and name in builtins:
        fn._metadata["type"] = builtins[name]
    call = ast.Call(func=fn, args=args, keywords=[])
    call._metadata = {"type": ret_type}
    return call


def binop(left: ast.VyperNode, op, right: ast.VyperNode, typ: VyperType) -> ast.BinOp:
    """Create a binary operation node."""
    node = ast.BinOp(left=left, op=op, right=right)
    node._metadata = {"type": typ}
    return node


def uint256_binop(left: ast.VyperNode, op, right: ast.VyperNode) -> ast.BinOp:
    """Create a uint256 binary operation node."""
    return binop(left, op, right, IntegerT(False, 256))


def compare(left: ast.VyperNode, op, right: ast.VyperNode) -> ast.Compare:
    """Create a comparison node (result type is always bool)."""
    node = ast.Compare(left=left, ops=[op], comparators=[right])
    node._metadata = {"type": BoolT()}
    return node


def ifexp(
    test: ast.VyperNode,
    body: ast.VyperNode,
    orelse: ast.VyperNode,
    typ: VyperType,
) -> ast.IfExp:
    """Create a ternary if-expression node."""
    node = ast.IfExp(test=test, body=body, orelse=orelse)
    node._metadata = {"type": typ}
    return node

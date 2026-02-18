import contextlib

from vyper.exceptions import SyntaxException
from vyper.semantics.analysis.common import VyperNodeVisitorBase

import vyper.ast as ast


def unparse(module: ast.Module) -> str:
    u = Unparser(module)
    return u.to_source()


class Unparser(VyperNodeVisitorBase):
    def __init__(self, module: ast.Module, *, indent_with="    "):
        self.buf: list[str] = []
        self.indent = 0
        self.indent_with = indent_with
        self.module = module  # root node

    def to_source(self) -> str:
        self.visit(self.module)
        return "".join(self.buf).rstrip() + "\n"

    def w(self, s=""):
        if s:
            self.buf.append(self.indent_with * self.indent + s)
        self.buf.append("\n")

    def ws(self, s):
        self.buf.append(s)

    @contextlib.contextmanager
    def block(self):
        self.indent += 1
        try:
            yield
        finally:
            self.indent -= 1

    PRECEDENCE = {
        ast.IfExp: -1,
        ast.Or: 0,
        ast.And: 1,
        ast.Not: 1,  # 'not' has lower precedence than comparisons
        ast.Compare: 2,
        ast.BitOr: 3,
        ast.BitXor: 4,
        ast.BitAnd: 5,
        ast.LShift: 6,
        ast.RShift: 6,
        ast.Add: 7,
        ast.Sub: 7,
        ast.Mult: 8,
        ast.Div: 8,
        ast.FloorDiv: 8,
        ast.Mod: 8,
        ast.UnaryOp: 9,
        ast.Pow: 10,
        # Subscript, Call, Attribute have precedence 11 (highest)
    }

    def p(self, node):
        if isinstance(node, ast.BinOp):
            return self.PRECEDENCE.get(type(node.op), 11)
        # 'not' has lower precedence than other unary ops
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.PRECEDENCE.get(ast.Not, 11)
        return self.PRECEDENCE.get(type(node), 11)

    def maybe_paren(self, expr, parent_prec):
        """Wraps expression in parentheses based on precedence"""
        expr_text = self._expr(expr)
        expr_prec = self.p(expr)

        if isinstance(expr, (ast.Compare, ast.BoolOp)) and parent_prec >= expr_prec:
            return f"({expr_text})"

        # Special case: Pow is right-associative, so 2**3**4 = 2**(3**4)
        # Only skip parens when child is also Pow (for right-associativity)
        if (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.op, ast.Pow)
            and isinstance(expr.parent, ast.BinOp)
            and isinstance(expr.parent.op, ast.Pow)
            and expr is expr.parent.right
        ):
            return expr_text

        return expr_text if expr_prec > parent_prec else f"({expr_text})"

    def _expr(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise SyntaxException(
                f"unparse: unsupported node {type(node).__name__}", node
            )
        return method(node)

    # Top-level nodes -------------------------------------------------------
    def visit_Module(self, node):
        # Emit pragmas first
        pragmas = self._get_pragmas(node)
        for pragma in pragmas:
            self.w(pragma)

        for i, item in enumerate(node.body):
            if i > 0 or pragmas:
                self.w()  # blank line between top-level items
            self.visit(item)

    def _get_pragmas(self, module):
        """Generate pragma lines from module settings."""
        pragmas = []
        settings = getattr(module, "settings", None)
        if settings is None:
            return pragmas

        # nonreentrancy pragma
        nonreentrancy = getattr(settings, "nonreentrancy_by_default", None)
        if nonreentrancy is True:
            pragmas.append("# pragma nonreentrancy on")
        elif nonreentrancy is False:
            pragmas.append("# pragma nonreentrancy off")

        return pragmas

    def visit_StructDef(self, node):
        self.w(f"struct {node.name}:")
        with self.block():
            self._process_body(node.body)

    def visit_FlagDef(self, node):
        self.w(f"flag {node.name}:")
        with self.block():
            self._process_body(node.body)

    def visit_EventDef(self, node):
        self.w(f"event {node.name}:")
        with self.block():
            self._process_body(node.body)

    def visit_InterfaceDef(self, node):
        self.w(f"interface {node.name}:")
        with self.block():
            self._process_body(node.body)

    # Function definitions --------------------------------------------------
    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.w(f"@{decorator.id}")

        # Function signature
        signature = f"def {node.name}("

        # Process arguments
        args = []
        if node.args.args:
            defaults = node.args.defaults or []
            default_offset = len(node.args.args) - len(defaults)

            for i, arg in enumerate(node.args.args):
                arg_str = f"{arg.arg}: {self._expr(arg.annotation)}"
                if i >= default_offset:  # This arg has a default
                    default_value = defaults[i - default_offset]
                    arg_str += f" = {self._expr(default_value)}"
                args.append(arg_str)

        signature += ", ".join(args)
        signature += ")"

        # Return type
        if node.returns:
            signature += f" -> {self._expr(node.returns)}"

        # Check if this is an interface function (single Expr with Name for mutability)
        if self._is_interface_function(node):
            mutability = node.body[0].value.id
            self.w(f"{signature}: {mutability}")
        else:
            self.w(signature + ":")
            with self.block():
                self._process_body(node.body)

    def _is_interface_function(self, node):
        """Check if this is an interface function definition (has mutability, no real body)."""
        if len(node.body) != 1:
            return False
        body_item = node.body[0]
        if not isinstance(body_item, ast.Expr):
            return False
        if not isinstance(body_item.value, ast.Name):
            return False
        # Check if it's a mutability keyword
        return body_item.value.id in ("pure", "view", "nonpayable", "payable")

    def _process_body(self, body):
        prev_was_def = False
        for i, stmt in enumerate(body):
            is_def = isinstance(stmt, (ast.FunctionDef, ast.StructDef))

            # Add blank line between consecutive function/struct definitions
            if i > 0 and prev_was_def and is_def:
                self.w()

            self.visit(stmt)
            prev_was_def = is_def

    # Statements -----------------------------------------------------------
    def visit_Return(self, node):
        if node.value:
            self.w(f"return {self._expr(node.value)}")
        else:
            self.w("return")

    def visit_Assign(self, node):
        self.w(f"{self._expr(node.target)} = {self._expr(node.value)}")

    def visit_AugAssign(self, node):
        op_token = self._get_op_token(node.op)
        self.w(f"{self._expr(node.target)} {op_token}= {self._expr(node.value)}")

    def visit_AnnAssign(self, node, as_expr=False):
        result = f"{self._expr(node.target)}: {self._expr(node.annotation)}"
        if node.value:
            result += f" = {self._expr(node.value)}"
        if as_expr:
            return result
        self.w(result)

    def visit_VariableDecl(self, node):
        result = f"{self._expr(node.target)}: "

        flags = []
        if getattr(node, "is_reentrant", False):
            flags.append("reentrant")
        if node.is_public:
            flags.append("public")
        if node.is_constant:
            flags.append("constant")
        if node.is_immutable:
            flags.append("immutable")
        if node.is_transient:
            flags.append("transient")

        if flags:
            inner_anno = self._expr(node.annotation)
            for flag in reversed(flags):  # Apply in reverse order
                inner_anno = f"{flag}({inner_anno})"
            result += inner_anno
        else:
            result += self._expr(node.annotation)

        if node.value:
            result += f" = {self._expr(node.value)}"

        self.w(result)

    def visit_Pass(self, node):
        self.w("pass")

    def visit_Raise(self, node):
        if node.exc:
            self.w(f"raise {self._expr(node.exc)}")
        else:
            self.w("raise")

    def visit_Assert(self, node):
        result = f"assert {self._expr(node.test)}"
        if node.msg:
            result += f", {self._expr(node.msg)}"
        self.w(result)

    def visit_If(self, node):
        self.w(f"if {self._expr(node.test)}:")
        with self.block():
            self._process_body(node.body)

        if node.orelse:
            self.w("else:")
            with self.block():
                self._process_body(node.orelse)

    def visit_For(self, node):
        assert isinstance(node.target, ast.AnnAssign)
        self.w(
            f"for {self.visit_AnnAssign(node.target, as_expr=True)} in {self._expr(node.iter)}:"
        )
        with self.block():
            self._process_body(node.body)

    def visit_Break(self, node):
        self.w("break")

    def visit_Continue(self, node):
        self.w("continue")

    def visit_Log(self, node):
        self.w(f"log {self._expr(node.value)}")

    def visit_Import(self, node):
        for alias in node.names:
            result = f"import {alias.name}"
            if alias.asname:
                result += f" as {alias.asname}"
            self.w(result)

    def visit_ImportFrom(self, node):
        level_dots = "." * node.level
        module_path = f"{level_dots}{node.module}" if node.module else level_dots
        names = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)
        self.w(f"from {module_path} import {', '.join(names)}")

    def visit_ImplementsDecl(self, node):
        self.w(f"implements: {self._expr(node.children[0])}")

    def visit_UsesDecl(self, node):
        self.w(f"uses: {self._expr(node.annotation)}")

    def visit_InitializesDecl(self, node):
        self.w(f"initializes: {self._expr(node.annotation)}")

    def visit_ExportsDecl(self, node):
        self.w(f"exports: {self._expr(node.annotation)}")

    # Expressions ----------------------------------------------------------
    def visit_Expr(self, node):
        self.w(self._expr(node.value))

    def visit_NameConstant(self, node):
        return str(node.value)

    def visit_Name(self, node):
        return node.id

    def visit_Ellipsis(self, node):
        return "..."

    def visit_Int(self, node):
        return str(node.value)

    def visit_Decimal(self, node):
        # Preserve original source format if available (e.g., .33 vs 0.33)
        if hasattr(node, "node_source_code") and node.node_source_code:
            return node.node_source_code
        # Fallback: avoid scientific notation
        return f"{node.value:f}"

    def visit_Hex(self, node):
        if isinstance(node.value, str):
            return node.value.replace("0X", "0x")
        return f"0x{node.value:x}"

    def visit_HexBytes(self, node):
        return f'x"{node.value.hex()}"'

    def visit_Bytes(self, node):
        # Bytes (dynamic) uses b'...' format, not hex
        return repr(node.value)

    def visit_Str(self, node):
        return repr(node.value)

    def visit_List(self, node):
        elements = [self._expr(e) for e in node.elements]
        return f"[{', '.join(elements)}]"

    def visit_Tuple(self, node):
        elements = [self._expr(e) for e in node.elements]
        if len(elements) == 1:
            return f"({elements[0]},)"  # singleton tuple needs trailing comma
        return f"({', '.join(elements)})"

    def visit_Dict(self, node):
        items = []
        for k, v in zip(node.keys, node.values):
            items.append(f"{self._expr(k)}: {self._expr(v)}")
        return f"{{{', '.join(items)}}}"

    def visit_UnaryOp(self, node):
        op_token = self._get_unary_op_token(node.op)
        operand = self.maybe_paren(node.operand, self.p(node))
        return f"{op_token}{operand}"

    def visit_BinOp(self, node):
        op_token = self._get_op_token(node.op)
        op_prec = self.p(node.op)
        left = self.maybe_paren(node.left, op_prec)
        # Negative integer literals need parentheses as base of power
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.left, ast.Int)
            and node.left.value < 0
        ):
            left = f"({left})"
        right = self.maybe_paren(node.right, op_prec)
        return f"{left} {op_token} {right}"

    def visit_BoolOp(self, node):
        op_token = "and" if isinstance(node.op, ast.And) else "or"
        values = [self.maybe_paren(val, self.p(node)) for val in node.values]
        return f" {op_token} ".join(values)

    def visit_Compare(self, node):
        op_token = self._get_cmp_op_token(node.op)
        left = self.maybe_paren(node.left, self.p(node))
        right = self.maybe_paren(node.right, self.p(node))
        return f"{left} {op_token} {right}"

    def visit_IfExp(self, node):
        body = self._expr(node.body)
        test = self._expr(node.test)
        orelse = self._expr(node.orelse)
        return f"{body} if ({test}) else {orelse}"

    def visit_Call(self, node):
        func = self._expr(node.func)
        args = []
        for arg in node.args:
            args.append(self._expr(arg))

        for kw in node.keywords:
            if kw.arg is None:
                raise SyntaxException("unparse: **kwargs not supported", kw)
            args.append(f"{kw.arg}={self._expr(kw.value)}")

        return f"{func}({', '.join(args)})"

    def visit_keyword(self, node):
        if node.arg is None:
            raise SyntaxException("unparse: **kwargs not supported", node)
        return f"{node.arg}={self._expr(node.value)}"

    def visit_ExtCall(self, node):
        return f"extcall {self._expr(node.value)}"

    def visit_StaticCall(self, node):
        return f"staticcall {self._expr(node.value)}"

    def visit_Attribute(self, node):
        value = self._expr(node.value)
        # staticcall/extcall must be parenthesized before attribute access
        # e.g. (staticcall IFoo(addr).bar()).field, not staticcall IFoo(addr).bar().field
        if isinstance(node.value, (ast.StaticCall, ast.ExtCall)):
            value = f"({value})"
        return f"{value}.{node.attr}"

    def visit_Subscript(self, node):
        value = self._expr(node.value)
        # staticcall/extcall must be parenthesized before subscript access
        if isinstance(node.value, (ast.StaticCall, ast.ExtCall)):
            value = f"({value})"
        # For Tuple slices, don't add parentheses (e.g., HashMap[address, uint256])
        if isinstance(node.slice, ast.Tuple):
            elements = [self._expr(e) for e in node.slice.elements]
            slice_expr = ", ".join(elements)
        else:
            slice_expr = self._expr(node.slice)
        return f"{value}[{slice_expr}]"

    def visit_NamedExpr(self, node):
        target = self._expr(node.target)
        value = self._expr(node.value)
        return f"{target} := {value}"

    # Helper methods for operator tokens ------------------------------------
    def _get_unary_op_token(self, op):
        if isinstance(op, ast.USub):
            return "-"
        elif isinstance(op, ast.Not):
            return "not "
        elif isinstance(op, ast.Invert):
            return "~"
        else:
            raise SyntaxException(
                f"unparse: unsupported unary operator {type(op).__name__}", op
            )

    def _get_op_token(self, op):
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        elif isinstance(op, ast.FloorDiv):
            return "//"
        elif isinstance(op, ast.Mod):
            return "%"
        elif isinstance(op, ast.Pow):
            return "**"
        elif isinstance(op, ast.LShift):
            return "<<"
        elif isinstance(op, ast.RShift):
            return ">>"
        elif isinstance(op, ast.BitOr):
            return "|"
        elif isinstance(op, ast.BitXor):
            return "^"
        elif isinstance(op, ast.BitAnd):
            return "&"
        else:
            raise SyntaxException(
                f"unparse: unsupported binary operator {type(op).__name__}", op
            )

    def _get_cmp_op_token(self, op):
        if isinstance(op, ast.Eq):
            return "=="
        elif isinstance(op, ast.NotEq):
            return "!="
        elif isinstance(op, ast.Lt):
            return "<"
        elif isinstance(op, ast.LtE):
            return "<="
        elif isinstance(op, ast.Gt):
            return ">"
        elif isinstance(op, ast.GtE):
            return ">="
        elif isinstance(op, ast.In):
            return "in"
        elif isinstance(op, ast.NotIn):
            return "not in"
        else:
            raise SyntaxException(
                f"unparse: unsupported compare operator {type(op).__name__}", op
            )

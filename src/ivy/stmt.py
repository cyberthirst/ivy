class StatementVisitor:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit method for {type(node).__name__}")

    def visit_Expr(self, node):
        # Evaluate the expression
        return self.interpreter.evaluate_expr(node.value)

    def visit_Pass(self, node):
        # Do nothing for pass statement
        return None

    def visit_Name(self, node):
        if node.id == "vdb":
            # Handle debugger
            return None
        raise Exception(f"Unsupported Name: {node.id}")

    def visit_AnnAssign(self, node):
        value = self.interpreter.evaluate_expr(node.value)
        self.interpreter.set_variable(node.target.id, value)
        return None

    def visit_Assign(self, node):
        value = self.interpreter.evaluate_expr(node.value)
        target = self.visit(node.target)
        self.interpreter.set_variable(target, value)
        return None

    def visit_If(self, node):
        condition = self.interpreter.evaluate_expr(node.test)
        if condition:
            return self.visit_body(node.body)
        elif node.orelse:
            return self.visit_body(node.orelse)
        return None


    def visit_Assert(self, node):
        condition = self.interpreter.evaluate_expr(node.test)
        if not condition:
            if node.msg:
                msg = self.interpreter.evaluate_expr(node.msg)
                raise AssertionError(msg)
            else:
                raise AssertionError()
        return None

    def visit_Raise(self, node):
        if node.exc:
            exc = self.interpreter.evaluate_expr(node.exc)
            raise exc
        else:
            raise Exception("Generic raise")

    def visit_For(self, node):
        # Implement for loop logic
        pass

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        right = self.interpreter.evaluate_expr(node.value)
        left = self.interpreter.get_variable(target)
        new_value = self.interpreter.handle_binop(node.op, left, right)
        self.interpreter.set_variable(target, new_value)
        return None

    def visit_Continue(self, node):
        return 'continue'

    def visit_Break(self, node):
        return 'break'

    def visit_Return(self, node):
        if node.value:
            return self.interpreter.evaluate_expr(node.value)
        return None

    def visit_body(self, body):
        for stmt in body:
            result = self.visit(stmt)
            if result == 'continue' or result == 'break' or isinstance(result, Exception):
                return result
        return None
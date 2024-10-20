from typing import Optional, Type, Union
import inspect
from collections import defaultdict
from contextlib import contextmanager

import vyper.ast.nodes as ast
from vyper.semantics.types import VyperType, TYPE_T, InterfaceT, StructT
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.function import (
    ContractFunctionT,
)
from vyper.builtins._signatures import BuiltinFunctionT
from vyper.codegen.core import calculate_type_for_external_return

from ivy.expr import ExprVisitor
from ivy.evm_structures import Account, Environment, Message, ContractData, EVMOutput
from ivy.stmt import ReturnException, StmtVisitor
from ivy.evaluator import VyperEvaluator
from ivy.variable import GlobalVariable
from ivy.context import ExecutionContext
import ivy.builtins as vyper_builtins
from ivy.utils import compute_call_abi_data, compute_contract_address
from ivy.abi import abi_decode, abi_encode
from ivy.journal import Journal
from ivy.exceptions import EVMException, StaticCallViolation
from ivy.types import Address


class VyperInterpreter(ExprVisitor, StmtVisitor):
    execution_ctxs: list[ExecutionContext]
    env: Optional[Environment]
    contract: Optional[ContractData]
    evaluator: Type[VyperEvaluator]
    accessed_accounts: set[Account]
    journal: Journal

    def __init__(self):
        self.state = defaultdict(lambda: Account(0, 0, {}, {}, None))
        self.returndata = None
        self.evaluator = VyperEvaluator
        self.execution_ctxs = []
        self.builtins = {}
        self._collect_builtins()
        self.env = None
        self.journal = Journal()
        self.accessed_accounts = set()

    def _collect_builtins(self):
        for name, func in inspect.getmembers(vyper_builtins, inspect.isfunction):
            if name.startswith("builtin_"):
                builtin_name = name[8:]  # Remove 'builtin_' prefix
                self.builtins[builtin_name] = func

    @property
    def deployer(self):
        from ivy.frontend.vyper_contract import VyperDeployer

        return VyperDeployer

    @property
    def fun_ctx(self):
        return self.execution_ctxs[-1].current_fun_context()

    @property
    def globals(self):
        return self.exec_ctx.globals

    @property
    def msg(self):
        return self.execution_ctxs[-1].msg

    @property
    def current_address(self):
        return self.exec_ctx.msg.to

    @property
    def memory(self):
        # function variables are located in memory
        # fun ctx has set/get methods which are used to access the variables from scopes
        return self.fun_ctx

    @property
    def exec_ctx(self):
        return self.execution_ctxs[-1]

    def clear_transient_storage(self):
        for address in self.accessed_accounts:
            self.state[address].transient.clear()
        self.accessed_accounts.clear()

    def get_nonce(self, address):
        return self.state[address].nonce

    def increment_nonce(self, address):
        self.state[address].nonce += 1

    def _push_fun_ctx(self, func_t):
        self.execution_ctxs[-1].push_fun_context(func_t)

    def _pop_fun_ctx(self):
        self.execution_ctxs[-1].pop_fun_context()

    def _push_scope(self):
        self.execution_ctxs[-1].push_scope()

    def _pop_scope(self):
        self.execution_ctxs[-1].pop_scope()

    def generate_create_address(self, sender):
        nonce = self.get_nonce(sender.canonical_address)
        self.increment_nonce(sender.canonical_address)
        return Address(compute_contract_address(sender.canonical_address, nonce))

    def execute_tx(
        self,
        sender: Address,
        to: Union[Address, bytes],
        value: int,
        calldata: bytes = b"",
        is_static: bool = False,
        module: Optional[ast.Module] = None,
    ):
        is_deploy = to == b""
        create_address, code = None, None

        if is_deploy:
            module_t = module._metadata["type"]
            assert isinstance(module_t, ModuleT)
            create_address = self.generate_create_address(sender)
            code = ContractData(module_t)
        else:
            code = self.get_code(to)

        message = Message(
            caller=sender,
            to=b"" if is_deploy else to,
            create_address=create_address,
            value=value,
            data=calldata,
            code_address=to,
            code=code,
            depth=0,
            is_static=is_static,
        )

        self.env = Environment(
            caller=sender,
            block_hashes=[],
            origin=to,
            coinbase=sender,
            number=0,
            time=0,
            prev_randao=b"",
            chain_id=0,
        )

        self.journal.begin_call()
        output = (
            self.process_create_message(message)
            if is_deploy
            else self.process_message(message)
        )
        self.journal.finalize_call(output.is_error)

        for a in self.accessed_accounts:
            # global_vars reference the storage, it's necessary to clear instead of assigning a new dict
            # it might be better to refactor GlobalVariable to receive a function to retrieve storage
            # instaed of receiving the storage directly
            a.transient.clear()
        self.accessed_accounts.clear()

        if output.is_error:
            raise output.error

        return create_address if is_deploy else output.bytes_output()

    def process_message(self, message: Message) -> EVMOutput:
        account = self.state[message.to]
        self.accessed_accounts.add(account)
        exec_ctx = ExecutionContext(
            account,
            message,
            account.contract_data.module_t if account.contract_data else None,
        )
        self.execution_ctxs.append(exec_ctx)

        output = EVMOutput()

        try:
            if message.value > 0:
                if self.state[message.caller].balance < message.value:
                    raise EVMException("Insufficient balance for transfer")
                self.state[message.caller].balance -= message.value
                self.state[message.to].balance += message.value

            if message.code:
                self._handle_incomming_extcall()

            output.data = self.exec_ctx.output

        except Exception as e:
            # TODO rollback the journal
            output.error = e

        finally:
            self.execution_ctxs.pop()

        return output

    def process_create_message(self, message: Message) -> EVMOutput:
        if message.create_address in self.state:
            raise EVMException("Address already taken")

        new_account = self.state[message.create_address]
        self.accessed_accounts.add(new_account)

        exec_ctx = ExecutionContext(new_account, message, message.code)
        self.execution_ctxs.append(exec_ctx)

        output = EVMOutput()

        try:
            if message.value > 0:
                if self.state[message.caller].balance < message.value:
                    raise EVMException("Insufficient balance for contract creation")
                self.state[message.caller].balance -= message.value

            module_t = message.code.module_t

            for decl in module_t.variable_decls:
                name = decl.target.id
                typ = decl._metadata["type"]
                loc = self.get_location_from_decl(decl)
                self._new_variable(name, typ, loc)

            # allocate a nonreentrant key for all contracts, they might not use it
            self._new_variable(REENTRANT_KEY, BoolT(), self.exec_ctx.transient)

            if module_t.init_function is not None:
                self._execute_function(module_t.init_function, message.data)

            new_account.contract_data = message.code

        except Exception as e:
            # TODO rollback the journal
            output.error = e
            del self.state[message.create_address]

        finally:
            self.execution_ctxs.pop()

        return output

    def get_code(self, address):
        return self.state[address].contract_data

    def _dispatch(self, selector):
        entry_points = self.exec_ctx.entry_points

        if selector not in entry_points:
            # TODO check fallback
            # TODO rollback the journal
            raise Exception(f"function {selector} not found")
        else:
            entry_point = entry_points[selector]

        if entry_point.function.is_payable:
            if self.msg.value != 0:
                # TODO raise and rollback
                pass

        # check args

        # check decorators

        return entry_point

    def lock(self, mutability):
        lock = self.get_variable(REENTRANT_KEY)
        if lock:
            raise AccessViolation("Reentrancy violation")

        # for view functions we can't write the lock, so we only check it's not locked
        if mutability == StateMutability.VIEW:
            return

        self.set_variable(REENTRANT_KEY, True)

    def unlock(self, mutability):
        lock = self.get_variable(REENTRANT_KEY)
        if mutability == StateMutability.VIEW:
            assert lock == False
            return

        assert lock == True
        self.set_variable(REENTRANT_KEY, False)

    def _prologue(self, func_t, args):
        self._push_fun_ctx(func_t)
        self._push_scope()

        # TODO handle reentrancy lock
        func_args = func_t.arguments
        for arg, param in zip(args, func_args):
            self._new_variable(param.name, param.typ)
            self.set_variable(param.name, arg)

        # check if we need to assign default values
        if len(args) < len(func_args):
            for param in func_args[len(args) :]:
                self._new_variable(param.name, param.typ)
                default_value = self.visit(param.default_value)
                self.set_variable(param.name, default_value)

    def _epilogue(self, *args):
        # TODO handle reentrancy lock
        # TODO handle return value
        self._pop_scope()
        self._pop_fun_ctx()
        pass

    def _exec_body(self):
        for stmt in self.fun_ctx.function.decl_node.body:
            try:
                self.visit(stmt)
            except ReturnException as e:
                # TODO handle the return value
                self.exec_ctx.output = e.value
                break

    def _execute_function(self, func_t, args):
        self._prologue(func_t, args)
        self._exec_body()
        self._epilogue()

        return self.exec_ctx.output

    def _handle_incomming_extcall(self):
        if len(self.msg.data) < 4:
            # TODO goto fallback or revert
            pass

        selector = self.msg.data[:4]
        entry_point = self._dispatch(selector)

        if entry_point.calldata_min_size > len(self.msg.data):
            raise BufferError(
                f"Provided calldata is too small, min_size is {entry_point.calldata_min_size}"
            )

        func_t = entry_point.function
        args = abi_decode(entry_point.calldata_args_t, self.msg.data[4:])

        self.exec_ctx.function = func_t

        self._execute_function(func_t, args)

        # abi-encode output
        typ = self.exec_ctx.function.return_type
        if typ is None:
            assert self.exec_ctx.output is None
            return None
        typ = calculate_type_for_external_return(typ)
        output = self.exec_ctx.output
        # from https://github.com/vyperlang/vyper/blob/a1af967e675b72051cf236f75e1104378fd83030/vyper/codegen/core.py#L694
        output = (
            (output,) if (not isinstance(output, tuple) or len(output) <= 1) else output
        )
        self.exec_ctx.output = abi_encode(typ, output)

    @contextmanager
    def modifiable_context(self, target):
        if self.msg.is_static:
            raise StaticCallViolation(f"Cannot modify {target} in a static context")
        try:
            yield
        finally:
            pass

    def get_variable(self, name: str):
        if name in self.globals:
            return self.globals[name].value
        else:
            return self.memory[name]

    def set_variable(self, name: str, value):
        if name in self.globals:
            with self.modifiable_context(name):
                var = self.globals[name]
                var.value = value
        else:
            self.memory[name] = value

    def _assign_target(self, target, value):
        if isinstance(target, ast.Name):
            self.set_variable(target.id, value)
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise TypeError("Cannot unpack non-iterable to tuple")
            if len(target.elements) != len(value):
                raise ValueError("Mismatch in number of items to unpack")
            for t, v in zip(target.elements, value):
                self._assign_target(t, v)
        elif isinstance(target, ast.Subscript):
            container = self.visit(target.value)
            index = self.visit(target.slice)
            container[index] = value
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                try:
                    self.set_variable(target.attr, value)
                except KeyError:
                    pass
            else:
                obj = self.visit(target.value)
                setattr(obj, target.attr, value)
        else:
            raise NotImplementedError(f"Assignment to {type(target)} not implemented")

    def get_location_from_decl(self, decl: ast.VariableDecl):
        if decl.is_immutable:
            return self.exec_ctx.immutables
        elif decl.is_constant:
            return self.exec_ctx.constants
        elif decl.is_transient:
            return self.exec_ctx.transient
        else:
            return self.exec_ctx.storage

    def _new_variable(self, name: str, typ: VyperType, global_loc: dict = None):
        if global_loc is not None:
            var = GlobalVariable(name, typ, global_loc)
            assert name not in self.globals
            self.globals[name] = var
        else:
            self.memory.new_variable(name, typ)

    def _handle_address_variable(self, node: ast.Attribute):
        # x.address
        if node.attr == "address":
            return self.visit(node.value)
        # x.balance: balance of address x
        if node.attr == "balance":
            addr = self.visit(node.value)
            return self.get_balance(addr)
        # x.codesize: codesize of address x
        elif node.attr == "codesize" or node.attr == "is_contract":
            addr = self.visit(node.value)
            if node.attr == "codesize":
                raise NotImplementedError("codesize")
            else:
                return self.get_code(addr) is not None
        # x.codehash: keccak of address x
        elif node.attr == "codehash":
            raise NotImplementedError("codehash")
        # x.code: codecopy/extcodecopy of address x
        elif node.attr == "code":
            raise NotImplementedError("code")

    def _handle_env_variable(self, node: ast.Attribute):
        key = f"{node.value.id}.{node.attr}"
        if key == "msg.sender":
            return self.msg.caller
        elif key == "msg.data":
            return self.msg.data
        elif (
            key == "msg.value"
        ):  # TODO check payble (context and self.context.is_payable:)
            return self.msg.value
        elif key in ("msg.gas", "msg.mana"):
            raise GasReference()
        elif key == "block.prevrandao":
            raise NotImplementedError("block.prevrandao")
        elif key == "block.difficulty":
            raise NotImplementedError("block.prevrandao")
        elif key == "block.timestamp":
            return self.env.time
        elif key == "block.coinbase":
            return self.env.coinbase
        elif key == "block.number":
            return self.env.number
        elif key == "block.gaslimit":
            raise GasReference()
        elif key == "block.basefee":
            raise GasReference()
        elif key == "block.blobbasefee":
            raise NotImplementedError("block.blobbasefee")
        elif key == "block.prevhash":
            raise NotImplementedError("block.prevhash")
        elif key == "tx.origin":
            return self.env.origin
        elif key == "tx.gasprice":
            raise GasReference()
        elif key == "chain.id":
            return self.env.chain_id

    def generic_call_handler(
        self,
        call: ast.Call,
        args,
        kws,
        typs,
        target: Optional[Address] = None,
        is_static: Optional[bool] = None,
    ):
        # `None` is a special case: range is not assigned `type` in Vyper's frontend
        func_t = call.func._metadata.get("type", None)

        if func_t is None or isinstance(func_t, BuiltinFunctionT):
            id = call.func.id
            if id == "raw_call":
                # dependency injection
                args = (self.message_call,) + args
            elif id == "abi_encode" or id == "_abi_encode":
                args = (typs, args)
            return self.builtins[id](*args, **kws)

        if isinstance(func_t, TYPE_T):
            # struct & interface constructors
            typedef = func_t.typedef
            if isinstance(typedef, InterfaceT):
                # TODO should we return an address here? or an interface object wrapping the address?
                assert len(args) == 1
                return args[0]
            else:
                assert isinstance(typedef, StructT)
                assert len(args) == 0
                return VyperEvaluator.construct_struct(typedef.name, kws)

        if func_t.is_external:
            assert target is not None
            assert isinstance(func_t, ContractFunctionT)
            return self.external_function_call(func_t, args, kws, is_static, target)

        assert func_t.is_internal
        return self._execute_function(func_t, args)

    def external_function_call(
        self, func_t: ContractFunctionT, args, kwargs, is_static: bool, target: Address
    ):
        num_kwargs = len(args) - func_t.n_positional_args

        selector, calldata_args_t = compute_call_abi_data(func_t, num_kwargs)

        data = selector
        data += abi_encode(calldata_args_t, args)

        output = self.message_call(
            target, kwargs.get("value", 0), data, is_static=is_static, is_delegate=False
        )

        self.exec_ctx.returndata = output.bytes_output()

        if len(self.exec_ctx.returndata) == 0 and "default_return_value" in kwargs:
            return kwargs["default_return_value"]

        typ = func_t.return_type
        typ = calculate_type_for_external_return(typ)
        abi_typ = typ.abi_type

        max_return_size = abi_typ.size_bound()

        actual_output_size = min(max_return_size, len(self.exec_ctx.returndata))
        to_decode = self.exec_ctx.returndata[:actual_output_size]

        # NOTE: abi_decode implicitly checks minimum return size
        decoded = abi_decode(typ, to_decode)
        assert len(decoded) == 1
        # unwrap the tuple
        return decoded[0]

    def message_call(
        self,
        target: Address,
        value: int,
        data: bytes,
        is_static: bool = False,
        is_delegate=False,
    ) -> EVMOutput:
        code_address = target
        code = self.get_code(code_address)

        if is_delegate:
            target = self.current_address

        msg = Message(
            caller=self.msg.to,
            to=target,
            create_address=None,
            value=value,
            data=data,
            code_address=code_address,
            code=code,
            depth=self.exec_ctx.msg.depth + 1,
            is_static=is_static,
        )

        self.journal.begin_call()
        output = self.process_message(msg)
        self.journal.finalize_call(output.is_error)

        return output

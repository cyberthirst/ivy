from vyper.semantics.types import TupleT, StringT
from vyper.utils import method_id


class EVMException(Exception):
    pass


class StaticCallViolation(EVMException):
    pass


class SelfDestruct(EVMException):
    pass


class UnsupportedFeature(EVMException):
    def __init__(self, message="Unsupported feature in Ivy"):
        super().__init__(message)


class GasReference(UnsupportedFeature):
    def __init__(self, message="Gas is not a supported concept in Ivy"):
        super().__init__(message)


class VyperException(Exception):
    pass


class Revert(VyperException):
    def __init__(self, message="Revert", data=b""):
        super().__init__(message)
        self.data = data

    def __str__(self):
        if self.data:
            if len(self.data) >= 4:
                error_method_id = method_id("Error(string)")
                if error_method_id == self.data[:4]:
                    from ivy.abi import abi_decode

                    ret = abi_decode(TupleT((StringT(2**16),)), self.data[4:])
                    # VyperString is bytes-based, decode to str for display
                    msg = ret[0]
                    if isinstance(msg, bytes):
                        return msg.decode("utf-8", errors="surrogateescape")
                    return msg
            return self.data.hex()
        return super().__str__()


class Invalid(VyperException):
    pass


class PayabilityViolation(VyperException):
    pass


class AccessViolation(Revert):
    pass


class FunctionNotFound(Revert):
    pass


class Raise(Revert):
    pass


class Assert(Revert):
    pass

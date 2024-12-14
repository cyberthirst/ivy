class EVMException(Exception):
    pass


class StaticCallViolation(EVMException):
    pass


class GasReference(EVMException):
    def __init__(self, message="Gas is not a supported concept in Ivy"):
        super().__init__(message)


class VyperException(Exception):
    pass


class Revert(VyperException):
    def __init__(self, message="Revert", data=b""):
        super().__init__(message)
        self.data = data


class Invalid(VyperException):
    pass


class AccessViolation(Revert):
    pass


class FunctionNotFound(Revert):
    pass


class Raise(Revert):
    pass


class Assert(Revert):
    pass

class EVMException(Exception):
    pass


class StaticCallViolation(EVMException):
    pass


class GasReference(EVMException):
    def __init__(self, message="Gas is not a supported concept in Ivy"):
        super().__init__(message)


class AccessViolation(Exception):
    pass


class FunctionNotFound(Exception):
    pass

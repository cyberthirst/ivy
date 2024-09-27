from ivy.frontend.env import Env


env = Env.get_singleton()


def set_interpreter(interpreter):
    env.interpreter = interpreter


def set_evm(evm):
    env.evm = evm

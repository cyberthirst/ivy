import sys

import titanoboa.boa as boa
import titanoboa.boa.interpret as boa_itp
from titanoboa.boa.ivy import IvyEnv


def main():
    boa.set_env(IvyEnv())
    out = boa_itp.load(sys.argv[1])
    print(out.compiler_data.compilation_target)
    out.foo()


if __name__ == "__main__":
    main() 
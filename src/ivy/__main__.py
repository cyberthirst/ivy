import sys

import ivy.interpret as itp

def main():
    out = itp.load(sys.argv[1])
    print(out.compiler_data.compilation_target)
    #out.foo()


if __name__ == "__main__":
    main() 
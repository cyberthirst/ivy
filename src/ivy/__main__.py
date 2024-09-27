import sys

import ivy.frontend.loader as itp


def main():
    out = itp.load(sys.argv[1])
    # print(out.compiler_data.compilation_target)
    print(out.foo())


if __name__ == "__main__":
    main()

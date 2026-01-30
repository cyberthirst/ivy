import random
import traceback
from pathlib import Path

from vyper.ast import nodes as ast

from fuzzer.compilation import compile_vyper
from fuzzer.mutator.ast_mutator import AstMutator
from unparser.unparser import unparse

FAILURES_DIR = Path("compilation_failures")


def main():
    FAILURES_DIR.mkdir(exist_ok=True)

    total = 1000
    successful = 0
    successful_xfail = 0
    failed_xfail = 0
    failed = 0
    ice = 0

    for i in range(total):
        seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        mutator = AstMutator(rng, generate=True)
        module = mutator.mutate(ast.Module(body=[]))
        source = unparse(module)

        if mutator.type_generator.source_fragments:
            source = "\n\n".join(mutator.type_generator.source_fragments) + "\n\n" + source

        result = compile_vyper(source)

        if result.is_success:
            successful += 1
        elif result.is_compiler_crash:
            ice += 1
        elif mutator.context.compilation_xfails:
            successful_xfail += 1
        else:
            failed += 1
            tb = "".join(
                traceback.format_exception(
                    type(result.error), result.error, result.error.__traceback__
                )
            )
            failure_path = FAILURES_DIR / f"seed_{seed}.txt"
            failure_path.write_text(f"{source}\n\n{tb}")

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{total}")

    print()
    print(f"successful compilation:          {successful:4d} ({successful / total:6.1%})")
    print(f"successful xfail:                {successful_xfail:4d} ({successful_xfail / total:6.1%})")
    print(f"compilation failed + xfail set:  {failed_xfail:4d} ({failed_xfail / total:6.1%})")
    print(f"compilation failed:              {failed:4d} ({failed / total:6.1%})")
    print(f"ice:                             {ice:4d} ({ice / total:6.1%})")


if __name__ == "__main__":
    main()

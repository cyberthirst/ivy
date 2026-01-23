import random

import pytest
from vyper.compiler.phases import CompilerData
from vyper.ast import nodes as ast
from vyper.exceptions import VyperException

from fuzzer.mutator.ast_mutator import AstMutator
from unparser.unparser import unparse


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("_iteration", range(100))
def test_generator_produces_valid_code(_iteration: int):
    seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)
    mutator = AstMutator(rng, generate=True)
    module = mutator.mutate(ast.Module(body=[]))
    source = unparse(module)

    if mutator.type_generator.source_fragments:
        source = "\n\n".join(mutator.type_generator.source_fragments) + "\n\n" + source

    try:
        CompilerData(source).bytecode
    except VyperException as e:
        if mutator.context.compilation_xfails:
            return
        pytest.fail(
            f"Generated code failed to compile.\n"
            f"seed={seed}\n"
            f"Error: {e}\n"
            f"Generated source:\n{source}"
        )

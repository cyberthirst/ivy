import random

from vyper.ast import nodes as ast

from fuzzer.mutator.ast_mutator import AstMutator
from unparser.unparser import unparse

seed = random.randint(0, 2**32 - 1)
rng = random.Random(seed)
mutator = AstMutator(rng, generate=True)
module = mutator.mutate(ast.Module(body=[]))
source = unparse(module)

if mutator.type_generator.source_fragments:
    source = "\n\n".join(mutator.type_generator.source_fragments) + "\n\n" + source

print(f"# seed={seed}")
print(source)

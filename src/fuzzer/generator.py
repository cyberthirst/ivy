import random
from typing import Optional

from fuzzer.mutator.ast_mutator import AstMutator
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import DeploymentTrace, Env, Tx
from unparser.unparser import unparse
from vyper.ast import nodes as ast


def generate_scenario(seed: Optional[int] = None) -> Optional[Scenario]:
    """Generate a fresh random contract scenario from scratch."""
    rng = random.Random(seed)
    ast_mutator = AstMutator(rng, generate=True)

    try:
        module = ast_mutator.mutate(ast.Module(body=[]))
        source = unparse(module)
        if ast_mutator.type_generator.source_fragments:
            source = (
                "\n\n".join(ast_mutator.type_generator.source_fragments)
                + "\n\n"
                + source
            )
    except Exception:
        return None

    solc_json = {"sources": {"<generated>": {"content": source}}}
    trace = DeploymentTrace(
        deployment_type="source",
        calldata=None,
        value=0,
        solc_json=solc_json,
        blueprint_initcode_prefix=None,
        deployed_address="0x0000000000000000000000000000000000001234",
        deployment_succeeded=True,
        env=Env(tx=Tx(origin="0xC28B8a66397691f40C92271a9EBC04Cabc1ACcA1")),
        compiler_settings={"enable_decimals": True},
        compilation_xfails=list(ast_mutator.context.compilation_xfails),
        runtime_xfails=list(ast_mutator.context.runtime_xfails),
    )
    return Scenario(
        traces=[trace],
        dependencies=[],
        scenario_id=f"generated_{seed}",
        use_python_args=True,
    )

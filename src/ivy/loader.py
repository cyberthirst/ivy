import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import vyper
from vyper.cli.vyper_compile import get_search_paths
from vyper.compiler.input_bundle import (
    FileInput,
    FilesystemInputBundle,
)
from vyper.compiler.phases import CompilerData
from vyper.compiler.settings import Settings

from titanoboa.boa.contracts.vvm.vvm_contract import _detect_version

from ivy.vyper_contract import VyperDeployer, VyperContract


if TYPE_CHECKING:
    from vyper.semantics.analysis.base import ImportInfo

_Contract = Union[VyperContract]

_search_path = None


def set_search_path(path: list[str]):
    global _search_path
    _search_path = path



def compiler_data(
    source_code: str, contract_name: str, filename: str | Path, deployer=None, **kwargs
) -> CompilerData:
    global _disk_cache, _search_path

    file_input = FileInput(
        contents=source_code,
        source_id=-1,
        path=Path(contract_name),
        resolved_path=Path(filename),
    )
    search_paths = get_search_paths(_search_path)
    input_bundle = FilesystemInputBundle(search_paths)

    settings = Settings(**kwargs)
    return CompilerData(file_input, input_bundle, settings)


def load(filename: str | Path, *args, **kwargs) -> _Contract:  # type: ignore
    name = Path(filename).stem
    # TODO: investigate if we can just put name in the signature
    if "name" in kwargs:
        name = kwargs.pop("name")
    with open(filename) as f:
        return loads(f.read(), *args, name=name, **kwargs, filename=filename)


def loads(
    source_code,
    *args,
    as_blueprint=False,
    name=None,
    filename=None,
    compiler_args=None,
    **kwargs,
):
    d = loads_partial(source_code, name, filename=filename, compiler_args=compiler_args)
    if as_blueprint:
        return d.deploy_as_blueprint(**kwargs)
    else:
        return d.deploy(*args, **kwargs)



def loads_partial(
    source_code: str,
    name: str = None,
    filename: str | Path | None = None,
    dedent: bool = True,
    compiler_args: dict = None,
) -> VyperDeployer:
    name = name or "VyperContract"  # TODO handle this upstream in CompilerData
    filename = filename or "<unknown>"
    if dedent:
        source_code = textwrap.dedent(source_code)

    version = _detect_version(source_code)
    if version is not None and version != vyper.__version__:
        raise ValueError(
            f"Source code is for Vyper version {version}, but the installed version is {vyper.__version__}"
        )

    compiler_args = compiler_args or {}

    deployer_class = VyperDeployer
    data = compiler_data(source_code, name, filename, deployer_class, **compiler_args)
    return deployer_class(data, filename=filename)


def load_partial(filename: str, compiler_args=None):
    with open(filename) as f:
        return loads_partial(
            f.read(), name=filename, filename=filename, compiler_args=compiler_args
        )

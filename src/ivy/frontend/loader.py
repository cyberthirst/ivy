import textwrap
from pathlib import Path
from typing import Union, Dict, Any

import vyper
from vyper.cli.vyper_compile import get_search_paths
from vyper.cli.vyper_json import (
    get_inputs,
    get_output_formats,
    get_search_paths as get_json_search_paths,
)
from vyper.compiler.input_bundle import (
    FileInput,
    FilesystemInputBundle,
    JSONInputBundle,
)
from vyper.compiler.phases import CompilerData
from vyper.compiler.settings import Settings

from ivy.frontend.vyper_contract import VyperDeployer, VyperContract
from ivy.frontend.env import Env
from ivy.utils import _detect_version

_Contract = Union[VyperContract]

_search_path = None


def set_search_path(path: list[str]):
    global _search_path
    _search_path = path


def compiler_data(
    source_code: str,
    contract_name: str,
    filename: str | Path,
    input_bundle=None,
    **kwargs,
) -> CompilerData:
    global _disk_cache, _search_path

    file_input = FileInput(
        contents=source_code,
        source_id=-1,
        path=Path(contract_name),
        resolved_path=Path(filename),
    )
    search_paths = get_search_paths(_search_path)
    bundle = input_bundle or FilesystemInputBundle(search_paths)

    settings = Settings.from_dict(kwargs)
    return CompilerData(file_input, bundle, settings)


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
    input_bundle=None,
    encoded_constructor_args=None,
    **kwargs,
):
    d = loads_partial(
        source_code,
        name,
        filename=filename,
        compiler_args=compiler_args,
        input_bundle=input_bundle,
    )
    if as_blueprint:
        return d.deploy_as_blueprint(**kwargs)
    else:
        if encoded_constructor_args is not None:
            kwargs["encoded_constructor_args"] = encoded_constructor_args
        return d.deploy(*args, **kwargs)


def loads_partial(
    source_code: str,
    name: str = None,
    filename: str | Path | None = None,
    dedent: bool = True,
    compiler_args: dict = None,
    input_bundle=None,
    get_compiler_data=False,
) -> Union[VyperDeployer, CompilerData]:
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

    deployer_class = _get_deployer_class()
    data = compiler_data(
        source_code, name, filename, input_bundle=input_bundle, **compiler_args
    )

    if get_compiler_data:
        return data

    return deployer_class(data, filename=filename)


def load_partial(filename: str, compiler_args=None):
    with open(filename) as f:
        return loads_partial(
            f.read(), name=filename, filename=filename, compiler_args=compiler_args
        )


def _get_deployer_class():
    env = Env.get_singleton()
    return env.interpreter.deployer


def loads_from_solc_json(
    solc_json: Dict[str, Any],
    *args,
    as_blueprint=False,
    # TODO we should probably split getting the compiler data to a separate method
    get_compiler_data=False,
    **kwargs,
) -> Union[VyperContract, CompilerData]:
    """
    Load and deploy a contract from solc_json format using Vyper's JSON compilation machinery.

    Args:
        solc_json: Dictionary with 'sources' and 'settings' keys (Vyper JSON format)
        Other args are passed to contract deployment
    """
    sources = get_inputs(solc_json)
    output_formats = get_output_formats(solc_json)
    search_paths = get_json_search_paths(solc_json)

    compilation_targets = list(output_formats.keys())

    if not compilation_targets:
        raise ValueError("No compilation targets found in solc_json")

    # Use first target
    # Note: add multi-target support in the future if necessary
    target = compilation_targets[0]

    input_bundle = JSONInputBundle(sources, search_paths=search_paths)

    # Load the target file
    if target in input_bundle.input_json:
        # Path is already in the bundle, load it directly
        file = input_bundle._load_from_path(target, target)
    else:
        # Try loading with just the filename (relative to search paths)
        file = input_bundle.load_file(target.name)

    if not isinstance(file, FileInput):
        raise ValueError(f"Expected FileInput for {target}, got {type(file)}")

    name = target.stem
    d = loads_partial(
        file.contents,
        name=name,
        filename=str(target),
        input_bundle=input_bundle,
        dedent=False,
        get_compiler_data=get_compiler_data,
    )

    if get_compiler_data:
        return d

    # Deploy the contract
    if as_blueprint:
        return d.deploy_as_blueprint(**kwargs)
    else:
        return d.deploy(*args, **kwargs)

from typing import Optional, Generator

import pytest
from contextlib import contextmanager

from vyper.compiler.input_bundle import FilesystemInputBundle
from vyper.utils import keccak256

from ivy.frontend.env import Env
from ivy.frontend.loader import loads
from ivy.frontend.vyper_contract import VyperContract


@pytest.fixture(scope="module")
def clear_env():
    environment = Env().get_singleton()
    environment.clear_state()


@pytest.fixture(scope="module")
def env(clear_env):
    environment = Env().get_singleton()
    return environment


@pytest.fixture(scope="module")
def get_contract(clear_env):
    def fn(source_code, *args, **kwargs) -> VyperContract:
        return loads(source_code, *args, **kwargs)

    return fn


# adapted from vyper: https://github.com/vyperlang/vyper/blob/6843e7915729f3a3ea0d8c765dffa52033f5818e/tests/conftest.py#L306
@pytest.fixture(scope="module")
def tx_failed():
    @contextmanager
    # TODO make ivy-specific general exception
    def fn(exception=Exception, text=None):
        with pytest.raises(exception) as excinfo:
            yield

        if text:
            # TODO test equality
            assert text in str(excinfo.value), (text, excinfo.value)

    return fn


@pytest.fixture
def make_file(tmp_path):
    # writes file_contents to file_name, creating it in the
    # tmp_path directory. returns final path.
    def fn(file_name, file_contents):
        path = tmp_path / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(file_contents)

        return path

    return fn


# adapted from vyper: https://github.com/vyperlang/vyper/blob/6843e7915729f3a3ea0d8c765dffa52033f5818e/tests/conftest.py#L160-L159
# this can either be used for its side effects (to prepare a call
# to get_contract), or the result can be provided directly to
# compile_code / CompilerData.
@pytest.fixture
def make_input_bundle(tmp_path, make_file):
    def fn(sources_dict):
        for file_name, file_contents in sources_dict.items():
            make_file(file_name, file_contents)
        return FilesystemInputBundle([tmp_path])

    return fn


@pytest.fixture
def get_logs():
    def fn(
        contract: VyperContract,
        event_name: Optional[str] = None,
        vyper_compat: bool = True,
    ):
        logs = contract.get_logs(include_id=True)

        if vyper_compat:
            for log in logs:
                log.args = log.args_obj

        if event_name:
            return [log for log in logs if log.event == event_name]

        return logs

    return fn


@pytest.fixture
def keccak():
    return keccak256


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item) -> Generator:
    """Isolate tests by reverting the state of the environment after each test.
    """
    env = item.funcargs.get("env")
    if env:
        with env.anchor():
            yield
    else:
        yield

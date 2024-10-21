import pytest
from contextlib import contextmanager

from vyper.compiler.input_bundle import FilesystemInputBundle

from ivy.frontend.env import Env
from ivy.frontend.loader import loads


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
    def fn(source_code, *args, **kwargs):
        return loads(source_code, *args, **kwargs)

    return fn


# adapted from vyper: https://github.com/vyperlang/vyper/blob/6843e7915729f3a3ea0d8c765dffa52033f5818e/tests/conftest.py#L306
@pytest.fixture(scope="module")
def tx_failed():
    @contextmanager
    # TODO make ivy-specific general exception
    def fn(exception=Exception, exc_text=None):
        with pytest.raises(exception) as excinfo:
            yield

        if exc_text:
            # TODO test equality
            assert exc_text in str(excinfo.value), (exc_text, excinfo.value)

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

import pytest
from contextlib import contextmanager

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

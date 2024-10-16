import pytest

from ivy.frontend.env import Env
from ivy.frontend.loader import loads


@pytest.fixture(scope="module")
def clear_env():
    env = Env().get_singleton()
    env.clear_state()


@pytest.fixture(scope="module")
def get_contract(clear_env):
    def fn(source_code, *args, **kwargs):
        return loads(source_code, *args, **kwargs)

    return fn

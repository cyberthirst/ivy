import boa

from fuzzer.base_fuzzer import BaseFuzzer
from fuzzer.runner.boa_scenario_runner import BoaScenarioRunner, MinimalNoopGasMeter
from fuzzer.runner.multi_runner import MultiRunner
from fuzzer.runner.scenario import Scenario


def test_minimal_noop_gas_meter_interface() -> None:
    meter = MinimalNoopGasMeter(1234)
    assert meter.start_gas == 1234
    assert meter.gas_remaining == 1234
    assert meter.gas_refunded == 0

    meter.consume_gas(500, "test")
    meter.return_gas(100)
    meter.refund_gas(100)

    assert meter.gas_remaining == 1234
    assert meter.gas_refunded == 0


def test_boa_scenario_runner_sets_noop_gas_meter_class() -> None:
    previous_gas_meter_class = boa.env.get_gas_meter_class()
    runner = BoaScenarioRunner(gas_meter_class=MinimalNoopGasMeter)
    try:
        runner.run(Scenario(traces=[]))
        assert boa.env.get_gas_meter_class() is MinimalNoopGasMeter
    finally:
        boa.env.set_gas_meter_class(previous_gas_meter_class)


def test_multi_runner_accepts_custom_boa_gas_meter_class() -> None:
    previous_gas_meter_class = boa.env.get_gas_meter_class()
    runner = MultiRunner(boa_gas_meter_class=MinimalNoopGasMeter)
    try:
        runner.run(Scenario(traces=[]))
        assert boa.env.get_gas_meter_class() is MinimalNoopGasMeter
    finally:
        boa.env.set_gas_meter_class(previous_gas_meter_class)


def test_base_fuzzer_uses_noop_boa_gas_meter() -> None:
    fuzzer = BaseFuzzer()
    assert fuzzer.multi_runner.boa_gas_meter_class is MinimalNoopGasMeter

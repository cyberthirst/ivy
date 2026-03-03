from __future__ import annotations

from fuzzer.deduper import Deduper, fingerprint_error
from fuzzer.divergence_detector import Divergence, DivergenceType
from fuzzer.runner.base_scenario_runner import CallResult
from fuzzer.runner.scenario import Scenario
from fuzzer.trace_types import DeploymentTrace, Env, Tx


def _make_scenario(source: str) -> Scenario:
    return Scenario(
        traces=[
            DeploymentTrace(
                deployment_type="source",
                calldata=None,
                value=0,
                solc_json={"sources": {"test.vy": {"content": source}}},
                blueprint_initcode_prefix=None,
                deployed_address="0x0000000000000000000000000000000000000001",
                deployment_succeeded=True,
                env=Env(tx=Tx(origin="0xabc")),
            )
        ]
    )


def _capture_runtime_error_a(message: str) -> Exception:
    def inner() -> None:
        raise RuntimeError(message)

    try:
        inner()
    except Exception as exc:
        return exc
    raise AssertionError("unreachable")


def _capture_runtime_error_b(message: str) -> Exception:
    def inner_alt() -> None:
        raise RuntimeError(message)

    try:
        inner_alt()
    except Exception as exc:
        return exc
    raise AssertionError("unreachable")


def _capture_value_error(message: str) -> Exception:
    try:
        raise ValueError(message)
    except Exception as exc:
        return exc
    raise AssertionError("unreachable")


def _make_result(success: bool | None, error: Exception | None = None) -> CallResult | None:
    if success is None:
        return None
    return CallResult(success=success, error=error)


def _make_result_divergence(
    *,
    div_type: DivergenceType = DivergenceType.EXECUTION,
    ivy_success: bool | None,
    boa_success: bool | None,
    ivy_error: Exception | None = None,
    boa_error: Exception | None = None,
    runner: str = "boa",
    source: str = "@external\ndef foo() -> uint256:\n    return 1\n",
) -> Divergence:
    return Divergence(
        type=div_type,
        step=0,
        scenario=_make_scenario(source),
        divergent_runner=runner,
        ivy_result=_make_result(ivy_success, ivy_error),
        boa_result=_make_result(boa_success, boa_error),
        function="foo",
    )


def _make_xfail_divergence(
    *,
    expected: str,
    actual: str,
    reasons: list[str],
    source: str = "@external\ndef foo() -> uint256:\n    return 1\n",
) -> Divergence:
    return Divergence(
        type=DivergenceType.XFAIL,
        step=0,
        scenario=_make_scenario(source),
        xfail_expected=expected,
        xfail_actual=actual,
        xfail_reasons=reasons,
    )


def test_result_divergence_all_success_state_combinations():
    combos = [
        (True, True, False),
        (False, False, False),
        (None, None, False),
        (True, False, True),
        (False, True, True),
        (True, None, True),
        (None, True, True),
        (False, None, True),
        (None, False, True),
    ]

    for ivy_success, boa_success, should_dedup in combos:
        deduper = Deduper()
        divergence = _make_result_divergence(
            ivy_success=ivy_success,
            boa_success=boa_success,
            ivy_error=_capture_runtime_error_a("ivy failed"),
            boa_error=_capture_runtime_error_a("boa failed"),
        )

        first = deduper.check_divergence(divergence)
        second = deduper.check_divergence(divergence)

        if should_dedup:
            assert first.keep is True
            assert first.reason == "new_divergence"
            assert second.keep is False
            assert second.reason == "duplicate_divergence"
            assert first.fingerprint
            assert second.fingerprint == first.fingerprint
        else:
            assert first.keep is True
            assert second.keep is True
            assert first.reason == "both_succeeded_different_results"
            assert second.reason == "both_succeeded_different_results"
            assert first.fingerprint == ""
            assert second.fingerprint == ""


def test_when_boa_fails_runner_name_is_part_of_dedup_fingerprint():
    deduper = Deduper()
    error = _capture_runtime_error_a("shared error")

    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=error,
            runner="boa:default",
        )
    )
    duplicate_same_runner = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=error,
            runner="boa:default",
        )
    )
    different_runner = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=error,
            runner="boa:venom",
        )
    )

    assert first.keep is True
    assert duplicate_same_runner.keep is False
    assert different_runner.keep is True
    assert different_runner.reason == "new_divergence"


def test_when_ivy_fails_runner_name_is_ignored_for_dedup():
    deduper = Deduper()
    error = _capture_runtime_error_a("ivy error")

    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=False,
            boa_success=True,
            ivy_error=error,
            runner="boa:default",
        )
    )
    second_different_runner = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=False,
            boa_success=True,
            ivy_error=error,
            runner="boa:venom",
        )
    )

    assert first.keep is True
    assert second_different_runner.keep is False


def test_mixed_success_dedup_distinguishes_error_type_message_prefix_and_frames():
    deduper = Deduper()

    # Same first 20 chars and same stack shape should dedup.
    same_prefix_first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_a("01234567890123456789_A"),
        )
    )
    same_prefix_second = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_a("01234567890123456789_B"),
        )
    )
    assert same_prefix_first.keep is True
    assert same_prefix_second.keep is False

    # Different first 20 chars should not dedup.
    different_message = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_a("X1234567890123456789_B"),
        )
    )
    assert different_message.keep is True

    # Different exception type should not dedup.
    different_type = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_value_error("01234567890123456789_A"),
        )
    )
    assert different_type.keep is True

    # Different stack frames should not dedup.
    different_frames = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_b("01234567890123456789_A"),
        )
    )
    assert different_frames.keep is True


def test_divergence_type_is_part_of_fingerprint():
    deduper = Deduper()
    error = _capture_runtime_error_a("same")

    execution = deduper.check_divergence(
        _make_result_divergence(
            div_type=DivergenceType.EXECUTION,
            ivy_success=True,
            boa_success=False,
            boa_error=error,
        )
    )
    deployment = deduper.check_divergence(
        _make_result_divergence(
            div_type=DivergenceType.DEPLOYMENT,
            ivy_success=True,
            boa_success=False,
            boa_error=error,
        )
    )

    assert execution.keep is True
    assert deployment.keep is True
    assert execution.fingerprint != deployment.fingerprint


def test_xfail_divergence_deduplication_keys():
    deduper = Deduper()

    first = deduper.check_divergence(
        _make_xfail_divergence(
            expected="runtime",
            actual="ivy: success",
            reasons=["reason-a", "reason-b"],
        )
    )
    duplicate = deduper.check_divergence(
        _make_xfail_divergence(
            expected="runtime",
            actual="ivy: success",
            reasons=["reason-a", "reason-b"],
        )
    )
    different_expected = deduper.check_divergence(
        _make_xfail_divergence(
            expected="compilation",
            actual="ivy: success",
            reasons=["reason-a", "reason-b"],
        )
    )
    different_actual = deduper.check_divergence(
        _make_xfail_divergence(
            expected="runtime",
            actual="boa: success",
            reasons=["reason-a", "reason-b"],
        )
    )
    different_reason_order = deduper.check_divergence(
        _make_xfail_divergence(
            expected="runtime",
            actual="ivy: success",
            reasons=["reason-b", "reason-a"],
        )
    )

    assert first.keep is True
    assert duplicate.keep is False
    assert different_expected.keep is True
    assert different_actual.keep is True
    assert different_reason_order.keep is True


def test_unknown_divergence_type_is_always_kept():
    deduper = Deduper()
    divergence = Divergence(
        type="unknown",  # type: ignore[arg-type]
        step=0,
        scenario=_make_scenario("@external\ndef foo() -> uint256:\n    return 1\n"),
    )

    first = deduper.check_divergence(divergence)
    second = deduper.check_divergence(divergence)

    assert first.keep is True
    assert second.keep is True
    assert first.reason == "unknown_type"
    assert second.reason == "unknown_type"
    assert first.fingerprint == ""
    assert second.fingerprint == ""


def test_similar_contracts_do_not_dedup_when_success_states_match():
    deduper = Deduper()

    one = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=True,
            source="@external\ndef foo() -> uint256:\n    return 1\n",
        )
    )
    two = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=True,
            source="@external\ndef foo() -> uint256:\n    return 2\n",
        )
    )
    three = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=False,
            boa_success=False,
            source="@external\ndef foo() -> uint256:\n    return 3\n",
            ivy_error=_capture_runtime_error_a("ivy fail"),
            boa_error=_capture_runtime_error_a("boa fail"),
        )
    )
    four = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=False,
            boa_success=False,
            source="@external\ndef foo() -> uint256:\n    return 4\n",
            ivy_error=_capture_runtime_error_a("ivy fail"),
            boa_error=_capture_runtime_error_a("boa fail"),
        )
    )

    assert one.keep is True
    assert two.keep is True
    assert three.keep is True
    assert four.keep is True
    assert one.fingerprint == ""
    assert two.fingerprint == ""
    assert three.fingerprint == ""
    assert four.fingerprint == ""


def test_boa_error_fingerprint_uses_structured_fields():
    from boa.contracts.base_evm_contract import BoaError, StackTrace

    from boa.contracts.vyper.vyper_contract import ErrorDetail

    class FakeVMError(Exception):
        pass

    class Revert(FakeVMError):
        pass

    class InvalidInstruction(FakeVMError):
        pass

    def make_error_detail(vm_error, error_detail_str):
        ed = ErrorDetail.__new__(ErrorDetail)
        ed.vm_error = vm_error
        ed.error_detail = error_detail_str
        return ed

    def make_boa_error(vm_error, error_detail_str):
        st = StackTrace([make_error_detail(vm_error, error_detail_str)])
        err = BoaError.__new__(BoaError)
        err.stack_trace = st
        err.call_trace = None
        return err

    # Different vm_error types produce different fingerprints
    revert_err = make_boa_error(Revert(), "safeadd")
    invalid_err = make_boa_error(InvalidInstruction(), "safeadd")
    fp_revert = fingerprint_error(revert_err)
    fp_invalid = fingerprint_error(invalid_err)
    assert fp_revert != fp_invalid
    assert fp_revert[0] == "Revert"
    assert fp_invalid[0] == "InvalidInstruction"

    # Different error_detail strings produce different fingerprints
    safeadd_err = make_boa_error(Revert(), "safeadd")
    bounds_err = make_boa_error(Revert(), "int128 bounds check")
    fp_safeadd = fingerprint_error(safeadd_err)
    fp_bounds = fingerprint_error(bounds_err)
    assert fp_safeadd != fp_bounds
    assert fp_safeadd[1] == "safeadd"
    assert fp_bounds[1] == "int128 bounds check"

    # Same vm_error type + same error_detail produce identical fingerprints
    err_a = make_boa_error(Revert(), "safeadd")
    err_b = make_boa_error(Revert(), "safeadd")
    assert fingerprint_error(err_a) == fingerprint_error(err_b)

    # Frames are always empty for BoaError fingerprints
    assert fp_revert[2] == ()


def test_boa_error_dedup_distinguishes_different_reverts():
    from boa.contracts.base_evm_contract import BoaError, StackTrace
    from boa.contracts.vyper.vyper_contract import ErrorDetail

    class Revert(Exception):
        pass

    def make_boa_error(error_detail_str):
        ed = ErrorDetail.__new__(ErrorDetail)
        ed.vm_error = Revert()
        ed.error_detail = error_detail_str
        st = StackTrace([ed])
        err = BoaError.__new__(BoaError)
        err.stack_trace = st
        err.call_trace = None
        return err

    deduper = Deduper()

    # Two divergences with different BoaError error_detail should both be kept
    div_safeadd = _make_result_divergence(
        ivy_success=True,
        boa_success=False,
        boa_error=make_boa_error("safeadd"),
    )
    div_bounds = _make_result_divergence(
        ivy_success=True,
        boa_success=False,
        boa_error=make_boa_error("int128 bounds check"),
    )
    div_safeadd_dup = _make_result_divergence(
        ivy_success=True,
        boa_success=False,
        boa_error=make_boa_error("safeadd"),
    )

    first = deduper.check_divergence(div_safeadd)
    second = deduper.check_divergence(div_bounds)
    third = deduper.check_divergence(div_safeadd_dup)

    assert first.keep is True
    assert second.keep is True  # different error_detail
    assert third.keep is False  # duplicate of first

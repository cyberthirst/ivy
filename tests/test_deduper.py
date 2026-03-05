from __future__ import annotations

from fuzzer.deduper import BoaFrameFP, Deduper, UnknownFrameFP, fingerprint_error
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
    reason: str | None = None,
    source: str = "@external\ndef foo() -> uint256:\n    return 1\n",
    contract_fingerprints: tuple[str, ...] = (),
) -> Divergence:
    return Divergence(
        type=div_type,
        step=0,
        scenario=_make_scenario(source),
        divergent_runner=runner,
        ivy_result=_make_result(ivy_success, ivy_error),
        boa_result=_make_result(boa_success, boa_error),
        function="foo",
        reason=reason,
        contract_fingerprints=contract_fingerprints,
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
    # Mixed success: one ok, one failed → dedups on error fingerprint
    dedup_combos = [
        (True, False),
        (False, True),
    ]

    for ivy_success, boa_success in dedup_combos:
        deduper = Deduper()
        divergence = _make_result_divergence(
            ivy_success=ivy_success,
            boa_success=boa_success,
            ivy_error=_capture_runtime_error_a("ivy failed"),
            boa_error=_capture_runtime_error_a("boa failed"),
        )

        first = deduper.check_divergence(divergence)
        second = deduper.check_divergence(divergence)

        assert first.keep is True
        assert first.reason == "new_divergence"
        assert second.keep is False
        assert second.reason == "duplicate_divergence"
        assert first.fingerprint
        assert second.fingerprint == first.fingerprint

    # Same outcome: both ok or both failed → no dedup (without contract fingerprints)
    no_dedup_combos = [
        (True, True),
        (False, False),
    ]

    for ivy_success, boa_success in no_dedup_combos:
        deduper = Deduper()
        divergence = _make_result_divergence(
            ivy_success=ivy_success,
            boa_success=boa_success,
            ivy_error=_capture_runtime_error_a("ivy failed"),
            boa_error=_capture_runtime_error_a("boa failed"),
        )

        first = deduper.check_divergence(divergence)
        second = deduper.check_divergence(divergence)

        assert first.keep is True
        assert second.keep is True
        assert first.reason == "both_succeeded_different_results"
        assert first.fingerprint == ""

    # One or both results missing → missing_result (no dedup)
    missing_combos = [
        (True, None),
        (None, True),
        (False, None),
        (None, False),
    ]

    for ivy_success, boa_success in missing_combos:
        deduper = Deduper()
        divergence = _make_result_divergence(
            ivy_success=ivy_success,
            boa_success=boa_success,
            ivy_error=_capture_runtime_error_a("ivy failed"),
            boa_error=_capture_runtime_error_a("boa failed"),
        )
        first = deduper.check_divergence(divergence)
        assert first.keep is True
        assert first.reason == "missing_result"


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


def test_mixed_success_dedup_distinguishes_error_type_and_frames():
    deduper = Deduper()

    # Same error type and stack shape should dedup regardless of message.
    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_a("message A"),
        )
    )
    same_type_different_msg = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_a("completely different message"),
        )
    )
    assert first.keep is True
    assert same_type_different_msg.keep is False

    # Different exception type should not dedup.
    different_type = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_value_error("message A"),
        )
    )
    assert different_type.keep is True

    # Different stack frames should not dedup.
    different_frames = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=_capture_runtime_error_b("message A"),
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


def _capture_boa_error(source: str, function_name: str, *args) -> Exception:
    import boa

    contract = boa.loads(source)
    fn = getattr(contract, function_name)
    try:
        fn(*args)
    except Exception as exc:
        return exc
    raise AssertionError("expected Boa call to fail")


def _make_synthetic_boa_error(stack_frames: list[object]) -> Exception:
    from boa.contracts.base_evm_contract import BoaError, StackTrace

    err = BoaError.__new__(BoaError)
    err.stack_trace = StackTrace(stack_frames)
    err.call_trace = None
    return err


def _make_error_detail(
    vm_error: Exception, *, error_detail: str | None, ast_source: object = None
) -> object:
    from boa.contracts.vyper.vyper_contract import ErrorDetail

    frame = ErrorDetail.__new__(ErrorDetail)
    frame.vm_error = vm_error
    frame.error_detail = error_detail
    frame.ast_source = ast_source
    return frame


def test_boa_error_fingerprint_top_level_similar_vs_different_expressions():
    max_uint256 = 2**256 - 1

    add_const_one = """
@external
def foo(x: uint256) -> uint256:
    return x + 1
"""
    add_const_two = """
@external
def foo(x: uint256) -> uint256:
    return x + 2
"""
    add_vars = """
@external
def foo(x: uint256, y: uint256) -> uint256:
    return x + y
"""

    err_add_const_one = _capture_boa_error(add_const_one, "foo", max_uint256)
    err_add_const_two = _capture_boa_error(add_const_two, "foo", max_uint256 - 1)
    err_add_vars = _capture_boa_error(add_vars, "foo", max_uint256, 1)

    fp_add_const_one = fingerprint_error(err_add_const_one)
    fp_add_const_two = fingerprint_error(err_add_const_two)
    fp_add_vars = fingerprint_error(err_add_vars)

    # Similar expressions should dedup to the same fingerprint.
    assert fp_add_const_one == fp_add_const_two
    # Different expression shape should produce a different fingerprint.
    assert fp_add_const_one != fp_add_vars

    frame = fp_add_const_one.frames[0]
    assert isinstance(frame, BoaFrameFP)
    assert frame.ast_fingerprint is not None and "Add" in frame.ast_fingerprint
    assert frame.error_detail == "safeadd"
    assert "Revert(" in frame.pretty_vm_reason


def test_boa_error_fingerprint_nested_call_keeps_innermost_first():
    source = """
interface Self:
    def do_sub(x: uint256, y: uint256) -> uint256: nonpayable

@external
def entry(x: uint256, y: uint256) -> uint256:
    return extcall Self(self).do_sub(x, y)

@external
def do_sub(x: uint256, y: uint256) -> uint256:
    return x - y
"""

    err = _capture_boa_error(source, "entry", 5, 10)
    fp = fingerprint_error(err)

    assert fp.error_type == "BoaError"
    assert len(fp.frames) == 2

    innermost, caller = fp.frames
    assert isinstance(innermost, BoaFrameFP)
    assert innermost.ast_fingerprint is not None and "Sub" in innermost.ast_fingerprint
    assert innermost.error_detail == "safesub"
    assert "Revert(" in innermost.pretty_vm_reason

    assert isinstance(caller, BoaFrameFP)
    assert caller.ast_fingerprint is not None and "Call" in caller.ast_fingerprint
    assert caller.error_detail == "external call failed"
    assert "Revert(" in caller.pretty_vm_reason


def test_boa_error_fingerprint_uses_first_three_frames_only():
    frames = [
        _make_error_detail(RuntimeError("vm-0"), error_detail="d0"),
        _make_error_detail(RuntimeError("vm-1"), error_detail="d1"),
        _make_error_detail(RuntimeError("vm-2"), error_detail="d2"),
        _make_error_detail(RuntimeError("vm-3"), error_detail="d3"),
    ]
    err = _make_synthetic_boa_error(frames)

    fp = fingerprint_error(err)

    assert len(fp.frames) == 3
    assert [frame.error_detail for frame in fp.frames] == ["d0", "d1", "d2"]
    assert all(frame.ast_fingerprint is None for frame in fp.frames)
    assert all("RuntimeError(" in frame.pretty_vm_reason for frame in fp.frames)


def test_boa_error_fingerprint_marks_unknown_string_frames():
    err = _make_synthetic_boa_error(
        [
            "Unknown contract 0x1234",
            _make_error_detail(ValueError("boom"), error_detail=None),
        ]
    )

    fp = fingerprint_error(err)

    assert fp.error_type == "BoaError"
    assert isinstance(fp.frames[0], UnknownFrameFP)
    assert isinstance(fp.frames[1], BoaFrameFP)
    assert fp.frames[1].ast_fingerprint is None
    assert fp.frames[1].error_detail is None
    assert fp.frames[1].pretty_vm_reason == "ValueError('boom')"


def test_boa_error_dedup_distinguishes_different_reverts():
    from boa.contracts.base_evm_contract import BoaError, StackTrace
    from boa.contracts.vyper.vyper_contract import ErrorDetail
    from vyper.ast.nodes import Name

    class Revert(Exception):
        pass

    dummy_ast = Name.__new__(Name)
    dummy_ast.id = "x"

    def make_boa_error(error_detail_str):
        ed = ErrorDetail.__new__(ErrorDetail)
        ed.vm_error = Revert()
        ed.error_detail = error_detail_str
        ed.ast_source = dummy_ast
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
        reason="success_revert_mismatch",
    )
    div_bounds = _make_result_divergence(
        ivy_success=True,
        boa_success=False,
        boa_error=make_boa_error("int128 bounds check"),
        reason="success_revert_mismatch",
    )
    div_safeadd_dup = _make_result_divergence(
        ivy_success=True,
        boa_success=False,
        boa_error=make_boa_error("safeadd"),
        reason="success_revert_mismatch",
    )

    first = deduper.check_divergence(div_safeadd)
    second = deduper.check_divergence(div_bounds)
    third = deduper.check_divergence(div_safeadd_dup)

    assert first.keep is True
    assert second.keep is True  # different error_detail
    assert third.keep is False  # duplicate of first


def test_both_ok_dedups_on_contract_fingerprints():
    deduper = Deduper()

    fp_a = ("abc123",)
    fp_b = ("def456",)

    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=True,
            contract_fingerprints=fp_a,
        )
    )
    duplicate = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=True,
            contract_fingerprints=fp_a,
        )
    )
    different_source = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=True,
            contract_fingerprints=fp_b,
        )
    )

    assert first.keep is True
    assert first.reason == "new_divergence"
    assert first.fingerprint != ""
    assert duplicate.keep is False
    assert duplicate.reason == "duplicate_divergence"
    assert different_source.keep is True


def test_both_ok_no_fingerprints_does_not_dedup():
    deduper = Deduper()

    first = deduper.check_divergence(
        _make_result_divergence(ivy_success=True, boa_success=True)
    )
    second = deduper.check_divergence(
        _make_result_divergence(ivy_success=True, boa_success=True)
    )

    assert first.keep is True
    assert second.keep is True
    assert first.fingerprint == ""


def test_boa_error_no_ast_dedups_on_contract_fingerprints():
    deduper = Deduper()

    fp_a = ("abc123",)

    err = _make_synthetic_boa_error(["Unknown contract 0x1234"])

    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=err,
            reason="success_revert_mismatch",
            contract_fingerprints=fp_a,
        )
    )
    duplicate = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=err,
            reason="success_revert_mismatch",
            contract_fingerprints=fp_a,
        )
    )

    assert first.keep is True
    assert first.reason == "new_divergence"
    assert first.fingerprint != ""
    assert duplicate.keep is False
    assert duplicate.reason == "duplicate_divergence"


def test_boa_error_no_ast_no_fingerprints_does_not_dedup():
    deduper = Deduper()

    err = _make_synthetic_boa_error(["Unknown contract 0x1234"])

    first = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=err,
            reason="success_revert_mismatch",
        )
    )
    second = deduper.check_divergence(
        _make_result_divergence(
            ivy_success=True,
            boa_success=False,
            boa_error=err,
            reason="success_revert_mismatch",
        )
    )

    assert first.keep is True
    assert second.keep is True
    assert first.reason == "boa_error_no_ast"

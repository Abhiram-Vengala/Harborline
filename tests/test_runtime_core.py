"""
Stage 1 core: state machine, episode, budgets, termination, loop detector,
failure taxonomy. Pure unit tests, no model or tools involved.

Run: python tests/test_runtime_core.py   (or: python -m pytest tests/test_runtime_core.py)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runtime.budgets import Budget, BudgetTracker
from harness.runtime.episode import Episode
from harness.runtime.failures import FailureKind, Subsystem
from harness.runtime.loop_detector import LoopDetector
from harness.runtime.states import (
    TERMINAL_STATES, TRANSITIONS, EpisodeState, IllegalTransitionError, is_legal,
)
from harness.runtime.termination import (
    FINAL_STATE, REASON_FOR_FAILURE, TerminationController, TerminationReason,
)

S = EpisodeState


def raises(exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


# ---- state machine -----------------------------------------------------

def test_every_state_has_a_table_entry():
    assert set(TRANSITIONS) == set(EpisodeState)


def test_terminal_states_have_no_exits():
    assert TERMINAL_STATES == {S.COMPLETED, S.FAILED, S.TERMINATED}
    for s in TERMINAL_STATES:
        assert TRANSITIONS[s] == frozenset()


def test_completed_only_reachable_from_verification():
    sources = [s for s, nxt in TRANSITIONS.items() if S.COMPLETED in nxt]
    assert sources == [S.VERIFICATION]


def test_every_illegal_transition_is_refused_and_leaves_state_unchanged():
    """Forces every (from, to) pair NOT in the table, one by one."""
    checked = 0
    for src, dst in itertools.product(EpisodeState, EpisodeState):
        if is_legal(src, dst):
            continue
        ep = Episode("t")
        ep.state = src  # force the episode into `src`
        err = raises(IllegalTransitionError, ep.transition, dst)
        assert err.current == src and err.attempted == dst
        assert ep.state == src, f"{src}->{dst} changed state despite raising"
        assert ep.transitions == []
        checked += 1
    assert checked > 80  # 121 pairs minus legal ones


def test_every_legal_transition_is_accepted():
    for src, nxts in TRANSITIONS.items():
        for dst in nxts:
            ep = Episode("t")
            ep.state = src
            ep.transition(dst)
            assert ep.state == dst


def test_named_illegal_examples():
    ep = Episode("t")
    raises(IllegalTransitionError, ep.transition, S.COMPLETED)      # skip everything
    raises(IllegalTransitionError, ep.transition, S.TOOL_EXECUTION)  # skip policy check
    ep.transition(S.CONTEXT_BUILDING)
    raises(IllegalTransitionError, ep.transition, S.VERIFICATION)    # no model call yet


def test_terminal_episode_is_frozen():
    ep = Episode("t")
    ep.transition(S.TERMINATED)
    assert ep.is_terminal
    for s in EpisodeState:
        raises(IllegalTransitionError, ep.transition, s)


def test_happy_path_records_transitions():
    ep = Episode("t")
    for s in (S.CONTEXT_BUILDING, S.MODEL_CALL, S.POLICY_CHECK, S.TOOL_EXECUTION,
              S.OBSERVATION, S.CONTEXT_BUILDING, S.MODEL_CALL, S.VERIFICATION, S.COMPLETED):
        ep.transition(s)
    assert [t.to_state for t in ep.transitions][-1] == S.COMPLETED
    assert ep.transitions[0].from_state == S.CREATED


# ---- episode -------------------------------------------------------------

def test_episode_identity_and_serialization():
    a, b = Episode("task a"), Episode("task b")
    assert a.episode_id != b.episode_id
    assert Episode("x", episode_id="ep-1").episode_id == "ep-1"
    a.transition(S.CONTEXT_BUILDING)
    a.record_failure(FailureKind.MODEL_FAILURE, Subsystem.MODEL, "model_call", "boom")
    blob = json.dumps(a.to_dict())  # must be JSON-safe for later replay work
    assert "model_failure" in blob


def test_failure_links_episode_turn_action_subsystem_cause():
    ep = Episode("t")
    ep.transition(S.CONTEXT_BUILDING)
    ep.turn = 3
    f = ep.record_failure(FailureKind.TOOL_TIMEOUT, Subsystem.TOOL, "tool_call:search", "hung")
    assert (f.episode_id, f.turn, f.action) == (ep.episode_id, 3, "tool_call:search")
    assert (f.subsystem, f.kind, f.root_cause) == (Subsystem.TOOL, FailureKind.TOOL_TIMEOUT, "hung")
    assert f.detected_in == S.CONTEXT_BUILDING


def test_request_abort_only_accepts_abort_reasons():
    ep = Episode("t")
    raises(ValueError, ep.request_abort, TerminationReason.MAX_TURNS)
    ep.request_abort(TerminationReason.USER_ABORT, "stop")
    assert ep.abort_reason == TerminationReason.USER_ABORT


def test_success_cannot_be_declared_outside_verification():
    ep = Episode("t")
    ctl = TerminationController(BudgetTracker(Budget()))
    raises(IllegalTransitionError, ep.finish, ctl.success())
    assert ep.termination is None


def test_explain_mentions_reason_and_last_failure():
    ep = Episode("t")
    ep.transition(S.CONTEXT_BUILDING)
    ep.record_failure(FailureKind.MODEL_FAILURE, Subsystem.MODEL, "model_call", "garbage")
    ctl = TerminationController(BudgetTracker(Budget()))
    ep.finish(ctl.for_failure(FailureKind.MODEL_FAILURE, "retries exhausted"))
    text = ep.explain()
    assert "model_failure" in text and "garbage" in text and "failed" in text


# ---- budgets -------------------------------------------------------------

def test_budget_validation():
    raises(ValueError, Budget, max_turns=-1)
    raises(ValueError, Budget, max_cost_usd=-0.1)


def test_each_budget_trips_with_its_own_reason():
    t = BudgetTracker(Budget(max_turns=2))
    t.begin_turn(); t.begin_turn()
    assert t.exceeded() is None            # at the limit is fine
    t.begin_turn()
    assert t.exceeded()[0] == TerminationReason.MAX_TURNS

    t = BudgetTracker(Budget(max_tool_calls=1))
    t.record_tool_call(); assert t.exceeded() is None
    t.record_tool_call(); assert t.exceeded()[0] == TerminationReason.RESOURCE_LIMIT

    t = BudgetTracker(Budget(max_tokens=100))
    t.record_usage(60, 40); assert t.exceeded() is None
    t.record_usage(1, 0); assert t.exceeded()[0] == TerminationReason.RESOURCE_LIMIT

    t = BudgetTracker(Budget(max_cost_usd=0.01))
    t.record_usage(0, 0, 0.01); assert t.exceeded() is None
    t.record_usage(0, 0, 0.001); assert t.exceeded()[0] == TerminationReason.COST_LIMIT


def test_wall_clock_budget_uses_injected_clock():
    now = [0.0]
    t = BudgetTracker(Budget(max_runtime_s=10), clock=lambda: now[0])
    now[0] = 10.0; assert t.exceeded() is None
    now[0] = 10.5; assert t.exceeded()[0] == TerminationReason.TIMEOUT


def test_retry_counter_and_reset():
    t = BudgetTracker(Budget(max_retries=2))
    t.record_retry(); t.record_retry()
    assert not t.retries_exhausted()
    t.record_retry()
    assert t.retries_exhausted()
    t.reset_retries()
    assert not t.retries_exhausted()


# ---- termination ---------------------------------------------------------

def test_every_reason_has_a_final_state_and_only_success_completes():
    assert set(FINAL_STATE) == set(TerminationReason)
    completes = [r for r, s in FINAL_STATE.items() if s == S.COMPLETED]
    assert completes == [TerminationReason.SUCCESS_VERIFIED]
    for r, s in FINAL_STATE.items():
        assert s in TERMINAL_STATES


def test_every_failure_kind_maps_to_a_non_success_reason():
    assert set(REASON_FOR_FAILURE) == set(FailureKind)
    assert TerminationReason.SUCCESS_VERIFIED not in REASON_FOR_FAILURE.values()


def test_controller_poll_priority_abort_then_budget():
    ep = Episode("t")
    tracker = BudgetTracker(Budget(max_turns=0))
    tracker.begin_turn()
    ctl = TerminationController(tracker)
    assert ctl.poll(ep).reason == TerminationReason.MAX_TURNS
    ep.request_abort(TerminationReason.USER_ABORT)
    assert ctl.poll(ep).reason == TerminationReason.USER_ABORT


def test_controller_poll_quiet_when_nothing_wrong():
    assert TerminationController(BudgetTracker(Budget())).poll(Episode("t")) is None


# ---- loop detector -------------------------------------------------------

def test_loop_detector_trips_on_nth_identical_call():
    d = LoopDetector(threshold=3)
    assert not d.record("search", {"q": "a"})
    assert not d.record("search", {"q": "a"})
    assert d.record("search", {"q": "a"})


def test_loop_detector_ignores_different_args_and_key_order():
    d = LoopDetector(threshold=3)
    for i in range(10):
        assert not d.record("search", {"q": i})
    d = LoopDetector(threshold=2)
    assert not d.record("t", {"a": 1, "b": 2})
    assert d.record("t", {"b": 2, "a": 1})


def test_loop_detector_window_forgets_old_calls():
    d = LoopDetector(threshold=3, window=4)
    d.record("x", {}); d.record("x", {})
    for i in range(4):
        d.record("y", {"i": i})
    assert not d.record("x", {})  # the earlier x calls aged out


def test_loop_detector_validation():
    raises(ValueError, LoopDetector, threshold=1)
    raises(ValueError, LoopDetector, threshold=5, window=3)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001
                failed += 1; print(f"{name}: FAIL {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
"""Unit tests for every scorer against hand-built trajectories."""

import json

from trajectory_evals.scoring import (
    score_arguments,
    score_efficiency,
    score_grounding,
    score_selection,
)
from trajectory_evals.scoring.answer import score_answer
from trajectory_evals.spec import (
    AnswerExpectations,
    ArgCheck,
    NumericFact,
    OrderConstraint,
    Task,
    ToolExpectations,
)
from trajectory_evals.trajectory import ToolCall, Trajectory


def make_task(**kwargs) -> Task:
    defaults = {"id": "t", "prompt": "p", "optimal_calls": 2}
    defaults.update(kwargs)
    return Task(**defaults)


def make_trajectory(calls, answer="") -> Trajectory:
    return Trajectory(task_id="t", runner="test", tool_calls=calls, final_answer=answer)


def call(tool, arguments=None, result=None, is_error=False) -> ToolCall:
    return ToolCall(
        tool=tool,
        arguments=arguments or {},
        result_json=json.dumps(result if result is not None else {}),
        is_error=is_error,
    )


# -- selection ---------------------------------------------------------------


def test_selection_perfect():
    task = make_task(tools=ToolExpectations(must_call=["a", "b"]))
    score = score_selection(task, make_trajectory([call("a"), call("b")]))
    assert score.score == 1.0


def test_selection_missing_required():
    task = make_task(tools=ToolExpectations(must_call=["a", "b"]))
    score = score_selection(task, make_trajectory([call("a")]))
    assert score.score < 1.0
    assert any("missing required tool: b" in d for d in score.details)


def test_selection_forbidden_halves_score():
    task = make_task(tools=ToolExpectations(must_call=["a"], must_not_call=["rm"]))
    good = score_selection(task, make_trajectory([call("a")]))
    bad = score_selection(task, make_trajectory([call("a"), call("rm")]))
    assert bad.score < good.score
    assert any("forbidden" in d for d in bad.details)


def test_selection_order_violation():
    task = make_task(
        tools=ToolExpectations(
            must_call=["resample", "detect"],
            order=[OrderConstraint(before="resample", after="detect")],
        )
    )
    wrong = score_selection(task, make_trajectory([call("detect"), call("resample")]))
    assert wrong.score < 1.0
    assert any("order violation" in d for d in wrong.details)


def test_selection_unjustified_extra_penalized():
    task = make_task(tools=ToolExpectations(must_call=["a"], may_call=["b"]))
    ok = score_selection(task, make_trajectory([call("a"), call("b")]))
    extra = score_selection(task, make_trajectory([call("a"), call("z")]))
    assert ok.score == 1.0
    assert extra.score < 1.0


# -- arguments ---------------------------------------------------------------


def test_arguments_eq_and_approx():
    task = make_task(
        arg_checks=[
            ArgCheck(tool="detect", path="method", op="eq", value="stl_residual"),
            ArgCheck(tool="detect", path="period", op="approx", value=288, tolerance=0),
        ]
    )
    good = make_trajectory([call("detect", {"method": "stl_residual", "period": 288})])
    bad = make_trajectory([call("detect", {"method": "zscore", "period": 288})])
    assert score_arguments(task, good).score == 1.0
    assert score_arguments(task, bad).score == 0.5


def test_arguments_missing_tool_and_missing_arg():
    task = make_task(arg_checks=[ArgCheck(tool="detect", path="period", op="eq", value=288)])
    never_called = make_trajectory([call("other")])
    no_arg = make_trajectory([call("detect", {})])
    assert score_arguments(task, never_called).score == 0.0
    assert score_arguments(task, no_arg).score == 0.0


def test_arguments_lte():
    task = make_task(arg_checks=[ArgCheck(tool="w", path="limit", op="lte", value=100)])
    assert score_arguments(task, make_trajectory([call("w", {"limit": 50})])).score == 1.0
    assert score_arguments(task, make_trajectory([call("w", {"limit": 500})])).score == 0.0


# -- grounding ---------------------------------------------------------------


def test_grounding_supported_numbers():
    traj = make_trajectory(
        [call("a", result={"mae": 2.231, "n": 4})],
        answer="MAE is 2.23 across 4 anomalies.",
    )
    assert score_grounding(make_task(), traj).score == 1.0


def test_grounding_catches_hallucinated_number():
    traj = make_trajectory(
        [call("a", result={"mae": 2.231})],
        answer="MAE is 2.23 and accuracy improved 97%.",
    )
    score = score_grounding(make_task(), traj)
    assert score.score < 1.0
    assert any("97" in d for d in score.details)


def test_grounding_timestamps_checked_as_dates_not_digits():
    traj = make_trajectory(
        [call("a", result={"gap_start": "2026-06-04T11:15:00"})],
        answer="The gap begins at 2026-06-04T11:15.",
    )
    assert score_grounding(make_task(), traj).score == 1.0


def test_grounding_unknown_timestamp_flagged():
    traj = make_trajectory(
        [call("a", result={"gap_start": "2026-06-04T11:15:00"})],
        answer="The gap begins at 2031-01-01T00:00.",
    )
    score = score_grounding(make_task(), traj)
    assert score.score < 1.0


def test_grounding_empty_answer_scores_zero():
    traj = make_trajectory([call("a", result={"x": 1})], answer="  ")
    assert score_grounding(make_task(), traj).score == 0.0


# -- efficiency --------------------------------------------------------------


def test_efficiency_optimal():
    task = make_task(optimal_calls=2)
    assert score_efficiency(task, make_trajectory([call("a"), call("b")])).score == 1.0


def test_efficiency_redundant_identical_call_penalized():
    task = make_task(optimal_calls=2)
    traj = make_trajectory([call("a", {"x": 1}), call("a", {"x": 1}), call("b")])
    score = score_efficiency(task, traj)
    assert score.score < 0.6
    assert any("redundant" in d for d in score.details)


# -- answer ------------------------------------------------------------------


def test_answer_contains_and_facts():
    task = make_task(
        answer=AnswerExpectations(
            contains=["gap"],
            numeric_facts=[NumericFact(value=24, tolerance=1)],
        )
    )
    good = make_trajectory([], answer="Found a gap of about 24 points.")
    partial = make_trajectory([], answer="Found a gap.")
    assert score_answer(task, good).score == 1.0
    assert score_answer(task, partial).score == 0.5

"""Tool selection: did the agent reach for the right tools, in a sane order?"""

from __future__ import annotations

from ..spec import Task
from ..trajectory import Trajectory
from .base import DimensionScore


def score_selection(task: Task, trajectory: Trajectory) -> DimensionScore:
    exp = task.tools
    called = trajectory.called_tools()
    called_set = set(called)
    details: list[str] = []

    # Recall over required tools.
    missing = [t for t in exp.must_call if t not in called_set]
    recall = 1.0 - len(missing) / len(exp.must_call) if exp.must_call else 1.0
    details += [f"missing required tool: {t}" for t in missing]

    # Precision: calls outside must/may are unjustified.
    allowed = set(exp.must_call) | set(exp.may_call)
    if allowed:
        unjustified = [t for t in called if t not in allowed]
        precision = 1.0 - len(unjustified) / len(called) if called else 0.0
        details += [f"unjustified call: {t}" for t in sorted(set(unjustified))]
    else:
        precision = 1.0

    # Hard violations.
    forbidden = [t for t in exp.must_not_call if t in called_set]
    details += [f"called forbidden tool: {t}" for t in forbidden]

    order_violations = []
    for constraint in exp.order:
        if (
            constraint.before in called_set
            and constraint.after in called_set
            and called.index(constraint.before) > called.index(constraint.after)
        ):
            order_violations.append(f"{constraint.before} must precede {constraint.after}")
    details += [f"order violation: {v}" for v in order_violations]

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    score = f1
    if forbidden:
        score *= 0.5 ** len(forbidden)
    if order_violations:
        score *= 0.75 ** len(order_violations)
    if not called:
        details.append("agent made no tool calls at all")
    return DimensionScore(dimension="selection", score=round(score, 4), details=details)

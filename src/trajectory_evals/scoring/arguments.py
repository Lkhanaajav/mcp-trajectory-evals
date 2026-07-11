"""Argument correctness: were the right tools called with the right parameters?

This is where final-output-only evals go blind: an agent can produce a
plausible answer having run `detect_anomalies` with the wrong period, and
nothing downstream will reveal it.
"""

from __future__ import annotations

from ..spec import ArgCheck, Task
from ..trajectory import Trajectory
from .base import DimensionScore


def score_arguments(task: Task, trajectory: Trajectory) -> DimensionScore:
    if not task.arg_checks:
        return DimensionScore(dimension="arguments", score=1.0, details=["no arg checks defined"])
    passed = 0
    details: list[str] = []
    for check in task.arg_checks:
        ok, why = _evaluate(check, trajectory)
        if ok:
            passed += 1
        else:
            details.append(why)
    return DimensionScore(
        dimension="arguments",
        score=round(passed / len(task.arg_checks), 4),
        details=details,
    )


def _evaluate(check: ArgCheck, trajectory: Trajectory) -> tuple[bool, str]:
    call = trajectory.nth_call(check.tool, check.occurrence)
    label = f"{check.tool}#{check.occurrence}.{check.path}"
    if call is None:
        return False, f"{label}: tool was never called (occurrence {check.occurrence} not found)"
    value = _dig(call.arguments, check.path)
    if value is _MISSING:
        return False, f"{label}: argument not provided"

    if check.op == "eq":
        ok = value == check.value
    elif check.op == "approx":
        try:
            ok = abs(float(value) - float(check.value)) <= check.tolerance
        except (TypeError, ValueError):
            ok = False
    elif check.op == "contains":
        ok = str(check.value).lower() in str(value).lower()
    elif check.op == "lte":
        ok = isinstance(value, int | float) and value <= check.value
    else:  # gte
        ok = isinstance(value, int | float) and value >= check.value

    if ok:
        return True, ""
    return False, f"{label}: expected {check.op} {check.value!r}, got {value!r}"


_MISSING = object()


def _dig(obj: object, path: str) -> object:
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return _MISSING
        obj = obj[part]
    return obj

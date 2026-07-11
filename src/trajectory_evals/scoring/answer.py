"""Answer correctness: required substrings and numeric facts in the final answer."""

from __future__ import annotations

import re

from ..spec import Task
from ..trajectory import Trajectory
from .base import DimensionScore

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def score_answer(task: Task, trajectory: Trajectory) -> DimensionScore:
    exp = task.answer
    checks = len(exp.contains) + len(exp.numeric_facts)
    if checks == 0:
        return DimensionScore(dimension="answer", score=1.0, details=["no answer checks defined"])

    answer = trajectory.final_answer
    answer_numbers = [float(tok) for tok in _NUMBER.findall(answer)]
    passed = 0
    details: list[str] = []

    for needle in exp.contains:
        if needle.lower() in answer.lower():
            passed += 1
        else:
            details.append(f"answer missing required phrase: {needle!r}")

    for fact in exp.numeric_facts:
        tol = max(fact.tolerance, 1e-9)
        if any(abs(n - fact.value) <= tol for n in answer_numbers):
            passed += 1
        else:
            label = f" ({fact.label})" if fact.label else ""
            details.append(f"answer missing numeric fact {fact.value}{label} ±{fact.tolerance}")

    return DimensionScore(dimension="answer", score=round(passed / checks, 4), details=details)

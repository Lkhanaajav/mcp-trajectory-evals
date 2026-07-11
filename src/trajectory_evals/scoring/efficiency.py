"""Efficiency: optimal-call ratio plus redundancy detection."""

from __future__ import annotations

import json

from ..spec import Task
from ..trajectory import Trajectory
from .base import DimensionScore


def score_efficiency(task: Task, trajectory: Trajectory) -> DimensionScore:
    n_calls = len(trajectory.tool_calls)
    details: list[str] = []
    if n_calls == 0:
        return DimensionScore(dimension="efficiency", score=0.0, details=["no tool calls"])

    seen: set[str] = set()
    redundant = 0
    for call in trajectory.tool_calls:
        key = call.tool + json.dumps(call.arguments, sort_keys=True)
        if key in seen:
            redundant += 1
            details.append(f"redundant identical call: {call.tool}({call.arguments})")
        seen.add(key)

    ratio = min(1.0, task.optimal_calls / n_calls)
    if n_calls > task.optimal_calls:
        details.append(f"{n_calls} calls vs {task.optimal_calls} optimal")
    score = ratio * (0.8 ** redundant)
    return DimensionScore(dimension="efficiency", score=round(score, 4), details=details)

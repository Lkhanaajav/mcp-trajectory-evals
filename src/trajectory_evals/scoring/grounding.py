"""Grounding: is every number in the final answer traceable to a tool result?

Extracts numeric claims from the answer text and searches the trajectory's
tool results for a matching value (0.5% relative or 0.01 absolute tolerance,
whichever is looser — answers legitimately round). Numbers with no source are
the classic "confident hallucination" failure.

Timestamps are stripped before extraction so '2026-06-04T11:15' doesn't
register as the numbers 2026, 6, 4, 11, and 15; dates are then checked
separately as substrings against the raw results.
"""

from __future__ import annotations

import re

from ..spec import Task
from ..trajectory import Trajectory
from .base import DimensionScore

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_REL_TOL = 0.005
_ABS_TOL = 0.01


def score_grounding(task: Task, trajectory: Trajectory) -> DimensionScore:
    answer = trajectory.final_answer
    if not answer.strip():
        return DimensionScore(
            dimension="grounding", score=0.0, details=["final answer is empty"]
        )

    source_numbers = _collect_numbers(trajectory) + list(task.grounding_allowlist)
    source_text = " ".join(c.result_json for c in trajectory.tool_calls)

    claims: list[tuple[str, bool]] = []

    for ts in _TIMESTAMP.findall(answer):
        claims.append((f"timestamp {ts}", ts[:10] in source_text))
    stripped = _TIMESTAMP.sub(" ", answer)

    for token in _NUMBER.findall(stripped):
        value = float(token)
        claims.append((token, _is_supported(value, source_numbers)))

    if not claims:
        return DimensionScore(
            dimension="grounding", score=1.0, details=["no numeric claims in answer"]
        )
    supported = sum(1 for _, ok in claims if ok)
    details = [f"unsupported claim: {claim}" for claim, ok in claims if not ok]
    return DimensionScore(
        dimension="grounding", score=round(supported / len(claims), 4), details=details
    )


def _is_supported(value: float, sources: list[float]) -> bool:
    tol = max(_ABS_TOL, abs(value) * _REL_TOL)
    return any(abs(value - s) <= tol for s in sources)


def _collect_numbers(trajectory: Trajectory) -> list[float]:
    numbers: list[float] = []
    for call in trajectory.tool_calls:
        _walk(call.result(), numbers)
        _walk(call.arguments, numbers)  # inputs the agent chose are fair to restate
    return numbers


def _walk(obj: object, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, str):
        # Results may embed numbers in strings (error messages, text content).
        out.extend(float(tok) for tok in _NUMBER.findall(_TIMESTAMP.sub(" ", obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)

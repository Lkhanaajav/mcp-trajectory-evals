"""CI regression gate: compare a run against a committed baseline."""

from __future__ import annotations

from .run import RunResult


def gate(current: RunResult, baseline: RunResult, tolerance: float = 0.02) -> tuple[bool, list[str]]:
    """Return (ok, reasons). Fails on must-pass failures, pass-rate drops, or
    mean-score regression beyond `tolerance`."""
    reasons: list[str] = []

    if current.summary.must_pass_failures:
        reasons.append(f"must-pass task(s) failed: {', '.join(current.summary.must_pass_failures)}")

    if current.summary.pass_rate < baseline.summary.pass_rate:
        reasons.append(
            f"pass rate regressed: {baseline.summary.pass_rate:.2%} -> {current.summary.pass_rate:.2%}"
        )

    drop = baseline.summary.mean_overall - current.summary.mean_overall
    if drop > tolerance:
        reasons.append(
            f"mean score regressed by {drop:.4f} (tolerance {tolerance}): "
            f"{baseline.summary.mean_overall:.4f} -> {current.summary.mean_overall:.4f}"
        )

    baseline_by_id = {t.task_id: t for t in baseline.tasks}
    for task in current.tasks:
        prior = baseline_by_id.get(task.task_id)
        if prior and prior.passed and not task.passed:
            reasons.append(f"task newly failing: {task.task_id} ({prior.overall:.3f} -> {task.overall:.3f})")

    return (not reasons, reasons)

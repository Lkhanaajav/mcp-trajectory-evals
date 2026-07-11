"""Human-readable reports from a RunResult."""

from __future__ import annotations

from .run import RunResult

_DIMS = ("selection", "arguments", "grounding", "efficiency", "answer")


def to_markdown(result: RunResult) -> str:
    lines = [
        f"# Eval report — {result.suite}",
        "",
        f"Runner: **{result.runner}**" + (f" · model: **{result.model}**" if result.model else ""),
        "",
        f"**{result.summary.n_passed}/{result.summary.n_tasks} tasks passed** "
        f"(pass rate {result.summary.pass_rate:.0%}, mean score {result.summary.mean_overall:.3f})",
        "",
        "| task | " + " | ".join(_DIMS) + " | overall | pass |",
        "|---|" + "---|" * (len(_DIMS) + 2),
    ]
    for task in result.tasks:
        cells = [f"{task.dimension(d).score:.2f}" for d in _DIMS]
        flag = "✅" if task.passed else "❌"
        lines.append(
            f"| {task.task_id} | " + " | ".join(cells) + f" | **{task.overall:.3f}** | {flag} |"
        )

    problems = [
        (task.task_id, dim.dimension, detail)
        for task in result.tasks
        for dim in task.dimensions
        if dim.score < 1.0
        for detail in dim.details
    ]
    if problems:
        lines += ["", "## Deductions", ""]
        lines += [f"- `{tid}` / {dim}: {detail}" for tid, dim, detail in problems]

    if result.summary.must_pass_failures:
        lines += ["", f"⛔ must-pass failures: {', '.join(result.summary.must_pass_failures)}"]
    return "\n".join(lines)

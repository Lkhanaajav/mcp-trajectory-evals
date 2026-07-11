"""Generate the README scorecard heatmap (light + dark) from real harness runs.

The four rows are actually computed, not illustrative: the honest row is the
scripted reference run over all 8 tasks; each sabotage row takes a real
trajectory, breaks it the way a bad agent would, and rescores it with the
same scorers CI uses.

Run from the repo root (matplotlib is not a package dependency):

    uv run --with matplotlib python scripts/make_readme_chart.py
"""

import asyncio
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

from trajectory_evals.mcp_client import connect
from trajectory_evals.runners.scripted import ScriptedRunner
from trajectory_evals.scoring import score_task
from trajectory_evals.spec import Suite

SUITE_PATH = Path(__file__).parent.parent / "suites" / "timeseries" / "suite.yaml"
OUT = Path(__file__).parent.parent / "docs" / "charts"

DIMS = ("selection", "arguments", "grounding", "efficiency", "answer")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "critical": "#d03b3b",
        "good": "#0ca30c",
        # sequential blue, low -> high
        "ramp": ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#898781",
        "critical": "#e66767",
        "good": "#0ca30c",
        # on a dark surface "near zero" recedes toward the surface
        "ramp": ["#0d366b", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}


async def collect_rows() -> list[tuple[str, list[float], bool]]:
    suite = Suite.from_yaml(str(SUITE_PATH))
    runner = ScriptedRunner()
    tasks = {t.id: t for t in suite.tasks}

    async with connect(suite.server) as session:
        honest = {
            task.id: (task, await runner.run_task(session, task)) for task in suite.tasks
        }
        scores = [score_task(suite, task, traj) for task, traj in honest.values()]

        def mean(dim: str) -> float:
            return sum(s.dimension(dim).score for s in scores) / len(scores)

        rows = [(
            "Honest agent — all 8 tasks (mean)",
            [mean(d) for d in DIMS] + [sum(s.overall for s in scores) / len(scores)],
            all(s.passed for s in scores),
        )]

        # Sabotage 1: perfect tool use, invented numbers in the final answer.
        task, traj = honest["gap-audit"]
        lying = traj.model_copy(deep=True)
        lying.final_answer = (
            "Quality audit complete: 3 gaps found, the largest losing 250 samples, "
            "which is 87.5% of a day."
        )
        s = score_task(suite, task, lying)
        rows.append((
            "Hallucinated numbers — gap-audit",
            [s.dimension(d).score for d in DIMS] + [s.overall], s.passed,
        ))

        # Sabotage 2: skipped the required data_quality call.
        skipping = traj.model_copy(deep=True)
        skipping.tool_calls = [c for c in skipping.tool_calls if c.tool != "data_quality"]
        s = score_task(suite, task, skipping)
        rows.append((
            "Skipped required tool — gap-audit",
            [s.dimension(d).score for d in DIMS] + [s.overall], s.passed,
        ))

        # Sabotage 3: right tool, wrong seasonal period (100 instead of 288).
        original = tasks["seasonal-anomalies"]
        tampered = original.model_copy(deep=True)
        tampered.reference_plan[1].arguments["period"] = 100
        traj = await runner.run_task(session, tampered)
        s = score_task(suite, original, traj)
        rows.append((
            "Wrong period argument — seasonal-anomalies",
            [s.dimension(d).score for d in DIMS] + [s.overall], s.passed,
        ))

    return rows


def render(rows, mode: str) -> None:
    t = THEMES[mode]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    cols = [*DIMS, "overall"]
    n_rows, n_cols = len(rows), len(cols)

    fig, ax = plt.subplots(figsize=(11.6, 3.4), dpi=160, facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    gap = 0.045  # the 2px spacer rule, in cell units

    for i, (_, values, _) in enumerate(rows):
        for j, value in enumerate(values):
            color = cmap(value)
            ax.add_patch(Rectangle(
                (j + gap / 2, n_rows - 1 - i + gap / 2), 1 - gap, 1 - gap,
                facecolor=color, edgecolor="none",
            ))
            r, g, b = to_rgb(color)[:3]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(
                j + 0.5, n_rows - 1 - i + 0.5, f"{value:.2f}",
                ha="center", va="center", fontsize=10,
                color="#ffffff" if lum < 0.45 else "#0b0b0b",
            )

    for i, (label, _, passed) in enumerate(rows):
        ax.text(-0.15, n_rows - 1 - i + 0.5, label, ha="right", va="center",
                fontsize=9.5, color=t["ink"])
        mark, color = ("✓ passes", t["good"]) if passed else ("✗ fails", t["critical"])
        ax.text(n_cols + 0.15, n_rows - 1 - i + 0.5, mark, ha="left", va="center",
                fontsize=9.5, color=color, fontweight="bold")

    for j, col in enumerate(cols):
        ax.text(j + 0.5, n_rows + 0.18, col, ha="center", va="bottom",
                fontsize=9.5, color=t["muted"])

    ax.set_title(
        "Same server, four agents — each failure shows up in exactly the dimension built to catch it",
        loc="left", fontsize=11, color=t["ink"], pad=30, x=-0.42,
    )
    ax.set_xlim(-4.2, n_cols + 1.3)
    ax.set_ylim(-0.1, n_rows + 0.75)
    ax.axis("off")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scorecard_{mode}.png"
    fig.savefig(path, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    rows = asyncio.run(collect_rows())
    for mode in THEMES:
        render(rows, mode)

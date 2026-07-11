"""Suite execution: connect to the server, run every task, score every trajectory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .mcp_client import connect
from .scoring import TaskScore, score_task
from .spec import Suite
from .trajectory import Trajectory, save_trajectories


class RunSummary(BaseModel):
    n_tasks: int
    n_passed: int
    pass_rate: float
    mean_overall: float
    must_pass_failures: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    suite: str
    runner: str
    model: str | None = None
    tasks: list[TaskScore]
    summary: RunSummary

    @classmethod
    def from_json_file(cls, path: str | Path) -> RunResult:
        return cls.model_validate_json(Path(path).read_text())


async def run_suite(
    suite: Suite,
    runner,
    save_trajectories_to: str | Path | None = None,
) -> RunResult:
    scores: list[TaskScore] = []
    trajectories: list[Trajectory] = []
    async with connect(suite.server) as session:
        for task in suite.tasks:
            trajectory = await runner.run_task(session, task)
            trajectories.append(trajectory)
            scores.append(score_task(suite, task, trajectory))

    if save_trajectories_to:
        save_trajectories(trajectories, save_trajectories_to)

    n_passed = sum(1 for s in scores if s.passed)
    model = next((t.model for t in trajectories if t.model), None)
    return RunResult(
        suite=suite.name,
        runner=runner.name,
        model=model,
        tasks=scores,
        summary=RunSummary(
            n_tasks=len(scores),
            n_passed=n_passed,
            pass_rate=round(n_passed / len(scores), 4) if scores else 0.0,
            mean_overall=round(sum(s.overall for s in scores) / len(scores), 4) if scores else 0.0,
            must_pass_failures=[s.task_id for s in scores if s.must_pass and not s.passed],
        ),
    )

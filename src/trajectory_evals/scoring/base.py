"""Score containers and the per-task aggregator."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..spec import Suite, Task
from ..trajectory import Trajectory


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0, le=1)
    details: list[str] = Field(default_factory=list, description="Evidence for every deduction.")


class TaskScore(BaseModel):
    task_id: str
    overall: float
    passed: bool
    must_pass: bool
    dimensions: list[DimensionScore]
    tool_sequence: list[str]
    final_answer: str

    def dimension(self, name: str) -> DimensionScore:
        for d in self.dimensions:
            if d.dimension == name:
                return d
        raise KeyError(name)


def score_task(suite: Suite, task: Task, trajectory: Trajectory) -> TaskScore:
    from .answer import score_answer
    from .arguments import score_arguments
    from .efficiency import score_efficiency
    from .grounding import score_grounding
    from .selection import score_selection

    dims = [
        score_selection(task, trajectory),
        score_arguments(task, trajectory),
        score_grounding(task, trajectory),
        score_efficiency(task, trajectory),
        score_answer(task, trajectory),
    ]
    w = suite.weights
    weight_by_dim = {
        "selection": w.selection,
        "arguments": w.arguments,
        "grounding": w.grounding,
        "efficiency": w.efficiency,
        "answer": w.answer,
    }
    total_weight = sum(weight_by_dim.values())
    overall = sum(d.score * weight_by_dim[d.dimension] for d in dims) / total_weight

    # Required means required: skipping a must_call tool or touching a
    # forbidden one fails the task no matter how the weighted score lands.
    called = set(trajectory.called_tools())
    hard_fail = any(t not in called for t in task.tools.must_call) or any(
        t in called for t in task.tools.must_not_call
    )
    return TaskScore(
        task_id=task.id,
        overall=round(overall, 4),
        passed=overall >= suite.pass_threshold and not hard_fail,
        must_pass=task.must_pass,
        dimensions=dims,
        tool_sequence=trajectory.called_tools(),
        final_answer=trajectory.final_answer,
    )

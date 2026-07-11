"""Replay runner: re-scores previously recorded trajectories.

Useful for (a) scoring runs made elsewhere, (b) iterating on scorers without
re-running agents, (c) committing regression fixtures.
"""

from __future__ import annotations

from pathlib import Path

from ..spec import Task
from ..trajectory import Trajectory, load_trajectories


class ReplayRunner:
    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self._by_task: dict[str, Trajectory] = {
            t.task_id: t for t in load_trajectories(path)
        }

    async def run_task(self, session: object, task: Task) -> Trajectory:
        if task.id not in self._by_task:
            raise KeyError(f"No recorded trajectory for task '{task.id}'.")
        return self._by_task[task.id]

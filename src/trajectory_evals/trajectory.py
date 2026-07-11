"""Trajectory: the full record of one agent attempt — every call, argument, and result."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)
    result_json: str = Field(description="JSON of the structured result, or raw text.")
    is_error: bool = False

    def result(self) -> object:
        try:
            return json.loads(self.result_json)
        except (json.JSONDecodeError, TypeError):
            return self.result_json


class Trajectory(BaseModel):
    task_id: str
    runner: str
    model: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str = ""
    llm_turns: int = 0

    def called_tools(self) -> list[str]:
        return [c.tool for c in self.tool_calls]

    def nth_call(self, tool: str, occurrence: int = 1) -> ToolCall | None:
        seen = 0
        for call in self.tool_calls:
            if call.tool == tool:
                seen += 1
                if seen == occurrence:
                    return call
        return None


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    with open(path, "w") as fh:
        for t in trajectories:
            fh.write(t.model_dump_json() + "\n")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    with open(path) as fh:
        return [Trajectory.model_validate_json(line) for line in fh if line.strip()]

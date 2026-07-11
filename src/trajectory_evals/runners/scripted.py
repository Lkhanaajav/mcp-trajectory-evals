"""Deterministic runner: executes each task's reference plan against the real server.

This is NOT a mock — every tool call hits the actual MCP server over the actual
protocol. What's scripted is only the *decision* of which tools to call, which
makes it the harness's own ground truth: if the reference plan can't score 1.0,
the task spec (or the server) is broken. It also runs in CI with no API key.

Plan arguments and the answer template may reference earlier step results with
{rN.dotted.path[i]} (or {rN} for the whole result) — e.g. pass the series_id
returned by step 0 into step 1 instead of hardcoding server-assigned handles.
"""

from __future__ import annotations

import json
import re

from ..mcp_client import MCPSession
from ..spec import Task
from ..trajectory import ToolCall, Trajectory

_PLACEHOLDER = re.compile(r"\{r(\d+)(?:\.([a-zA-Z0-9_.\[\]]+))?\}")


class ScriptedRunner:
    name = "scripted"

    async def run_task(self, session: MCPSession, task: Task) -> Trajectory:
        if not task.reference_plan:
            raise ValueError(f"Task '{task.id}' has no reference_plan for the scripted runner.")
        calls: list[ToolCall] = []
        for step in task.reference_plan:
            arguments = _render_value(step.arguments, calls)
            result_json, is_error = await session.call(step.tool, arguments)
            if is_error and not step.expect_error:
                raise RuntimeError(
                    f"Reference plan step {step.tool} errored unexpectedly in task "
                    f"'{task.id}': {result_json}"
                )
            if not is_error and step.expect_error:
                raise RuntimeError(
                    f"Reference plan step {step.tool} in task '{task.id}' was expected "
                    "to error but succeeded — the task spec is stale."
                )
            calls.append(
                ToolCall(
                    tool=step.tool, arguments=arguments,
                    result_json=result_json, is_error=is_error,
                )
            )
        answer = _render_text(task.reference_answer, calls)
        return Trajectory(
            task_id=task.id, runner=self.name, tool_calls=calls, final_answer=answer
        )


def _render_value(value: object, calls: list[ToolCall]) -> object:
    """Recursively resolve placeholders inside plan arguments."""
    if isinstance(value, str):
        full = _PLACEHOLDER.fullmatch(value)
        if full:  # whole-value reference keeps the native type (int, dict, ...)
            return _resolve(full, calls)
        return _render_text(value, calls)
    if isinstance(value, dict):
        return {k: _render_value(v, calls) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, calls) for v in value]
    return value


def _render_text(template: str, calls: list[ToolCall]) -> str:
    def replace(match: re.Match) -> str:
        resolved = _resolve(match, calls)
        if isinstance(resolved, float):
            return f"{resolved:g}"
        return str(resolved)

    return _PLACEHOLDER.sub(replace, template)


def _resolve(match: re.Match, calls: list[ToolCall]) -> object:
    step_idx = int(match.group(1))
    path = match.group(2)
    if step_idx >= len(calls):
        raise ValueError(f"Template references step r{step_idx}; only {len(calls)} steps ran.")
    result = calls[step_idx].result()
    return _walk(result, path) if path else result


def _walk(obj: object, path: str) -> object:
    for part in path.replace("]", "").split("."):
        key, _, index = part.partition("[")
        if key:
            if not isinstance(obj, dict) or key not in obj:
                raise KeyError(f"Path '{path}' failed at '{key}' in {json.dumps(obj)[:200]}")
            obj = obj[key]
        if index:
            obj = obj[int(index)]
    return obj

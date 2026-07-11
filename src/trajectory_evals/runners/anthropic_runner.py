"""Live runner: a real Claude agent loop over the MCP server's tools.

Requires ANTHROPIC_API_KEY and the `live` extra (`pip install "mcp-trajectory-evals[live]"`).
Kept to the stable Messages API (no beta surfaces): loop on stop_reason == "tool_use",
execute against the MCP session, return tool_result blocks.
"""

from __future__ import annotations

import json
import os

from ..mcp_client import MCPSession
from ..spec import Task
from ..trajectory import ToolCall, Trajectory

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicRunner:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = "") -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AnthropicRunner needs ANTHROPIC_API_KEY. For keyless runs use --runner scripted."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                'AnthropicRunner needs the live extra: pip install "mcp-trajectory-evals[live]"'
            ) from exc
        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.system_prompt = system_prompt

    async def run_task(self, session: MCPSession, task: Task) -> Trajectory:
        tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in await session.list_tools()
        ]
        messages: list[dict] = [{"role": "user", "content": task.prompt}]
        calls: list[ToolCall] = []
        llm_turns = 0
        final_answer = ""

        while llm_turns <= task.max_steps:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=tools,
                messages=messages,
            )
            llm_turns += 1
            if response.stop_reason != "tool_use":
                final_answer = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                arguments = block.input if isinstance(block.input, dict) else {}
                result_json, is_error = await session.call(block.name, arguments)
                calls.append(
                    ToolCall(
                        tool=block.name, arguments=arguments,
                        result_json=result_json, is_error=is_error,
                    )
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_json if is_error else _clip(result_json),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        return Trajectory(
            task_id=task.id, runner=self.name, model=self.model,
            tool_calls=calls, final_answer=final_answer, llm_turns=llm_turns,
        )


def _clip(result_json: str, limit: int = 20_000) -> str:
    if len(result_json) <= limit:
        return result_json
    return json.dumps({"truncated": True, "prefix": result_json[:limit]})

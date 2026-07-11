"""Thin wrapper that connects to the MCP server under evaluation.

Two modes:
- import: in-process FastMCP instance (fast, deterministic — used in CI)
- stdio:  spawn the real server binary (what production hosts do)
"""

from __future__ import annotations

import importlib
import json
from contextlib import asynccontextmanager

from fastmcp import Client

from .spec import ServerSpec


def _resolve_import(target: str):
    module_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"server.target must look like 'package.module:attr', got '{target}'")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def build_client(server: ServerSpec) -> Client:
    if server.type == "import":
        return Client(_resolve_import(server.target))
    config = {
        "mcpServers": {
            "target": {"command": server.command, "args": server.args, "env": server.env}
        }
    }
    return Client(config)


@asynccontextmanager
async def connect(server: ServerSpec):
    async with build_client(server) as client:
        yield MCPSession(client)


class MCPSession:
    """Uniform call surface used by every runner."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def list_tools(self) -> list:
        return await self._client.list_tools()

    async def call(self, tool: str, arguments: dict) -> tuple[str, bool]:
        """Call a tool; return (result JSON/text, is_error). Never raises on tool errors —
        an agent's error-recovery behavior is part of what gets evaluated."""
        try:
            result = await self._client.call_tool(tool, arguments)
        except Exception as exc:  # ToolError and transport-level validation errors
            return str(exc), True
        if result.structured_content is not None:
            return json.dumps(result.structured_content), False
        text = "".join(getattr(block, "text", "") for block in result.content)
        return text, False

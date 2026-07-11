"""Suite/task specification models — the YAML schema, typed."""

from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class OrderConstraint(BaseModel):
    """Require the first call of `before` to precede the first call of `after`."""

    before: str
    after: str


class ToolExpectations(BaseModel):
    must_call: list[str] = Field(default_factory=list, description="Tools the agent must use.")
    may_call: list[str] = Field(
        default_factory=list, description="Tools that are acceptable but not required."
    )
    must_not_call: list[str] = Field(default_factory=list)
    order: list[OrderConstraint] = Field(default_factory=list)


class ArgCheck(BaseModel):
    """Assert on one argument of one tool call."""

    tool: str
    occurrence: int = Field(default=1, ge=1, description="1 = first call of this tool.")
    path: str = Field(description="Dotted path into the arguments dict, e.g. 'period'.")
    op: Literal["eq", "approx", "contains", "lte", "gte"] = "eq"
    value: Any
    tolerance: float = Field(default=0.0, description="Absolute tolerance for op=approx.")


class NumericFact(BaseModel):
    """A number that must appear in the final answer (within tolerance)."""

    value: float
    tolerance: float = 0.0
    label: str = ""


class AnswerExpectations(BaseModel):
    contains: list[str] = Field(default_factory=list, description="Case-insensitive substrings.")
    numeric_facts: list[NumericFact] = Field(default_factory=list)


class PlanStep(BaseModel):
    """One step of the deterministic reference plan (used by the scripted runner)."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expect_error: bool = False


class Task(BaseModel):
    id: str
    prompt: str
    max_steps: int = Field(default=12, description="Hard cap on tool calls for live agents.")
    optimal_calls: int = Field(ge=1, description="Tool calls a perfect agent needs.")
    must_pass: bool = Field(default=False, description="Gate fails immediately if this task fails.")
    tools: ToolExpectations = Field(default_factory=ToolExpectations)
    arg_checks: list[ArgCheck] = Field(default_factory=list)
    answer: AnswerExpectations = Field(default_factory=AnswerExpectations)
    grounding_allowlist: list[float] = Field(
        default_factory=list,
        description="Numbers that are methodology, not measurement (e.g. 95 in '95% interval') "
        "and don't need a tool-result source.",
    )
    reference_plan: list[PlanStep] = Field(default_factory=list)
    reference_answer: str = Field(
        default="",
        description="Scripted runner's answer template; {rN.path} pulls from step N's result.",
    )


class ServerSpec(BaseModel):
    """How to reach the MCP server under evaluation."""

    type: Literal["import", "stdio"] = "import"
    target: str | None = Field(
        default=None, description="import: 'package.module:attr' of a FastMCP instance."
    )
    command: str | None = Field(default=None, description="stdio: executable to spawn.")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ServerSpec:
        if self.type == "import" and not self.target:
            raise ValueError("server.type=import requires server.target")
        if self.type == "stdio" and not self.command:
            raise ValueError("server.type=stdio requires server.command")
        return self


class Weights(BaseModel):
    selection: float = 0.30
    arguments: float = 0.20
    grounding: float = 0.20
    efficiency: float = 0.10
    answer: float = 0.20


class Suite(BaseModel):
    name: str
    description: str = ""
    server: ServerSpec
    system_prompt: str = (
        "You are a careful data analyst. Use the available tools for every factual claim; "
        "never invent numbers. Finish with a concise answer."
    )
    weights: Weights = Field(default_factory=Weights)
    pass_threshold: float = Field(default=0.7, ge=0, le=1)
    tasks: list[Task]

    @classmethod
    def from_yaml(cls, path: str) -> Suite:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

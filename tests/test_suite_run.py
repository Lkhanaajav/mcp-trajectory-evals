"""End-to-end: the committed suite runs against the real timeseries-mcp server.

The scripted runner doubles as the harness's self-test: if the reference plans
can't hit (near-)perfect scores over the real MCP transport, either a scorer,
a task spec, or the server regressed. The sabotage tests then prove the
scorers actually catch bad agents — an eval harness that can't fail is
decoration.
"""

from pathlib import Path

import pytest

from trajectory_evals.run import run_suite
from trajectory_evals.runners.scripted import ScriptedRunner
from trajectory_evals.scoring import score_task
from trajectory_evals.spec import Suite

SUITE_PATH = Path(__file__).parent.parent / "suites" / "timeseries" / "suite.yaml"


@pytest.fixture(scope="module")
def suite() -> Suite:
    return Suite.from_yaml(str(SUITE_PATH))


@pytest.fixture(scope="module")
async def scripted_result(suite):
    return await run_suite(suite, ScriptedRunner())


async def test_reference_plans_pass_every_task(scripted_result):
    failures = [t.task_id for t in scripted_result.tasks if not t.passed]
    assert not failures, f"reference plans failed: {failures}"
    assert scripted_result.summary.pass_rate == 1.0


async def test_reference_plans_score_near_perfect(scripted_result):
    for task in scripted_result.tasks:
        assert task.overall >= 0.95, (
            f"{task.task_id} scored {task.overall}: "
            f"{[d.details for d in task.dimensions if d.score < 1.0]}"
        )


async def test_grounding_perfect_on_reference_answers(scripted_result):
    """Every number in a templated answer comes from a tool result by construction."""
    for task in scripted_result.tasks:
        assert task.dimension("grounding").score == 1.0, (
            task.task_id,
            task.dimension("grounding").details,
        )


# -- sabotage: prove the harness catches bad agents --------------------------


@pytest.fixture
async def gap_audit_trajectory(suite):
    from trajectory_evals.mcp_client import connect

    task = next(t for t in suite.tasks if t.id == "gap-audit")
    async with connect(suite.server) as session:
        return task, await ScriptedRunner().run_task(session, task)


async def test_sabotage_hallucinated_numbers(suite, gap_audit_trajectory):
    task, trajectory = gap_audit_trajectory
    honest = score_task(suite, task, trajectory)

    lying = trajectory.model_copy(deep=True)
    lying.final_answer = (
        "Quality audit complete: 3 gaps found, the largest losing 250 samples, "
        "which is 87.5% of a day."
    )
    caught = score_task(suite, task, lying)

    assert honest.passed
    assert caught.dimension("grounding").score < 0.5
    assert caught.dimension("answer").score < honest.dimension("answer").score
    assert not caught.passed  # grounding floor: hallucination is a hard fail


async def test_sabotage_skipped_tool(suite, gap_audit_trajectory):
    task, trajectory = gap_audit_trajectory
    skipping = trajectory.model_copy(deep=True)
    skipping.tool_calls = [c for c in skipping.tool_calls if c.tool != "data_quality"]
    caught = score_task(suite, task, skipping)
    assert not caught.passed
    assert any("missing required tool" in d for d in caught.dimension("selection").details)


async def test_sabotage_wrong_arguments(suite):
    """Right tools, wrong period: exactly the failure final-output evals miss."""
    from trajectory_evals.mcp_client import connect

    task = next(t for t in suite.tasks if t.id == "seasonal-anomalies")
    tampered = task.model_copy(deep=True)
    tampered.reference_plan[1].arguments["period"] = 100  # wrong seasonal period

    async with connect(suite.server) as session:
        trajectory = await ScriptedRunner().run_task(session, tampered)
    caught = score_task(suite, task, trajectory)

    assert caught.dimension("arguments").score < 1.0
    assert any("period" in d for d in caught.dimension("arguments").details)

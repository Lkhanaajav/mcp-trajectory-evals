"""Gate logic and the CLI surface, end to end on the real suite."""

from pathlib import Path

import pytest

from trajectory_evals.cli import main
from trajectory_evals.gate import gate
from trajectory_evals.run import RunResult

SUITE = str(Path(__file__).parent.parent / "suites" / "timeseries" / "suite.yaml")


@pytest.fixture(scope="module")
def run_json(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("runs") / "run.json"
    exit_code = main(["run", SUITE, "--runner", "scripted", "--out", str(out)])
    assert exit_code == 0
    return out


def test_cli_run_writes_valid_result(run_json):
    result = RunResult.from_json_file(run_json)
    assert result.summary.n_tasks == 8
    assert result.summary.pass_rate == 1.0


def test_cli_report(run_json, capsys):
    assert main(["report", str(run_json)]) == 0
    out = capsys.readouterr().out
    assert "gap-audit" in out
    assert "8/8 tasks passed" in out


def test_cli_gate_passes_against_itself(run_json):
    assert main(["gate", str(run_json), "--baseline", str(run_json)]) == 0


def test_gate_fails_on_regression(run_json):
    current = RunResult.from_json_file(run_json)
    baseline = RunResult.from_json_file(run_json)

    # Simulate a regression: one previously passing task now fails badly.
    current.tasks[0].passed = False
    current.tasks[0].overall = 0.3
    current.summary.pass_rate = 7 / 8
    current.summary.mean_overall -= 0.1
    current.summary.must_pass_failures = [current.tasks[0].task_id]

    ok, reasons = gate(current, baseline)
    assert not ok
    assert any("must-pass" in r for r in reasons)
    assert any("newly failing" in r for r in reasons)


def test_gate_tolerates_small_drift(run_json):
    current = RunResult.from_json_file(run_json)
    baseline = RunResult.from_json_file(run_json)
    current.summary.mean_overall -= 0.01  # within default 0.02 tolerance
    ok, reasons = gate(current, baseline)
    assert ok, reasons

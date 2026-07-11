"""trajeval CLI: run suites, render reports, gate CI."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .gate import gate
from .report import to_markdown
from .run import RunResult, run_suite
from .spec import Suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trajeval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a suite and write a JSON result.")
    p_run.add_argument("suite", help="Path to suite YAML.")
    p_run.add_argument("--runner", choices=["scripted", "anthropic", "replay"], default="scripted")
    p_run.add_argument("--model", default=None, help="Model id for --runner anthropic.")
    p_run.add_argument("--replay-from", default=None, help="Trajectory JSONL for --runner replay.")
    p_run.add_argument("--out", default="run.json", help="Where to write the RunResult JSON.")
    p_run.add_argument("--save-trajectories", default=None, help="Also record trajectories (JSONL).")

    p_report = sub.add_parser("report", help="Render a RunResult as Markdown.")
    p_report.add_argument("run_json")

    p_gate = sub.add_parser("gate", help="Compare a run against a baseline; exit 1 on regression.")
    p_gate.add_argument("run_json")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--tolerance", type=float, default=0.02)

    args = parser.parse_args(argv)

    # Tasks with expect_error steps make the in-process server log tracebacks
    # for failures the harness deliberately provokes; keep run output readable.
    logging.getLogger("FastMCP").setLevel(logging.CRITICAL)

    if args.command == "run":
        suite = Suite.from_yaml(args.suite)
        runner = _build_runner(args, suite)
        result = asyncio.run(run_suite(suite, runner, save_trajectories_to=args.save_trajectories))
        Path(args.out).write_text(result.model_dump_json(indent=2) + "\n")
        print(to_markdown(result))
        print(f"\nwrote {args.out}")
        return 0

    if args.command == "report":
        print(to_markdown(RunResult.from_json_file(args.run_json)))
        return 0

    # gate
    ok, reasons = gate(
        RunResult.from_json_file(args.run_json),
        RunResult.from_json_file(args.baseline),
        tolerance=args.tolerance,
    )
    if ok:
        print("gate: OK — no regressions against baseline")
        return 0
    print("gate: FAIL")
    for reason in reasons:
        print(f"  - {reason}")
    return 1


def _build_runner(args, suite: Suite):
    if args.runner == "scripted":
        from .runners.scripted import ScriptedRunner

        return ScriptedRunner()
    if args.runner == "replay":
        if not args.replay_from:
            sys.exit("--runner replay requires --replay-from <trajectories.jsonl>")
        from .runners.replay import ReplayRunner

        return ReplayRunner(args.replay_from)
    from .runners.anthropic_runner import DEFAULT_MODEL, AnthropicRunner

    return AnthropicRunner(model=args.model or DEFAULT_MODEL, system_prompt=suite.system_prompt)


if __name__ == "__main__":
    sys.exit(main())

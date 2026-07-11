# mcp-trajectory-evals

**Trajectory-level evaluation for tool-using agents.** Drives an agent against a *real* MCP server and scores every step of the run — which tools it picked, what arguments it passed, whether its final answer is grounded in tool results, and how efficiently it got there. Not just "did the answer look right."

[![CI](https://github.com/Lkhanaajav/mcp-trajectory-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Lkhanaajav/mcp-trajectory-evals/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Why step-level scoring

Final-output evals have a documented blind spot: an agent can run the wrong tool with the wrong parameters, invent a number, and still produce an answer that *reads* correct. Practitioner reports put the gap at 20–40% of cases passing output-only checks while the trajectory underneath is broken. The failure modes this harness scores directly:

| Dimension | The failure it catches |
|---|---|
| **selection** | Skipped a required tool, called a forbidden one, ran analysis before preprocessing |
| **arguments** | `detect_anomalies` with the wrong seasonal period — output plausible, analysis meaningless |
| **grounding** | Numbers in the final answer that appear in **no tool result** (confident hallucination) |
| **efficiency** | Redundant identical calls, 9 calls where 3 suffice |
| **answer** | Required facts and phrases missing from the final response |

A task **hard-fails** if a `must_call` tool was skipped or a `must_not_call` tool was touched — no weighted average can launder that away.

## What a run looks like

Real output of `trajeval run suites/timeseries/suite.yaml` (the committed CI baseline, scripted runner):

| task | selection | arguments | grounding | efficiency | answer | overall | pass |
|---|---|---|---|---|---|---|---|
| gap-audit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| seasonal-anomalies | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| deploy-changepoint | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| methane-trend | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| forecast-backtest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| overlap-error | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| window-inspection | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |
| stationarity-check | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | ✅ |

Every deduction comes with evidence, e.g. from the harness's own sabotage tests:

```
- gap-audit / grounding: unsupported claim: 250
- gap-audit / grounding: unsupported claim: 87.5
- gap-audit / selection: missing required tool: data_quality
- seasonal-anomalies / arguments: detect_anomalies#1.period: expected approx 288, got 100
```

## How it works

```
suite.yaml ── tasks, expectations, reference plans
    │
    ▼
runner ──────────────── scripted │ anthropic │ replay
    │  every tool call hits the real MCP server (in-process or spawned stdio)
    ▼
Trajectory ── every call, argument, result, error, final answer (JSONL-serializable)
    │
    ▼
scorers ── selection · arguments · grounding · efficiency · answer
    │
    ▼
RunResult ── JSON + Markdown report ──> trajeval gate (CI exit code)
```

Three runners, one contract:

- **`scripted`** — executes each task's committed reference plan against the live server. Zero API keys, fully deterministic, runs in CI. It doubles as the harness's ground truth: if a reference plan can't score 1.0, a scorer, a task spec, or the server regressed.
- **`anthropic`** — a real Claude agent loop (stable Messages API) over the server's advertised tools. `trajeval run suite.yaml --runner anthropic --model claude-sonnet-5` with `ANTHROPIC_API_KEY` set.
- **`replay`** — re-scores recorded trajectory JSONL; iterate on scorers without re-running agents.

The demo suite evaluates [timeseries-mcp](https://github.com/Lkhanaajav/timeseries-mcp): 8 tasks over its seeded sample datasets, so every expected value in the YAML is reproducible ground truth (a 24-point telemetry gap, a +25% CPU deploy plateau, a 28.88°C seasonal-context anomaly), not a guess.

## The harness must be able to fail

An eval harness that can't catch a bad agent is decoration, so the test suite proves each scorer catches deliberate sabotage of otherwise-perfect trajectories:

- **hallucinated numbers** in the final answer → grounding drops below 0.5, task fails
- **skipped required tool** → hard fail regardless of weighted score
- **wrong `period` argument** (100 instead of 288) → arguments scorer names the exact call and expectation

CI runs the full suite against the real server on every push and `trajeval gate` fails the build on any regression vs. the committed baseline — pass-rate drop, mean-score drop beyond tolerance, or any newly-failing task.

## Install & use

```bash
git clone https://github.com/Lkhanaajav/mcp-trajectory-evals && cd mcp-trajectory-evals
uv sync --extra dev                       # dev extra pulls in timeseries-mcp as the demo target

uv run trajeval run suites/timeseries/suite.yaml --out run.json
uv run trajeval report run.json
uv run trajeval gate run.json --baseline baselines/scripted.json

# Live model evaluation (needs ANTHROPIC_API_KEY + the live extra):
uv run --extra live trajeval run suites/timeseries/suite.yaml \
    --runner anthropic --model claude-sonnet-5 --save-trajectories claude-run.jsonl
```

Point it at your own server by writing a suite YAML: `server.type: import` for an in-process FastMCP instance, or `server.type: stdio` with a `command` to spawn any MCP server in any language. Task expectations are plain data — see `suites/timeseries/suite.yaml`, which is deliberately over-commented.

## Design notes

- **Grounding** strips ISO timestamps before number extraction (so `2026-06-04T11:15` isn't "the numbers 2026, 6, 4, 11, 15") and checks dates as substrings against raw results; tolerances allow legitimate rounding (0.5% relative / 0.01 absolute). Methodology constants like the 95 in "95% interval" go in a per-task `grounding_allowlist` rather than a scorer special-case.
- **Reference plans resolve handles at runtime** — `series_id: "{r0.series_id}"` pipes step 0's result into step 1's arguments, so plans don't hardcode server-assigned ids.
- **Expected errors are first-class.** The `overlap-error` task *requires* the agent to attempt a comparison the server must reject, then explain the failure. Error-recovery behavior is part of agent quality; most harnesses can't express it.

## Honest limitations

- Grounding is numeric-and-date only; it won't catch a hallucinated *qualitative* claim ("the sensor was replaced"). An optional LLM-judge dimension is the natural extension (see prior art below for who does this well).
- The scripted runner validates the harness and the server contract, not model intelligence — live-model comparison matrices are exactly what the `anthropic` runner + `--save-trajectories` + `replay` are for; run them with your own key.
- One suite ships today. The spec layer is server-agnostic, but I've only proven it against one server.

## Prior art

[lastmile-ai/mcp-eval](https://github.com/lastmile-ai/mcp-eval) (general MCP eval framework, LLM-judge rubrics), [DeepEval's MCP metrics](https://deepeval.com/docs/evaluation-mcp) (scores traces you captured), [langchain-ai/agentevals](https://github.com/langchain-ai/agentevals) (trajectory match over OpenAI-format messages), and academic benchmarks ([MCP-Universe](https://mcp-universe.github.io/), [LiveMCPBench](https://arxiv.org/abs/2508.01780)). This project's angle: a *server-paired*, fully deterministic suite with grounded-claim checking and a CI gate that needs no API key — the eval layer a server ships with, rather than a platform you adopt.

## Development notes

Built with AI assistance (Claude Code) for scaffolding and test generation; the scoring model, task design, and failure-mode analysis are mine. Decisions I'd defend in review: hard-fail semantics for required tools (a weighted average must not launder a skipped audit step), timestamp-aware number extraction in the grounding scorer, and making the reference plans hit the real server so the harness self-tests its own task specs.

MIT © Lkhanaajav Mijiddorj

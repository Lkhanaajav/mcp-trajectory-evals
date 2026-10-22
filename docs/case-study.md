# Evaluating tool-using agents before release

Lkhanaajav Mijiddorj | Python, MCP, Pydantic, YAML, CI | Verified October 22, 2026

## Problem

A plausible final answer can hide an incorrect tool call, a skipped audit, or an unsupported number. I built mcp-trajectory-evals to inspect the sequence of tool calls as well as the final answer, paired with my typed timeseries-mcp server.

## What I built

The harness runs tasks against a real MCP server, records calls and results, and scores tool selection, arguments, numeric/date grounding, efficiency, and required answer content. YAML describes task expectations and reference plans. JSON/Markdown reports identify deductions, and a CI gate compares a run with a committed baseline. The tool server performs statistical analysis through typed functions with schema-validated, bounded outputs.

## Verification captured in this session

| Scenario | Overall task score | Task passes | Baseline gate |
|---|---:|---|---|
| Eight reference tasks | 1.000 each | 8 of 8 | Baseline |
| Deliberately unsupported numbers | 0.700 | No | Blocks |
| Required data_quality call removed | 0.780 | No | Blocks |
| Seasonal period changed from 288 to 100 | 0.900 | Yes | Allows |

The repository's existing tests also passed: 27 tests. These are deterministic reference-plan runs against the real MCP server and intentionally corrupted trajectories. No live LLM was evaluated, and the scores do not measure production model accuracy.

## Engineering judgment

Required-tool and grounding failures have hard-fail semantics, so a high weighted score cannot hide them. Argument errors are currently deductions: this run flagged the wrong seasonal period but still passed the task and default gate. A deployment that requires strict argument correctness should strengthen that policy. This package documents the behavior without changing it.

Numeric/date grounding does not establish qualitative factual correctness or causality. The shipped suite covers one tool server. Broader suites and live-model runs are needed before generalizing reliability claims.

## Reproduce and inspect

Source: https://github.com/Lkhanaajav/mcp-trajectory-evals

Verified commit: de31ede4cf0c29679b9b2ed6a56d3ad476fbbbe1

Run the reference suite, render its report, and compare against the committed baseline using the repository's documented trajeval commands. Supporting evidence: [captured scores and gate decisions](case-study-results.json), [architecture diagram](case-study-architecture.svg), and [walkthrough script](case-study-demo.md). AI assistance was used in the original project for scaffolding and test generation, as disclosed in its README.
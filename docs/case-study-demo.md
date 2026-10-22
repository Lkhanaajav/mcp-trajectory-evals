# Two minute walkthrough

0:00-0:20 - State the problem: plausible answers can hide broken tool use. Open replay.html.

0:20-0:40 - Show the eight-task reference baseline. Explain that this runs reference plans against a real MCP server; it is not a live-model benchmark.

0:40-1:05 - Show invented numeric claims: grounding drops to zero and the gate blocks the changed run. Show the specific unsupported claims in captured_results.json.

1:05-1:25 - Remove the required data_quality call. Explain why a missing required tool is a hard failure even when other dimensions score well.

1:25-1:45 - Show the wrong seasonal period. The argument scorer detects it, but the weighted task score and default gate still allow it. Explain how a stricter policy could address this.

1:45-2:00 - Finish with the architecture and scope: reproducible failures, inspectable deductions, and an honest distinction between reference validation and live-model evaluation.

Be ready to explain timestamp handling, numeric rounding tolerance, hard-fail rules, and why qualitative claims require a different evaluation approach.
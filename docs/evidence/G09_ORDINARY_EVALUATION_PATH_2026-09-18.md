# Evaluation requests use ordinary reasoning

The cognitive engine returned non-safety, prompt-shaped answers before its
phase pipeline whenever the caller was a test, proof, evaluation or benchmark.
A routing request could receive the fixed Node-A/Node-E plan even when it
named different nodes. This shortcut could neither establish broad reasoning
nor test the path a person uses.

The engine now keeps only the existing governance refusal at that boundary.
Planning, introspection, arithmetic and factual requests continue to ordinary
reasoning for every caller origin. The legacy helper remains available for
historical component tests; it no longer supplies these engine answers.
No prompt instructions were added, no safety authority was removed, and no
measurement result was relabeled. Refusal receipts still report that the
phase pipeline did not run.

Verification: 79 focused tests pass, including the actual `think` entry point
with a recording pipeline backend. Another 580 engine, conversation, causal
influence and durable-turn tests pass. Smoke passes 164 with one skip.
These are routing and regression checks, not a live run or broad gain.

# R10 semantic executable examples

Closed 2026-09-08 from `origin/main` commit `c337800c4`.

The executable-example contract requires more than a successful process exit.
The code truth engine compiles candidate blocks, executes module-level
assertions in the symbolic sandbox, runs doctest examples through an explicit
`DocTestFinder` harness, and demotes executable claims when the sandbox cannot
provide trustworthy evidence. The answer egress path separately executes and
grounds labelled Python output claims against observed stdout.

## Receipt

Command:

```text
AURA_LOG_DIR=/tmp/aura-r10-logs AURA_STATE_ROOT=/tmp/aura-r10-state
/Users/bryan/.aura/live-source/.venv/bin/python -m pytest -q
tests/test_code_answers_are_actually_run.py
tests/test_reasoning_verifiers.py
tests/test_executable_output_grounding.py
tests/test_semantic_program_execution.py
tests/test_semantic_program_floor.py
tests/test_executable_reasoning.py
```

Result: **97 passed in 10.03s**.

The receipt covers positive and negative cases:

- a doctest that matches its implementation passes;
- a doctest that contradicts its implementation fails;
- a module-level assertion that fails in the sandbox fails the verifier even
  when the source is syntactically valid;
- an assertion-bearing block whose sandbox execution is refused is reported as
  unchecked rather than semantically verified;
- a plain function without executable claims keeps its static verdict;
- labelled Python output is compared with captured sandbox stdout, with
  normalization limited to terminal formatting;
- an output claim is removed when execution cannot establish it, while the
  surrounding answer remains;
- semantic programs execute their declared intermediate operations and reject
  invalid inputs and malformed programs.

The evidence is offline acceptance evidence for R10. It does not close live
delivery (R09), latency (R11), or the broader reasoning claims.

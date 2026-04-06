# Quality Gates

## Pre-commit checks (run automatically)

Every commit must pass these before merging:

1. **Syntax**: All Python files parse without SyntaxError
2. **Tests**: `pytest tests/ -q` — all tests pass
3. **No hardcoded paths**: No `/Users/bryan` in tracked files
4. **No model artifacts**: No `.safetensors`, `.gguf`, or files > 1MB in git
5. **No log files**: No `.log` files tracked
6. **Imports resolve**: Core modules import without error

## Per-feature checks

Before any new feature is considered complete:

1. **Unit tests exist** covering the happy path and at least one error case
2. **Integration test** showing it works with the live kernel (if applicable)
3. **No new stubs**: No `pass`-only functions, no `"not implemented"` returns
4. **Personality preserved**: Run 3-turn conversation, verify no assistant-speak
5. **Context window**: System prompt stays under 20K chars with the new feature active

## Response quality gates (automated in benchmarks/)

1. **Generic marker count** = 0 (no "How can I help?", "Certainly!", etc.)
2. **Hedging marker count** = 0 (no "it depends", "both are great")
3. **First-person usage** > 0 per response (personality present)
4. **No raw metrics** in response text (no "valence=", "arousal=", "phi=")
5. **Memory recall**: Can recall topic from 5 turns ago

## Long-horizon checks (benchmarks/long_horizon_stress.py)

1. **Drift detection**: Late-conversation generic markers ≤ 1.5x early markers
2. **Identity persistence**: First-person usage doesn't drop > 50% in second half
3. **Substrate coherence**: Mood, energy, coherence don't go to NaN or extreme values
4. **Memory**: Can recall a topic from turn 5 when asked at turn 25

## Deployment readiness

1. **INSTALL.md matches README** (same Python version, same launch command)
2. **No author-specific paths** in any tracked file
3. **No daemon plists** with RunAtLoad/KeepAlive
4. **All dependencies in requirements.txt**
5. **Fresh venv install succeeds**: `pip install -r requirements.txt` on clean machine

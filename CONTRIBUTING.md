# Contributing to Aura

## Quick Start

```bash
# Clone and install
git clone https://github.com/youngbryan97/aura.git
cd aura
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -x -q -k "not slow and not integration"

# Lint
ruff check .
ruff format --check .

# Type check (strict on public APIs)
mypy core/will.py core/constitution.py core/executive/ --ignore-missing-imports
```

## Architecture Rules

1. **One authority**: All consequential actions route through `UnifiedWill.decide()` (see `core/will.py`). Never add a parallel gate.

2. **One owner per concern**: See `OWNERSHIP.md` for the canonical ownership map. If you need a new governance check, add it as an advisor to the existing owner.

3. **No monkey-patching**: Use event bus hooks, provider registries, or typed extension points instead of `setattr` on live objects.

4. **Immutable messages**: Inter-subsystem communication should use frozen dataclasses from `core/runtime/immutable_messages.py`.

5. **Service states**: All subsystems should track lifecycle via `core/runtime/service_state.py:ServiceState`.

## Adding a New Consciousness Module

1. Create your module in `core/consciousness/your_module.py`
2. Register it in `core/container.py` during boot
3. Add it to the consciousness bridge tick cycle if it needs periodic updates
4. Write at least one ablation test proving what breaks when it's removed
5. Add it to `OWNERSHIP.md` under the appropriate domain
6. Add it to `core/consciousness/theory_arbitration.py` if it makes falsifiable predictions

## Test Categories

| Marker | Purpose | When to run |
|--------|---------|-------------|
| (default) | Unit + fast integration | Every commit |
| `@pytest.mark.slow` | Long-running | Nightly CI |
| `@pytest.mark.integration` | Full pipeline | Before merge |
| `@pytest.mark.stress` | Load/fault injection | Weekly |

## Commit Style

```
<type>: <short description>

<body explaining why, not what>

Co-Authored-By: <name> <email>
```

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `perf`, `ci`

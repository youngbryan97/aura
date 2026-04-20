# Contributing

## Quick start

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
pip install -e ".[dev]"

# Fast tests only
python -m pytest tests/ -x -q -k "not slow and not integration"

# Lint and format
ruff check .
ruff format --check .

# Type-check the parts where we care most
mypy core/will.py core/constitution.py core/executive/ --ignore-missing-imports
```

## Architecture rules

1. **One authority.** Every consequential action routes through
   `UnifiedWill.decide()` in `core/will.py`. Don't add a parallel gate;
   add an advisor to the Will.
2. **One owner per concern.** See [OWNERSHIP.md](OWNERSHIP.md) for the
   canonical map. If you want a new governance check, attach it to the
   existing owner.
3. **No monkey-patching.** Use event-bus hooks, provider registries, or
   typed extension points — not `setattr` on live objects.
4. **Immutable messages.** Inter-subsystem communication uses the frozen
   dataclasses in `core/runtime/immutable_messages.py`.
5. **Lifecycle tracking.** Subsystems report state via
   `core/runtime/service_state.py:ServiceState`.

## Adding a consciousness module

1. Drop the module in `core/consciousness/your_module.py`.
2. Register it in `core/container.py` during boot.
3. If it needs periodic updates, wire it into the consciousness bridge
   tick cycle.
4. Write at least one ablation test that shows what breaks when the
   module is removed.
5. Add an entry to [OWNERSHIP.md](OWNERSHIP.md) under the right domain.
6. If it makes falsifiable predictions, register it in
   `core/consciousness/theory_arbitration.py`.

## Test markers

| Marker | Meaning | When to run |
|--------|---------|-------------|
| (default) | Unit + fast integration | Every commit |
| `@pytest.mark.slow` | Long-running | Nightly CI |
| `@pytest.mark.integration` | Full pipeline | Before merge |
| `@pytest.mark.stress` | Load / fault injection | Weekly |

## Commits

```
<type>: <short description>

<body that explains why, not what>

Co-Authored-By: <name> <email>
```

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `perf`, `ci`.

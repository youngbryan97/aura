# Aura Flagship Readiness Gate

A conservative release gate. Conservative meaning it only flags things that
are wrong in every case, so a hit is a bug rather than a discussion:

```bash
python -m core.runtime.flagship_readiness --strict .
```

Each pattern below is here because it caused a real incident in this
runtime, not because a style guide dislikes it:

- raw production `asyncio.create_task` / `asyncio.ensure_future`
- direct `Path.write_text` calls that may bypass durable persistence policy
- `sys.exit(...)` inside async functions
- import-time `asyncio.Lock/Event/Semaphore/Queue`
- missing morphogenesis boot wiring
- missing global asyncio task supervision patch
- missing direct morphogenesis lifecycle counters

This is a fast red/green gate for the kinds of issues that most often stop a large local AI runtime from feeling flagship-grade. It does not prove the runtime is correct.

## Run

```bash
python -m core.runtime.flagship_readiness --strict .
```

Exit non-zero (in `--strict`) on any finding, so it drops straight into a CI
gate. The historical one-shot "closure patch" bootstrap that originally
installed this gate has been merged into the tree and removed.

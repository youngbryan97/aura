# Aura Flagship Readiness Gate

This patch adds a conservative release gate:

```bash
python -m core.runtime.flagship_readiness --strict .
```

It scans for high-leverage runtime hygiene problems:

- raw production `asyncio.create_task` / `asyncio.ensure_future`
- direct `Path.write_text` calls that may bypass durable persistence policy
- `sys.exit(...)` inside async functions
- import-time `asyncio.Lock/Event/Semaphore/Queue`
- missing morphogenesis boot wiring
- missing global asyncio task supervision patch
- missing direct morphogenesis lifecycle counters

This is not a complete proof that Aura is perfect. It is a fast red/green gate for the kinds of issues that most often prevent a large local AI runtime from feeling flagship-grade.

## Apply

```bash
python scripts/aura_apply_closure_patch_03.py /path/to/aura
cd /path/to/aura
python -m pytest tests/test_closure_patch_03.py -q
python -m core.runtime.flagship_readiness --strict .
```

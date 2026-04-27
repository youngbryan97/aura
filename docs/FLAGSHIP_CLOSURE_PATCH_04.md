# Aura Closure Patch 04

Patch 04 tightens the gaps left after patch 03:

1. Morphogenesis organ-formation episodic logging becomes task-owned via `fire_and_forget`.
2. Morphogenesis registry status exposes direct lifecycle counters:
   `active`, `dormant`, `hibernating`, `quarantined`, `apoptotic`, and `dead`.
3. `TerminalMonitor` blacklist persistence uses `atomic_write_json` when available.
4. A conservative task ownership scanner/codemod is installed at:
   `scripts/aura_task_ownership_codemod.py`.

## Apply

```bash
python scripts/aura_apply_closure_patch_04.py /path/to/aura
cd /path/to/aura
python -m pytest tests/test_closure_patch_04.py -q
python scripts/aura_task_ownership_codemod.py . --json
python -m core.runtime.flagship_readiness --strict .
```

## Notes

The codemod defaults to report-only. `--apply` only rewrites the known-safe
morphogenesis organ-episode `ensure_future` pattern. Everything else is reported
as a manual ownership migration because blind task rewrites in a large runtime
can introduce subtle lifecycle bugs.

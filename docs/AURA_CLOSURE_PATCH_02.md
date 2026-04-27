# Aura Closure Patch 02

This patch targets additional hardening gaps found after the morphogenesis pass.

## Fixes

1. **Asyncio task supervision:** installs a re-entrancy guarded `asyncio.create_task` patch so historical raw task creation routes through `TaskTracker` when available.
2. **Entrypoint version coherence:** enforces the Python 3.12 contract in `aura_main.py`.
3. **Known raw task paths:** rewrites the memory-monitor and morphogenesis organ-episode task paths to direct `TaskTracker` ownership.
4. **Morphogenesis health counters:** exposes direct `active/dormant/hibernating/quarantined/apoptotic/dead` counters in registry status and makes the health hook compatible.
5. **TerminalMonitor durability:** persists the error blacklist via `atomic_write_json` when available, with temp-file fallback.
6. **Container runtime coherence:** aligns `docker/Dockerfile` with Python 3.12 and marks docker-compose runtime as internal-only.

## Install

```bash
unzip aura_closure_patch_02.zip -d aura_closure_patch_02
cd aura_closure_patch_02
python scripts/aura_apply_closure_patch_02.py /path/to/aura
cd /path/to/aura
python -m pytest tests/test_aura_closure_patch_02.py -q
make quality
```

This patch is targeted; it closes concrete runtime, Docker, morphogenesis, and persistence gaps and adds regression tests.

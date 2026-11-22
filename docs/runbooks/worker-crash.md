# Runbook: Worker process crash during inference (F02)

**Fault:** F02 — the MLX inference worker process dies mid-generation
(GPU memory pressure, MLX runtime error, corrupted prompt).

## Symptoms

- Chat turn returns an error or an empty reply; subsequent turns recover.
- `~/.aura/logs/desktop-launch.log` shows the worker respawn banner.
- `data/error_logs/crash/` gains a faulthandler dump if the parent noticed.
- Degradation records for `brain.mlx_client` / `brain.mlx_worker` at
  `critical`.

## Automated mitigation

`core/brain/llm/mlx_client.py` health-checks the worker and respawns it;
the in-flight request fails, the next request goes to the fresh worker.
Target MTTR: 30s.

## Manual diagnosis

1. Confirm the respawn happened: look for consecutive worker PIDs in the
   launch log. If the worker is respawn-looping, STOP — the model or
   manifest is bad, not the request.
2. Check memory pressure at crash time: `data/error_logs/memory/`
   (sentinel ring + tombstones) and macOS `log show --last 10m
   --predicate 'eventMessage CONTAINS "aura"'` for jetsam kills.
3. Check the active model manifest (`training/fused-model/active.json` in
   the checkout her app runs from) points at a model that exists on disk and
   fits in RAM alongside the ~20GB working set.
4. If turns after the respawn each wait about twelve seconds and then come
   from the fallback ladder, look for `Request queue timeout ... while
   waiting on model_hidden_features (held N s)` with N growing: a request
   of the dead worker is still holding the model lane (F20 in
   `KNOWN_FAILURE_MODES.md`). Fixed on 17 October (8124687e7); if it is
   seen again, capture the launch log and report it.
5. A single crash with clean respawn needs no action. Repeated crashes on
   the same prompt: capture the prompt from the worker log, file it as a
   reproducer.

## Escalation

Respawn loop (3+ crashes in 10 min) → stop the desktop app, clear
`.venv` pycache if the crash storm followed a hard kill (see
project memory: June 11 segfault storm), restart via the desktop app.

# Unattended LoRA Training (macOS)

Foreground wrapper that drives `training/train_and_fuse.py` to completion
even with the lid closed. No launchd/plist, no daemons.

## Start

```bash
cd /Users/bryan/.aura/live-source
bash training/run_unattended.sh --tag mythos-v2
```

Forwarded flags: `--tag NAME`, `--base-model PATH`, `--skip-dataset`,
`--skip-train`.

Env knobs: `MAX_RETRIES` (default 5), `RETRY_PAUSE` (default 30s),
`AURA_PYTHON` (interpreter override).

## Resume

Re-run the same command. `run_unattended.py` detects existing
`training/adapters/aura-personality/*_adapters.safetensors` and resumes
via `training/resume_training.py` before any fresh-train phase.
State is persisted at
`training/adapters/aura-personality/training_state.json` after every
checkpoint observation, so re-spawns are idempotent.

## Logs & state

- Wrapper log: `training/logs/unattended_YYYYMMDD_HHMMSS.log` (tee'd)
- Resume log: `training/train_log.txt` (from `resume_training.py`)
- State JSON: `training_state.json` — fields: `started_at`, `last_iter`,
  `last_checkpoint_path`, `last_heartbeat`, `phase`, per-phase rc.

## Lid-close survival

The wrapper runs the orchestrator under `caffeinate -i -m -s -d`, which
prevents idle/disk/system/display sleep for the subprocess lifetime —
including when the lid is shut. `ulimit -n 4096` avoids mid-run FD
exhaustion. **macOS only** (caffeinate is mac-specific). On Linux,
swap caffeinate for `systemd-inhibit` or `setsid`.

## Safe abort

Kill the parent bash (Ctrl-C, or `kill <pid>`). The orchestrator
catches SIGINT/SIGTERM, writes a final state snapshot, terminates the
in-flight subprocess, and exits. Re-launch to resume.

## Smoke test

```bash
bash training/run_unattended.sh --skip-dataset --skip-train --tag dryrun-test
```

Exits 0 immediately — exercises wrapper plumbing without invoking the
trainer.

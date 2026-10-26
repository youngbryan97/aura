# Unattended LoRA Training (macOS)

<!-- reader-navigation:start -->
> **In plain English:** Learning changes later behavior using earlier outcomes. A kept procedure is executable work retained for reuse under a defined checking rule.
>
> **Why it matters:** Experience should improve later work without accumulating unreliable shortcuts or expensive rules that no longer help.
>
> **In everyday use:** After solving one kind of task, the system may reuse a checked procedure on a later task of that kind instead of deriving every step again.
>
> [Example and metaphor](../docs/PLAIN_ENGLISH_GUIDE.md#learning) · [Variables and practical meanings](../docs/VARIABLES_EXPLAINED.md#learning) · [Demos and related work](../docs/DEMOS_AND_RELATED_WORK.md#learning) · [Document map](../docs/DOCUMENTATION_INDEX.md)
> [Glossary](../docs/GLOSSARY.md) · [Formulas with worked examples](../docs/FORMULAS.md)
<!-- reader-navigation:end -->

Foreground wrapper that drives `training/train_and_fuse.py` to completion
even with the lid closed. No launchd/plist, no daemons.

## Start

```bash
cd <AURA_ROOT>
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

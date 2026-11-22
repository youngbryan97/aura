# Unattended LoRA Training (macOS)

<!-- reader-navigation:start -->
> Aura can learn a small program from examples. It keeps a program after it passes checks, then can call it again on a similar request.
>
> Keeping a checked method can save work the next time. A bad match can also send an answer off course. After learning how to count days between dates, Aura answered three new date questions using the method the next day.
>
> [Plain-language guide](../docs/PLAIN_ENGLISH_GUIDE.md#learning) · [Terms and examples](../docs/VARIABLES_EXPLAINED.md#learning) · [Evidence and related work](../docs/DEMOS_AND_RELATED_WORK.md#learning)
> [All docs](../docs/DOCUMENTATION_INDEX.md) · [Glossary](../docs/GLOSSARY.md) · [Worked formulas](../docs/FORMULAS.md)
<!-- reader-navigation:end -->

Foreground wrapper that runs the training orchestrator under a sleep
inhibitor and retries nonzero exits up to a declared limit. Completion,
lid-close survival and recovery after a host crash are not guaranteed.
Do not run it beside a resident model without an exclusive resource budget.

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
subject to macOS power and lid policy. A sleep assertion is not proof that
work continues through lid close, reboot or power loss. `ulimit -n 4096` sets
an FD limit; it does not establish that the workload cannot exhaust it. **macOS only** (caffeinate is mac-specific). On Linux,
port the wrapper and use an appropriate sleep inhibitor. `setsid` only
changes session attachment; it does not inhibit sleep.

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

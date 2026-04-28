# Aura — Operator Guide

This document is the runbook for running Aura on your own hardware.

## Requirements
- macOS Apple Silicon (M1 → M5) recommended.
- 32 GB+ RAM for the 32B Cortex; 64 GB+ to keep 32B and 72B warm.
- 50 GB+ free disk for models + data.
- Python 3.12.

## Install + boot
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup        # creates .venv, installs runtime requirements
make quality      # compile + lint + governance-lint + typecheck + smoke
make run          # foreground launch
```

## Backup & restore
- Backup: `tar czf aura-backup.tar.gz ~/.aura/data ~/.aura/live-source`.
- Restore: `tar xzf aura-backup.tar.gz -C ~/`.

## Diagnostics
- `aura doctor`             — pre-boot self-check (python, sqlite, mlx,
  data dir, atomic writer round-trip)
- `aura doctor --bundle [--bundle-path PATH]` — assembles a redacted
  tarball (health, config, metrics, tasks, models, memory, gateway,
  receipts, audit chain export, recent logs) for incident triage. The
  bundle is what every runbook in `docs/runbooks/` references.
- `aura conformance`        — schema + integrity sweep
- `aura verify-state`       — cross-subsystem state coherence
- `aura verify-memory`      — memory facade integrity
- `aura rebuild-index`      — vector index rebuild
- `aura chaos`              — fault injection smoke
- Dashboard: open `http://localhost:<port>/api/dashboard/snapshot` for a
  raw JSON view of every live subsystem.

## Service-level objectives
The contract operators can hold Aura to lives in [`docs/SLO.md`](SLO.md).
Numbers are measured by `python -m slo.measure` and gated in CI
(`.github/workflows/slo-gate.yml`); a regression past tolerance or a
hard-limit breach fails the release gate.

## Runbooks
Every documented incident class has a runbook under
[`docs/runbooks/`](runbooks/) with concrete symptoms tied to fields the
diagnostics bundle emits, plus diagnosis, mitigation, rollback, and
verification steps.

## Tamper-evident audit trail
Every receipt the runtime emits is appended to a hash-chained ledger at
`~/.aura/receipts/_chain.jsonl`. To verify the chain after an incident:
`python -c "from core.runtime.receipts import get_receipt_store;
print(get_receipt_store().verify_chain())"`. The diagnostics bundle
includes a portable export at `audit_chain/chain.jsonl` plus a
`MANIFEST.txt` with the head hash and length.

## Self-modification quarantine
When Aura proposes a code mutation, the typed evaluator in
`core/self_modification/mutation_safety.py` runs it in a subprocess
with rlimits and emits one of seven outcomes: `passed`, `compile_fail`,
`import_fail`, `runtime_exception`, `assertion_fail`, `timeout`, `oom`.
Any non-`passed` outcome is written to
`~/.aura/data/mutation_quarantine/<id>/` with the source, optional
test source, stdout, stderr, and a structured `result.json`. A
malformed mutation cannot crash the parent process.

## Reading logs
- Live tail: `tail -f ~/.aura/data/logs/aura.log`
- Receipt log: `~/.aura/data/agency_receipts/agency_receipts.jsonl`
- Will receipts: `~/.aura/data/will_receipts/receipts.jsonl`
- Stem cells: `~/.aura/data/stem_cells/`
- Migration ledger: `~/.aura/data/migration/ledger.jsonl`

## Service lifecycle
- macOS launchd: `launchctl load ~/Library/LaunchAgents/aura.plist`
- Linux systemd: `systemctl --user start aura`
- Stop: SIGTERM is graceful (drains receipts, revokes capability tokens).

## Model configuration
- `AURA_MODEL`        — primary model name (default: Qwen2.5-32B-Instruct-8bit)
- `AURA_DEEP_MODEL`   — heavy lane for solver tier
- `AURA_LLM__MLX_DEEP_MODEL_PATH` — explicit on-disk path
- Cloud fallback: Settings → Models → Cloud Fallback (off by default).

## Performance tuning
- Settings → Performance: cap on warm models (1, 2, or 3 concurrent
  heavy lanes), tick interval, dashboard refresh rate.
- Memory monitor lowers max_tokens at >85% RAM and triggers VRAM purge
  at >90% RAM (configurable via env: AURA_MEM_THRESHOLDS).

## Security settings
- Conscience: hard-line rules at `~/.aura/data/conscience/rules.sha256`
  — tampering refuses all actions until the file is restored.
- World bridge: per-channel permissions live at
  `~/.aura/data/world/permissions.json`.
- Capability tokens are bound to PID + thread. Restart revokes all live
  tokens.

## Governance lint
`make governance-lint` fails the build if any code outside the allow-list
makes a direct consequential call. The allow-list is defined in
`tools/lint_governance.py:ALLOW_LIST`.

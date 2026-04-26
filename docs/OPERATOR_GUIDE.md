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
- `aura doctor`           — runtime self-check
- `aura conformance`      — schema + integrity sweep
- `aura verify-state`     — cross-subsystem state coherence
- `aura verify-memory`    — memory facade integrity
- `aura rebuild-index`    — vector index rebuild
- `aura chaos --kind random` — fault injection (chaos engineering)
- Dashboard: open `http://localhost:<port>/api/dashboard/snapshot` for a
  raw JSON view of every live subsystem.

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

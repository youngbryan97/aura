# Q02 disk retention, 2026-09-13

Produced by `tools/disk_retention.py`; the classification is held by
`tests/test_disk_retention_tells_live_from_orphaned.py`. The tool reports and
removes nothing. A checkpoint, a fusion, a session's worktree: each is the
owner's to remove, and the tool's job is to make that decision possible by
saying what each thing is and whether anything reads it.

## What bounds what

| path | GB | status | last written (days) | bound, or why not |
|---|---|---|---|---|
| `/Users/bryan/.aura/logs` | 1.2 | bounded | — | rotation (bounds not read) |
| `/Users/bryan/.aura/data/aura_state.db` | 0.0 | live | — | StateRepository: DB_PAYLOAD_MAX_BYTES per row, prune every 100 commits, VACUUM every 1000 |
| `/Users/bryan/.aura/data/state/aura_state.db` | 46.0 | unreferenced | 161.6 | the vault writes data/aura_state.db; this path was the proxy's standalone fallback, unbounded until 2026-09-13, and nothing reads it |
| `/Users/bryan/.aura/live-source/training/fused-model/Aura-32B-20260510-151144` | 17.2 | live | — |  |
| `/Users/bryan/.aura/live-source/training/fused-model/Aura-32B-crsm-closeout-20260628-181638` | 17.2 | unreferenced | 76.7 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118` | 17.2 | live | — |  |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-pre-round3-snapshot` | 10.0 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-round1-zenith-2026-04-27` | 4.3 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality` | 4.0 | live | — |  |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-crsm-delta-20260628-173312` | 4.0 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-crsm-delta-20260701-212358` | 4.0 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-round2-final-2026-04-29` | 4.0 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-v4-backup` | 4.0 | unreferenced | 30.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.aura/live-source/training/adapters/aura-personality-iter7500` | 1.0 | unreferenced | 138.6 | no manifest, code, config or tool names this directory |
| `/Users/bryan/.cache/huggingface/hub/models--Systran--faster-whisper-base` | 0.15 | unreferenced | None |  |
| `/Users/bryan/.cache/huggingface/hub/models--mlx-community--Qwen3-32B-4bit` | 18.45 | unreferenced | None |  |
| `/Users/bryan/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit` | 16.08 | unreferenced | None |  |
| `/Users/bryan/.aura/live-source/artifacts` | 23.4 | unknown | — | sealed evidence; canaries pin what they read by hash |
| `/Users/bryan/.aura/training-campaigns` | 90.7 | unknown | — | campaign outputs; nothing prunes them |
| `/Users/bryan/.aura/knowledge` | 59.4 | unknown | — | knowledge stores; nothing prunes them |
| `/Users/bryan/.aura/data/stream_of_being` | 1.3 | unknown | — | append-only record |
| `/Users/bryan/.aura/data/outcome_ledger.db` | 0.3 | unknown | — | append-only ledger |

`status`: **bounded** — a mechanism holds it; **live** — something running or
configured names it; **unreferenced** — no code, manifest, config or tool
names it; **unknown** — the tool cannot tell and says so.

## The one that was unbounded, and is not now

`~/.aura/data/state/aura_state.db` is 46GB: 707 rows from March 23 to April 5,
2026, single rows of 932MB. The vault writes `~/.aura/data/aura_state.db`
(15MB, pruned every 100 commits, each row capped at 8MB). The 46GB file was
the proxy repository's *standalone fallback* — the path a process takes when it
boots the container with no vault — and that path serialized the whole state
every version with no cap and no pruning. Fixed 2026-09-13: the fallback goes
through the owner's bounded commit, and the "bounded" snapshot now bounds
`world.facts` and `world.user_preferences`, which it had left as they came (a
9MB fact rode through it whole; `tests/test_a_standalone_commit_is_bounded_like_the_owners.py`).
Nothing has read the 46GB file for five months. It is listed above and left.

## Worktrees

27 agent worktrees under `.claude/worktrees`, 84.6GB. A worktree is a
session's; whether it is finished is that session's owner's to say, so every one
is `unknown`. The one that matters:

| worktree | GB | last commit | untracked over 1GB |
|---|---|---|---|
| `codex-cp917-canary` | 59.0 | 2026-08-23 | artifacts/closeout/cortex_upgrade/cp918/ (58.3GB) |
| `codex-cp915-canary` | 2.4 | 2026-08-23 | artifacts/closeout/cortex_upgrade/cp916/ (1.7GB) |
| `codex-cp916-canary` | 1.6 | 2026-08-23 | — |
| `connectome-sep11` | 1.5 | 2026-09-13 | — |
| `agi-gauntlet-sep5` | 1.2 | 2026-09-09 | — |
| `isc-completion` | 1.2 | 2026-09-13 | — |

`codex-cp917-canary/artifacts/closeout/cortex_upgrade/cp918/training-runs/` is
58GB of untracked training-run output from a worktree last committed on
2026-08-23 — 97 safetensors files from the CP918 cortex-upgrade experiment.
Nothing in the tree names it.

## Logs

Bounded by rotation: `RotatingFileHandler`, 100MB per file, and the total is
1.2GB. No unrotated file over 200MB.

## Unreferenced, in total

129.2GB across 12 entries that no code, manifest, config or
tool names — plus the 58GB untracked run above. Listed, not removed.

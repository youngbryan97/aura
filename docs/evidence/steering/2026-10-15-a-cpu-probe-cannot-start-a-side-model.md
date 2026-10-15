# A CPU probe cannot start a side model worker

The first assigned-case-18 probe at 54d3c6f07 requested software browser
graphics, CPU OCR and classical points. Those controls did not cover a
separate scene-recognition route. Its output reported a Qwen3-VL-4B worker
loading on the GPU. The intended CPU restriction was therefore incomplete.

The diagnostic was stopped on discovering that output. Its child PID and
process group were checked against the supervisor record and command line
before sending SIGTERM to that group alone (48829). G09's process was in a
different group (4535), and no signal or control request was sent to it.
It remained running under PID 4559. This does not prove that the brief
additional GPU load had no effect on its throughput.

The supervisor retained return code -15, `passed=false`, after 68.25
seconds. The case checkpoint still has an unfinished attempt and cannot
be blindly resumed. No original-game success follows. The retained screen
at 26 seconds still shows the emulator's play overlay. A later log read
the game's start control; the run did not establish useful body control.
Raw evidence is in this worktree's
`artifacts/flash-campaign/2026-10-15-case18-cpu/`.

## Repair

`AURA_MODEL_WORKERS_ALLOWED=0` now refuses declared and inferred model
processes at the subprocess gateway. The vision client checks before its
model-lane claim and again before its blocking spawn. The worker entry
checks before importing MLX, selecting a device or starting its heartbeat.
The default permits the existing behavior. The policy does not stop or
reconfigure another process or an already-running resident model.

The archived runner's `--cpu-only` selects software graphics, acknowledged
CPU OCR and classical points, prohibits model-worker starts, and refuses
a model endpoint argument. The chosen policy is retained in campaign
identity. This closes the observed escape through scene recognition; it
does not claim to contain arbitrary third-party code or every future
unregistered accelerator call.

The regression set has 101 passes and one skip in 14.22 seconds. It
includes the existing vision-lane and subprocess-gateway contracts.
New tests refuse a vision request before any lane claim or spawn, call
the actual worker entry with accelerator imports forbidden, and reject
both declared and inferred generic model processes. No model is loaded
by these tests. The skipped check is an opt-in native OCR test already
measured in the preceding checkpoint. Lint, changed-file compilation,
layering, dependency consistency, changed-method size and writing passed.
Smoke has 165 passes and one skip in 45.31 seconds.

The broader governance gate still fails. Before these repairs it reported
124 regressions, including a pre-existing raw write in the corpus index;
the prior retained-material checkpoint had recorded 123 before that corpus
addition. None of the new CPU policy or patch-fit call sites appears in
that report. The baseline was not changed.

# Aura NetHack Challenge: The Embodied Cognition Stress Test

## Objective
Use NetHack (3.6.7) as the proof environment for Aura's general embodied
cognition substrate. The point is not a NetHack-only bot; the point is a
reusable loop that can perceive, maintain belief, quantify uncertainty, manage
risk, select goals/skills, gate actions, trace outcomes, and learn from
postmortems in any environment.

## Components
- **NetHack Adapter**: A terminal-grid adapter using the real NetHack binary
  in strict mode, with deterministic canary mode for CI.
- **Environment Kernel**: The canonical general loop:
  observation → typed state → belief/spatial memory → policy/HTN →
  simulation → governance → action gate → command compiler → execute →
  semantic diff → outcome learning → trace/postmortem.
- **Capability Matrix**: `core/environment/capability_matrix.py` verifies
  the live kernel has the required general organs before deep runs.
- **General Hardening Organs**: action semantics, action budgets, external
  proof gating, hindsight replay, abstraction discovery, curriculum generation,
  proof-kernel runtime bridging, startup prompt policy, tactical threat
  response, and concurrency health sampling are shared infrastructure, not
  NetHack strategy.
- **Monitoring**: Hash-chained black-box trace rows under
  `~/.aura/logs/nethack/` plus run-manager postmortems.

## Progress
- [x] Install NetHack & Dependencies
- [x] Implement NetHack Terminal Adapter
- [x] Implement Challenge Orchestrator
- [x] Add general embodied cognition runtime with belief, risk, goals, skills, action gating, traces, and postmortems
- [x] Replace the challenge loop with the canonical general EnvironmentKernel
- [x] Add executable capability audit, terminal death handling, run records, semantic outcome learning, and policy-driven action selection
- [x] Add final general hardening for action semantics, replay, abstraction discovery, curriculum, external proof, activation certainty, and concurrency health
- [x] Add general startup/modal, adapter-death, procedural-store, information-loop, and threat-response hardening from strict-real smoke
- [x] Update final documentation to the current code-grounded state
- [/] Autonomous Gameplay Loop (In Progress; strict-real NetHack is wired as a stress adapter)
- [ ] Record Successful Ascension
- [/] Repository Commit & Push

## How to Run
```bash
python challenges/nethack_challenge.py --mode strict_real --steps 5000
python challenges/nethack_challenge.py --mode simulated --steps 100
```

## Logs & Recordings
- Kernel Trace: `~/.aura/logs/nethack/kernel_trace.jsonl`
- Run/Postmortem: `EnvironmentKernel.run_manager.records`
- General capability documentation: `docs/GENERAL_ENVIRONMENT_AUTONOMY.md`

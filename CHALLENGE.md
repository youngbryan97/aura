# Aura NetHack Challenge: The Embodied Cognition Stress Test

## Objective
Use NetHack (3.6.7) as the proof environment for Aura's general embodied
cognition substrate. The point is not a NetHack-only bot; the point is a
reusable loop that can perceive, maintain belief, quantify uncertainty, manage
risk, select goals/skills, gate actions, trace outcomes, and learn from
postmortems in any environment.

## Components
- **NetHack Adapter**: A headless terminal wrapper using `pexpect` and `pyte`.
- **Embodied Cognition Runtime**: The general perception → belief → risk → goal → skill → action-gate → trace loop.
- **Challenge Orchestrator**: A stress adapter that feeds terminal states into the general runtime and Aura's cognitive pipeline.
- **Monitoring**: Continuous recording via `asciinema` and verbose logging.

## Progress
- [x] Install NetHack & Dependencies
- [x] Implement NetHack Terminal Adapter
- [x] Implement Challenge Orchestrator
- [x] Add general embodied cognition runtime with belief, risk, goals, skills, action gating, traces, and postmortems
- [/] Autonomous Gameplay Loop (In Progress; NetHack is wired as a stress adapter)
- [ ] Record Successful Ascension
- [ ] Final Documentation & Repository Push

## How to Run
```bash
python challenges/nethack_challenge.py
```

## Logs & Recordings
- Session Record: `~/.aura/logs/nethack/ascension.cast`
- Execution Log: `~/.aura/logs/nethack/challenge_*.log`

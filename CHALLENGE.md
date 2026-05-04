# Aura NetHack Challenge: The Ascension Protocol

## Objective
Enable Aura to autonomously play and defeat NetHack (3.6.7) from a fresh start.

## Components
- **NetHack Adapter**: A headless terminal wrapper using `pexpect` and `pyte`.
- **Challenge Orchestrator**: A specialized loop that feeds terminal states to Aura and executes her chosen keystrokes.
- **Monitoring**: Continuous recording via `asciinema` and verbose logging.

## Progress
- [x] Install NetHack & Dependencies
- [x] Implement NetHack Terminal Adapter
- [x] Implement Challenge Orchestrator
- [/] Autonomous Gameplay Loop (In Progress)
- [ ] Record Successful Ascension
- [ ] Final Documentation & Repository Push

## How to Run
```bash
python challenges/nethack_challenge.py
```

## Logs & Recordings
- Session Record: `~/.aura/logs/nethack/ascension.cast`
- Execution Log: `~/.aura/logs/nethack/challenge_*.log`

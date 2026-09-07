# R01 Runtime Survival

Measured on 2026-09-06, local desktop host. This closes process survival,
not the remaining R02-R11 acceptance requirements.

## Reproduction and Repair

Default launch PID 91312 disappeared after API startup. Attached launch
PID 91626 remained alive, reached readiness, and delivered a complete reply.
A model-free reproduction launched `nohup sleep 60` as PID 93102 and a
new-session sleep as PID 93105. After the execution shell returned, only
93105 remained. Ignoring SIGHUP did not isolate the process group.

Commit e82ccce56 makes the detached launcher exec through a new session,
with stdin disconnected and the same PID retained through exec. It does not
change runtime shutdown, singleton locks, or model ownership.

## Live Acceptance

- Attached predecessor 91626 shut down gracefully, exit code 0. Logs report
  container teardown clean and root runtime shutdown complete.
- The repaired default launcher exited 0 and left PID 94572 running with
  PPID 1 and PGID 94572. Boot returned ready=true and conversation_ready=true.
- With the conversation idle, authenticated POST /api/reboot returned
  scheduled=true, waiter_pid=96426, replacing_pid=94572.
- PID 94572 exited. The waiter execed aura_main.py as PID 96426, PPID 1,
  PGID 96426. Boot reached kernel_ready, ready=true.
- lsof reported exactly one port-8000 listener: PID 96426.

## Tests

- 60 launcher, reboot, and session-isolation tests passed.
- Smoke: 164 passed, 1 skipped.
- Subsequent reboot/session/guardian focused pass: 8 passed.

## Still Open

R02: the reboot inherits the predecessor's source snapshot. Health correctly
reports drift against e43870132, but still reports ready. Source identity
refresh and readiness semantics require separate repair and proof.

R06: e43870132 moves guardian report append and rotation off the event loop;
11 focused tests passed. Other synchronous writers remain, including
continuity, mind_model, and autonomy_conductor.

R09-R11: the attached run produced one complete spoon-conductivity reply,
but that single turn does not prove multi-turn reliability. It also suggested
touching oven-heated metal, which is an unacceptable example. Worker timings
were 2227 prefill tokens in 21.66s and 246 decode tokens in 28.00s. End-to-end
delivery took approximately 74s; preflight sight processing consumed 6.01s.
Those costs and response quality remain under review.

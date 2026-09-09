# R08: Keep the observer alive while the observed loop recovers

During resident turn `aura-chat-e96aa0b4-2ea8-4c3d-888b-50f597c60220`,
the neural stream reported 15.7-second event-loop lag and repeated hypervisor
shutdown/start messages. The later health pulse correctly reported that the
watchdog was no longer running. Restarting the monitor had converted measured
lag into missing supervision.

The hardening initializer supplied `is_alive` to lifecycle reconciliation.
For the hypervisor this includes both task existence and recent lag health.
The control plane therefore cancelled a running sampling task when its
measurements failed health. Restart admission could then defer its replacement.
The event-loop monitor already had a separate lifecycle probe, but registration
used that distinction only for the named monitor.

Hardening registration now prefers a component's published `is_running`
contract, retaining the previous liveness fallback for other components.
The hypervisor publishes that contract from its running flag and actual task
state. Runtime health still uses `is_alive`, including the unchanged healthy
sample count and recovery interval. Restart does not erase the lag evidence.
A cancelled or completed task can be started even if its old running flag
was never cleared.

Two regression tests failed before repair: reconciliation cancelled a real
live hypervisor task, and the lifecycle contract was absent. The repaired
initializer and control-plane suites passed 64 tests in 5.15 seconds. The
checks keep the original task through lag recovery and replace a genuinely
dead task while retaining its unrecovered health evidence.

The wider loop-budget, live-surface regression, sleep/startup, and admission
suites passed 174 tests in 34.08 seconds. Smoke, lint, compile, governance
lint, and layering passed (37 grandfathered layering entries).

This fixes lifecycle ownership, not the work that caused the event-loop lag.
Live replay remains required; R06 and R08 remain unchecked.

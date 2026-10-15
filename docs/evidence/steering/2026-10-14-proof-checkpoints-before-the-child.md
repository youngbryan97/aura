# Proof checkpoints before the child

The reboot removed the temporary diagnostic scripts and captures. The
shared proof supervisor also wrote its result only after its child ended.
A stopped supervisor could therefore leave no record of the command it
had launched. The supervisor now writes a prepared checkpoint before
launch and a running checkpoint before waiting. Both keep `passed=false`.
The running checkpoint names its supervisor, child and owned process group.

Checkpoint writes use the governed atomic file gateway. Failure to save
the running record kills and joins that supervisor's child. Signal handlers
are installed before the running record becomes visible. A received fatal
signal kills the owned group and leaves an interrupted terminal record.
Invalid, infinite and NaN time budgets are refused before launch. The
existing wall-clock sleep bound and sleep inhibitor remain in place.

The result still reports the command's exit status. Its task outcome is
explicitly left for independent review: a normally completed gameplay
script can exit zero after failing to win. A power loss can leave a running
checkpoint, which is evidence of unfinished work and never a passing result.
The supervisor cannot write a terminal record after its own SIGKILL or a
machine power loss. Evidence paths must be outside temporary storage.

Eight tests passed in 2.12 seconds before the handler-publication ordering
change. They exercise an actual child failure, timeout, a visible running
checkpoint followed by supervisor interruption, invalid budgets and a
refused running-checkpoint write. The final retained patch is checked again
before its checkpoint is pushed. No model or desktop runtime is launched
by these fixtures.

The final patch passed all eight tests in 2.66 seconds. Smoke passed
165 tests with one skip in 51.40 seconds. Ruff, compile, layering,
dependency graph and writing checks passed. The changed-method gate
reported no production Python change in this checkpoint. The existing
aggregate size excess and `ResponseGenerationPhase` method excess remain.

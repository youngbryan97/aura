# G09: retain outcomes that arrive during value refresh

The shared ActionValueModel cleared its stale flag after reading the outcome
ledger. If a new outcome arrived during that read, its notification was lost.
Two concurrent refreshes could also publish snapshots in reverse order.

Refreshes now have one owner. Evidence notifications increment a generation
under the existing short state lock; publication acknowledges only the
generation present when its read started. Notifications arriving during a
failed read also remain pending. Scoring and notifications do not acquire the
database-read lock, and the asynchronous refresh remains off the event loop.

Both notification tests failed before the repair. The repaired tests and the
existing marginal/contextual value suites passed 36 tests in 24.88 seconds.
An overlapping-reader test holds the first snapshot open, checks that the
second reader cannot overtake it, and verifies that the final score comes
from the newer snapshot.

Thirty additional tests passed in 5.99 seconds, including a real SQLite
resolution during the read and the production resolution observer. The next
refresh includes that measured failure and changes the action value from
1.0 to 0.5 with two observations. Event-loop, ledger and counterfactual
consumer regressions passed in the same run.

Lint, compile, governance, layering and writing passed. Smoke reported 163
passed, one skipped and the existing resident-manifest drift alarm in 262.13
seconds. The running steering qualification has not yet resolved that alarm.

This repairs retention of measured evidence in the common learner. It does
not establish task-grounded procedure selection or broad reasoning gain.

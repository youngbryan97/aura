# Audit repair pass

The baseline is the source assessment at `9d1b5bd145`. Historical audit files and
experiment receipts stay unchanged. Repairs are developed in this worktree;
they do not authorize a runtime restart or a new model load.

The active report-grounding campaign owns its model and measurement processes.
Validation runs one small process at a time with numerical threads capped,
resource admission, a process-tree RSS ceiling and a wall-clock bound. Large
suite chunks and model experiments wait until the host is available.

| Item | Acceptance evidence | State |
|---|---|---|
| SQLite readiness and shutdown | Forced setup interleaving retains both writes; failed/cancelled setup closes; sync entry cannot wedge the loop | Implemented, checks pending |
| State ownership and backlog | Thread/async updates and snapshots share one lock; slow consumers have bounded backlog and isolated snapshots | Open |
| Report validity | Echoed questions/private reasoning/ambiguous and nonfinite readings cannot become ratings; valid public ratings remain usable | Open |
| Evidence liveness | An unavailable or invalid liveness reader cannot leave a live claim citable | Open |
| Patch validation | Generated checks enforce their claimed public API contract; behavioral evidence is identified separately | Open |
| Causal measurement | Correlation and identified interventions have separate contracts and negative controls | Open |
| Steering qualification | A new protocol measures paired superiority against matched controls and lesions without rewriting old qualifications | Open |
| Learning lifecycle | A timed-out trainer stop cannot report quiescence or start a second trainer | Open |
| Whole-use and comparative proof | Fresh tasks, retained learning, ordinary computer use and independent comparison are measured on an available host | Deferred while campaign owns resources |
| Delivery | Source checks, bounded tests, precise commits and main checkpoints; open items retain their actual status | Open |

Passing a code contract does not establish broad competence, consciousness or
personhood. Experimental gaps remain open until the required result exists.

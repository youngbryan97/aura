"""core/interiority — what an event means to this agent, and what follows.

Forty-three mechanisms, each with its own appraisal, its own declared
refutation, and a place in a subsystem that existed before it. The
package is organised in three layers.

**Substrate.** :mod:`~core.interiority.receptors` puts adaptive gain in
the path so no channel can shout forever;
:mod:`~core.interiority.cleft` is the medium every faculty publishes
into, with probabilistic release, a clearance time constant and
third-party modulation. Both are items on the list and both are load
paths rather than descriptions.

**Appraisal.** :mod:`~core.interiority.appraisal` computes relational
meaning from an event and :mod:`~core.interiority.ledger` — what she is
holding. Change the wording and nothing moves; change what she is
committed to and everything does.
:mod:`~core.interiority.other_minds` infers another agent's readiness
from channel evidence read against their own baseline, and there is no
channel through which their actual state can be supplied.

**Faculties.** :mod:`~core.interiority.faculties` holds the forty-three.
:mod:`~core.interiority.arbitration` combines the ones that fire at
once, and :mod:`~core.interiority.service` lands the result on the
affect engine, the somatic marker gate, the drive budgets, the goal
stack, the curiosity queue and the memory retention check.

Importing the package registers its structural invariants and declares
its telemetry channels, so a runtime that has the module has the checks.
"""

from __future__ import annotations

from core.interiority import invariants as _invariants  # noqa: F401 — registers checks
from core.interiority import telemetry as _telemetry  # noqa: F401 — declares channels

__all__ = ["faculties", "service"]

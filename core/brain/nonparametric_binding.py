"""The identity a non-parametric memory operation runs under.

``NonParametricMemory`` knows how to enforce provenance and principal
isolation: ``add`` takes an ``EntryProvenance``, ``query`` takes a
principal, entries can be revoked by source and erased by principal. The
storage layer is correct.

The live seams did not carry any of it. The foreground logits processor
called ``memory.query(key, k=k)`` with no principal, which the store's own
docstring says searches EVERYTHING. The ingest path called
``_mem.add(key, token_id, ...)`` with no provenance, so every entry the
resident worker wrote defaulted to unattributed and unverified. An
abstraction whose isolation arguments can silently be empty is not
providing isolation at the point that matters.

So the identity is a TYPE that has to be built, and the seams refuse
rather than default:

* a foreground processor with no principal is not installed — recall is
  an enhancement, and running it unscoped is the leak itself
* an ingestor with no provenance raises at construction, so no code path
  can write an anonymous entry by omission

``tests/test_an_isolation_argument_that_can_be_empty.py`` walks the parse
tree of the live seam files and fails if either call ever loses its
keyword again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.brain.nonparametric_identity import EntryProvenance, TrustLevel

__all__ = [
    "LOCAL_OWNER_PRINCIPAL",
    "MemoryBinding",
    "binding_for_job",
    "ingest_provenance",
    "principal_of_the_turn",
]

#: The person a generation is for when nothing more particular is known. There is one local operator on this
#: machine, so a turn that names nobody is theirs; the boundary exists so that adding real principals later is a
#: change of value, not a change of shape (CP126 d0b864a8).
LOCAL_OWNER_PRINCIPAL = "local_owner"


def principal_of_the_turn(request: Any) -> str:
    """Whose turn a generation request is: the principal an authenticating layer put on it, or the local operator.

    Only ``authenticated_principal`` is read. A plain ``principal`` on a caller's mapping is a claim, not an
    authentication (core/brain/llm/mlx_client.py keeps it out of execution contexts for that reason), and naming
    someone else's principal would read their entries.

    MEASURED 10 November 2026: the worker refuses recall to a job that names nobody (``binding_for_job``), and no
    job the client built named anybody. Her log held 251 openings of her store and not one installed recall.
    """
    if isinstance(request, dict):
        value = request.get("authenticated_principal")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return LOCAL_OWNER_PRINCIPAL


@dataclass(frozen=True)
class MemoryBinding:
    """Who a memory operation is for, and on whose authority.

    Both fields are required and neither may be blank. The class exists so
    that "we forgot to pass the principal" is a construction error rather
    than a silent widening of what a query can see.
    """

    principal: str
    source_id: str
    trust: str = TrustLevel.UNVERIFIED
    verifier: str = ""
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.principal or "").strip():
            raise ValueError("a memory binding needs a principal; empty searches everything")
        if not str(self.source_id or "").strip():
            raise ValueError("a memory binding needs a source that can later be revoked")

    def provenance(self) -> EntryProvenance:
        return EntryProvenance(
            source_id=self.source_id,
            trust=self.trust,
            verifier=self.verifier,
            evidence_id=self.evidence_id,
            principal=self.principal,
        )


def binding_for_job(job: Any, *, source_id: str) -> MemoryBinding | None:
    """The binding a worker job authorizes, or None when it names nobody.

    Returning None is the refusal: a job that does not say whose turn this
    is gets no foreground recall. Reading another principal's memory
    because the request forgot to identify itself is the failure this
    exists to stop.
    """
    if not isinstance(job, dict):
        return None
    for key in ("principal", "user_id", "subject", "principal_id", "owner_id"):
        candidate = str(job.get(key) or "").strip()
        if candidate:
            return MemoryBinding(principal=candidate, source_id=source_id)
    return None


def ingest_provenance(
    *,
    principal: str,
    source_id: str,
    trust: str = TrustLevel.VERIFIED,
    verifier: str = "",
    evidence_id: str = "",
) -> EntryProvenance:
    """Provenance for one trusted-pair ingest. Refuses an anonymous write."""
    return MemoryBinding(
        principal=principal,
        source_id=source_id,
        trust=trust,
        verifier=verifier,
        evidence_id=evidence_id,
    ).provenance()

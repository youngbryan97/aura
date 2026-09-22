"""Shared relational evidence for transfer, perspective, and revision.

This is an adapter over Aura's existing language substrate, concept formation,
and agent-model APIs. It stores no second vocabulary and executes no second
program language. Its job is to preserve structure while surface names,
values, interpretations, and contexts change.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.cognition.agent_model import AgentModel

_WORD = re.compile(r"[A-Za-z0-9_]+")


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(str(text).lower()))


@dataclass(frozen=True, slots=True)
class RelationalCase:
    """A typed relation graph plus the public facts and goal it constrains."""

    case_id: str
    entities: tuple[tuple[str, str], ...]
    relations: tuple[tuple[str, str, str], ...]
    facts: tuple[tuple[str, str], ...] = ()
    goal: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        names = tuple(name for name, _kind in self.entities)
        if (
            not self.case_id
            or not names
            or len(set(names)) != len(names)
            or any(not name or not kind for name, kind in self.entities)
            or any(
                len(row) != 3 or not all(isinstance(x, str) and x for x in row)
                for row in self.relations
            )
            or any(
                left not in names or right not in names for _relation, left, right in self.relations
            )
        ):
            raise ValueError("relational case has invalid typed graph")

    @property
    def shape_key(self) -> tuple:
        """Canonical graph shape, invariant to entity names and declaration order."""
        names = tuple(name for name, _kind in self.entities)
        kinds = {name: kind for name, kind in self.entities}
        rows = tuple(
            (relation, names.index(left), names.index(right))
            for relation, left, right in self.relations
        )
        candidates = []
        for permutation in itertools.permutations(range(len(names))):
            inverse = {old: new for new, old in enumerate(permutation)}
            typed = tuple(kinds[names[old]] for old in permutation)
            edges = tuple(
                sorted((relation, inverse[left], inverse[right]) for relation, left, right in rows)
            )
            candidates.append((typed, edges, _words(self.goal)))
        return min(candidates)

    @property
    def invariant_facts(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(kind), str(value)) for kind, value in self.facts))


@dataclass(frozen=True, slots=True)
class Interpretation:
    """A revisable view of a case, separate from the case's observed facts."""

    hypothesis: str
    perspective_agent: str = ""
    believed_facts: tuple[tuple[str, bool], ...] = ()
    predicted_outcome: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    context_id: str
    outcome: str
    supports: bool
    falsification_attempt: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.context_id.strip() or not self.outcome.strip() or not self.evidence.strip():
            raise ValueError("evidence needs a context, outcome, and trace")
        if type(self.supports) is not bool or type(self.falsification_attempt) is not bool:
            raise ValueError("evidence verdicts must be measured booleans")


@dataclass
class PrincipleCandidate:
    """A principle whose status is earned by independent evidence."""

    shape_key: tuple
    hypothesis: str
    records: list[EvidenceRecord] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)

    @property
    def independent_contexts(self) -> int:
        return len({record.context_id for record in self.records})

    @property
    def support(self) -> int:
        return sum(record.supports for record in self.records)

    @property
    def contradictions(self) -> int:
        return sum(not record.supports for record in self.records)

    @property
    def falsifications(self) -> int:
        return sum(record.falsification_attempt for record in self.records)

    def status(self, *, minimum_support: int, minimum_contexts: int) -> str:
        if self.contradictions:
            return "needs_revision"
        if (
            self.support >= minimum_support
            and self.independent_contexts >= minimum_contexts
            and self.falsifications >= 1
        ):
            return "consolidated"
        return "provisional"


class RelationalGeneralizer:
    """One evidence path for structural transfer and interpretation revision."""

    def __init__(
        self,
        *,
        minimum_support: int = 3,
        minimum_contexts: int = 2,
        concept_engine: Any = None,
        agent_model: AgentModel | None = None,
    ) -> None:
        if (
            type(minimum_support) is not int
            or minimum_support < 1
            or type(minimum_contexts) is not int
            or minimum_contexts < 1
        ):
            raise ValueError("relational evidence thresholds must be positive integers")
        self.minimum_support = minimum_support
        self.minimum_contexts = minimum_contexts
        self.concept_engine = concept_engine
        self.agent_model = agent_model
        self._candidates: dict[tuple, PrincipleCandidate] = {}
        self._continuity: dict[str, tuple[str, ...]] = defaultdict(tuple)

    def same_problem(self, first: RelationalCase, second: RelationalCase) -> bool:
        """Compare structure, typed roles, and goal words, not literal names."""
        return first.shape_key == second.shape_key

    def observe(
        self,
        case: RelationalCase,
        interpretation: Interpretation,
        *,
        outcome: str,
        context_id: str,
        supports: bool,
        falsification_attempt: bool = False,
        evidence: str = "",
    ) -> PrincipleCandidate:
        """Add one outcome without turning a single example into a principle."""
        if not context_id or not outcome:
            raise ValueError("relational evidence needs context and outcome")
        record = EvidenceRecord(context_id, outcome, supports, falsification_attempt, evidence)
        if not interpretation.hypothesis.strip():
            raise ValueError("interpretations need hypotheses")
        key = (case.shape_key, interpretation.hypothesis)
        candidate = self._candidates.setdefault(
            key, PrincipleCandidate(case.shape_key, interpretation.hypothesis)
        )
        if record in candidate.records:
            return candidate
        candidate.records.append(record)
        if not supports:
            candidate.revisions.append(outcome)
            if self.concept_engine is not None:
                self.concept_engine.observe_prediction_error(
                    ["relational", *map(str, case.shape_key[0])], 1.0, context=context_id
                )
        self._continuity[case.context] = (*self._continuity[case.context], case.case_id)
        return candidate

    def revise(
        self, case: RelationalCase, old: Interpretation, new: Interpretation
    ) -> dict[str, Any]:
        """Change an interpretation while retaining the observed invariant facts."""
        if not old.hypothesis or not new.hypothesis:
            raise ValueError("interpretations need hypotheses")
        return {
            "case_id": case.case_id,
            "preserved_facts": case.invariant_facts,
            "old_hypothesis": old.hypothesis,
            "new_hypothesis": new.hypothesis,
            "changed_perspective": old.perspective_agent != new.perspective_agent,
            "changed_prediction": old.predicted_outcome != new.predicted_outcome,
        }

    def record_perspective(
        self, interpretation: Interpretation, *, world_truth: Mapping[str, bool], evidence: str
    ) -> dict[str, Any]:
        """Keep another agent's belief state distinct from world truth."""
        if self.agent_model is None or not interpretation.perspective_agent:
            return {"recorded": False, "reason": "no_agent_model"}
        for proposition, believes in interpretation.believed_facts:
            self.agent_model.observe_belief(
                proposition,
                supports=believes,
                evidence=evidence,
                about=interpretation.perspective_agent,
            )
        return {
            "recorded": True,
            "divergences": {
                proposition: believes != world_truth.get(proposition)
                for proposition, believes in interpretation.believed_facts
                if proposition in world_truth
            },
        }

    def candidate(self, case: RelationalCase, hypothesis: str) -> PrincipleCandidate | None:
        return self._candidates.get((case.shape_key, hypothesis))

    def continuity(self, context: str) -> tuple[str, ...]:
        return self._continuity.get(context, ())

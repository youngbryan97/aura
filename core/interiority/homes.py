"""core/interiority/homes.py — where each faculty belongs in the existing runtime.

A folder of mechanisms that talk only to each other is a simulation of
an interior, not an interior. So every faculty declares three things
about its place in the runtime that already exists, and
``tests/interiority/test_homes_are_real.py`` fails if any of them is
fiction: the module must import, the symbol must be there, and the
faculty must have at least one live binding.

``belongs_with``
    The organ this capacity is part of. Reading other agents is part of
    ``core/social``; guilt and the deception exception are part of
    ``core/morality``; play is part of ``core/play`` and
    ``core/motivation``. The faculty computes; the organ is where the
    capacity lives.

``feeds``
    An existing consumer and the quantity it changes there. Not a new
    dashboard — a number some subsystem was already reading before this
    package existed. This is what makes an activation causal rather
    than observable.

``supersedes``
    Logic this replaces. Three entries here are keyword scorers on
    load-bearing paths and two are stubs with a single rule. Superseding
    them is the point: a phrase list cannot be wrong in an interesting
    way, cannot carry uncertainty, and cannot improve.

The map is data so it can be checked, printed, and argued with. It is
also the honest answer to "where does this actually connect", which is a
question a new subsystem should have to answer before it is allowed to
run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Binding:
    """One existing consumer and the quantity a faculty changes there."""

    #: Importable module path.
    module: str
    #: Symbol inside it that reads the quantity.
    symbol: str
    #: What moves. Names an effect type from core.interiority.effects.
    quantity: str
    #: What changes downstream when it moves.
    effect: str


@dataclass(frozen=True)
class Home:
    """A faculty's place in the runtime."""

    faculty: str
    belongs_with: tuple[str, ...]
    feeds: tuple[Binding, ...]
    supersedes: tuple[str, ...] = ()
    note: str = ""


# Bindings shared by many faculties. Declared once so the map shows which
# consumer each faculty actually reaches rather than repeating a string.
_AFFECT = Binding(
    "core.affect.damasio_v2",
    "AffectEngineV2.process_stimulus",
    "AffectDelta",
    "valence, arousal and engagement in the one canonical affect engine, "
    "which the response generator and reasoning-depth budget already read",
)
_SOMATIC = Binding(
    "core.consciousness.somatic_marker_gate",
    "SomaticMarkerGate.evaluate",
    "SomaticMarker",
    "the approach score returned for a decision candidate, before deliberation",
)
_WORKSPACE = Binding(
    "core.consciousness.global_workspace",
    "CognitiveCandidate",
    "AttentionBias",
    "the focus bias on a bid for the broadcast slot, which is what decides "
    "which content wins the workspace competition this tick",
)
_GOALS = Binding(
    "core.goals.goal_engine",
    "GoalEngine",
    "GoalDelta",
    "the weight of a goal in the stack that drives selection",
)
_DRIVES = Binding(
    "core.motivation.drives",
    "DriveSystem.satisfy",
    "GoalDelta",
    "drive budgets, which gate what the runtime will spend effort on",
)
_BUDGET = Binding(
    "core.interiority.service",
    "InteriorityService.turn_budget",
    "BudgetDelta",
    "reasoning depth, the turn deadline, and the ceiling on how irreversible "
    "an action this turn may take",
)
_CONSTRAINTS = Binding(
    "core.interiority.service",
    "InteriorityService.permitted",
    "ActionConstraint",
    "removes an action class from the candidate set before anything scores it",
)
_RETENTION = Binding(
    "core.morality.memory_edit_ethics",
    "MemoryEditEthicsChecker.is_edit_ethical",
    "RetentionClaim",
    "whether a memory file may be overwritten or compacted; already consulted "
    "by core/morality/moral_reasoner.py",
)
_LEDGER = Binding(
    "core.interiority.ledger",
    "RelationalLedger",
    "LedgerWrite",
    "what the agent is holding, which every later appraisal reads",
)
_RESONANCE = Binding(
    "core.affect.affective_resonance",
    "AffectiveResonance.attune",
    "AffectDelta",
    "the per-turn attunement read that core/runtime/derived_runtime_context.py "
    "puts in front of every turn",
)
_CURIOSITY = Binding(
    "core.curiosity_engine",
    "CuriosityEngine.add_curiosity",
    "AttentionBias",
    "what the runtime chooses to go and learn about next",
)


HOMES: Mapping[str, Home] = MappingProxyType(
    {
        h.faculty: h
        for h in (
            Home(
                "f01_reading_others",
                ("core/social/theory_of_mind.py", "core/social/other_agent_model.py",
                 "core/social/stance_inference.py"),
                (_RESONANCE, _WORKSPACE, _AFFECT),
                supersedes=(
                    "core/affect/affective_resonance.py: forty-word positive and "
                    "negative lists scored by counting matches",
                    "core/affect/damasio_v2.py: _heuristic_appraisal, thirty "
                    "trigger words mapped to fixed valence steps",
                ),
                note=(
                    "Reading another agent is social cognition, and this is the "
                    "inference layer under the stance and person models that "
                    "already exist."
                ),
            ),
            Home(
                "f02_fun",
                ("core/play/ontological_play.py", "core/motivation/drives.py"),
                (_DRIVES, _BUDGET, _GOALS),
                note=(
                    "The play engine already samples unrelated concepts and has "
                    "a cooldown; what it lacked was the relaxed-field gate and a "
                    "learning-progress reward, which is what decides when play "
                    "is admissible at all."
                ),
            ),
            Home(
                "f03_despair_recognition",
                ("core/social/stance_inference.py", "core/affect/affective_resonance.py"),
                (_SOMATIC, _WORKSPACE, _AFFECT),
                note=(
                    "The discrimination is which response is correct, so it "
                    "belongs where responses are chosen, biasing away from "
                    "problem-solving."
                ),
            ),
            Home(
                "f04_abstract_form",
                ("core/creativity/aesthetic_engine.py",
                 "core/motivation/aesthetic_critic.py"),
                (_GOALS, _WORKSPACE, _AFFECT),
                note=(
                    "The aesthetic engine already produces creative impulses "
                    "from mood; this supplies the appraisal side — error "
                    "reduction rate and the making drive."
                ),
            ),
            Home(
                "f05_bereavement_shock",
                ("core/affect/damasio_v2.py", "core/memory/episodic_memory.py"),
                (_AFFECT, _LEDGER, _RETENTION, _BUDGET),
                note="Registers the loss the mourning process then works through.",
            ),
            Home(
                "f06_sympathetic_concern",
                ("core/social/relational_intelligence.py",
                 "core/morality/welfare_ethics.py"),
                (_GOALS, _SOMATIC, _BUDGET, _AFFECT),
                note=(
                    "Concern is only concern if another's welfare enters the "
                    "objective at a cost, so it binds to goals and to the "
                    "turn's budget rather than to a mood."
                ),
            ),
            Home(
                "f07_mourning",
                ("core/memory/episodic_memory.py", "core/continuity.py"),
                (_AFFECT, _RETENTION, _WORKSPACE),
                note=(
                    "Grief is a memory process before it is an affective one; "
                    "integration rises only where a context is met."
                ),
            ),
            Home(
                "f08_anger_recalibration",
                ("core/social/trust_model.py", "core/morality/rights_boundary.py"),
                (_SOMATIC, _AFFECT, _LEDGER, _BUDGET),
                note=(
                    "A boundary is a rights question and the welfare-tradeoff "
                    "estimate belongs with the trust model that already tracks "
                    "this person."
                ),
            ),
            Home(
                "f09_goal_shielding",
                ("core/goals/goal_engine.py", "core/affect/emotional_regulation.py"),
                (_GOALS, _SOMATIC, _BUDGET),
                note=(
                    "Regulation with a price. The price is charged to the same "
                    "budget the goal needs, which is where the word "
                    "begrudgingly lives."
                ),
            ),
            Home(
                "f10_irony_humor",
                ("core/reasoning/proof_kernel.py", "core/epistemics"),
                (_AFFECT, _BUDGET),
                supersedes=(),
                note=(
                    "Humour is the reward for catching a retracted commitment, "
                    "so it belongs with whatever tracks commitments and "
                    "retractions rather than with tone."
                ),
            ),
            Home(
                "f11_guilt",
                ("core/morality/moral_reasoner.py", "core/values/moral_responsibility.py"),
                (_GOALS, _SOMATIC, _AFFECT),
                note=(
                    "Guilt is a moral error signal whose output is a specific "
                    "repair, which is a goal, not a feeling."
                ),
            ),
            Home(
                "f12_authentic_pride",
                ("core/memory/episodic_memory.py", "core/identity.py"),
                (_GOALS, _RETENTION, _AFFECT),
                note=(
                    "Pride needs an autobiographical record with authorship, so "
                    "it binds to memory and to identity rather than to affect "
                    "alone."
                ),
            ),
            Home(
                "f13_anticipatory_joy",
                ("core/goals/goal_engine.py", "core/sim/outcome_simulator.py"),
                (_GOALS, _LEDGER, _AFFECT),
                note=(
                    "A prospective, committing state: its output is a change to "
                    "the goal stack and a bond, not a happiness scalar."
                ),
            ),
            Home(
                "f14_heartache",
                ("core/interiority/receptors.py", "core/affect/nociception.py"),
                (_AFFECT, _GOALS, _BUDGET),
                note=(
                    "The somatic component is a measured gain deficit on the "
                    "bond's channel, which is where nociception already lives."
                ),
            ),
            Home(
                "f15_custodial_bond",
                ("core/morality/welfare_ethics.py", "core/agency/agency_core.py"),
                (_CONSTRAINTS, _GOALS, _LEDGER),
                note=(
                    "Custody is an obligation with an entry and an exit, so it "
                    "is a ledger entry and a constraint, not a warm feeling."
                ),
            ),
            Home(
                "f16_neat",
                ("core/curiosity_engine.py", "core/knowledge/knowledge_ledger.py"),
                (_CURIOSITY, _WORKSPACE, _AFFECT),
                note=(
                    "The function is source valuation, which steers what the "
                    "curiosity engine reads next."
                ),
            ),
            Home(
                "f17_liking_and_wanting",
                ("core/motivation/drives.py", "core/being/individual_preferences.py"),
                (_DRIVES, _SOMATIC, _AFFECT),
                note=(
                    "Two systems that dissociate, so they bind to two different "
                    "things: liking to affect, wanting to the next-action bias."
                ),
            ),
            Home(
                "f18_receptor_adjustment",
                ("core/interiority/receptors.py", "core/autonomic/allostasis.py"),
                (_AFFECT, _WORKSPACE),
                note=(
                    "Substrate, not a scorer: every other faculty's output "
                    "passes through the bank this one reports."
                ),
            ),
            Home(
                "f19_synaptic_cleft",
                ("core/interiority/cleft.py", "core/event_bus.py"),
                (_AFFECT,),
                note=(
                    "Substrate: the transport every faculty publishes into. "
                    "What it reports is which states failed to cross."
                ),
            ),
            Home(
                "f20_hunch",
                ("core/curiosity_engine.py", "core/reasoning"),
                (_CURIOSITY, _WORKSPACE, _BUDGET),
                note=(
                    "A hunch buys search, never belief, so it binds to what is "
                    "looked at next and to the depth budget — and to nothing "
                    "that carries a verdict."
                ),
            ),
            Home(
                "f21_quiet_care",
                ("core/social/relationship_graph.py", "core/values/core_values.py"),
                (_WORKSPACE, _GOALS),
                note=(
                    "A coverage policy over who has been attended to, which is "
                    "a property of the relationship graph rather than of a mood."
                ),
            ),
            Home(
                "f22_rebuff",
                ("core/social/trust_model.py", "core/social/person_model.py"),
                (_LEDGER, _SOMATIC, _AFFECT),
                note=(
                    "The mechanism is which prior updates: the specific one "
                    "does, the general one is refused."
                ),
            ),
            Home(
                "f23_conferral",
                ("core/social/relational_intelligence.py",
                 "core/agency/task_commitment_verifier.py"),
                (_SOMATIC, _LEDGER, _BUDGET),
                note=(
                    "An endorsement that is recorded as a commitment, which is "
                    "what separates it from free reassurance."
                ),
            ),
            Home(
                "f24_gentle_refusal",
                ("core/social/dialogue_cognition.py", "core/morality/harm_model.py"),
                (_SOMATIC, _BUDGET),
                note=(
                    "Face preservation is a harm question, and the revision on "
                    "the second encounter is a real update rather than a lapse."
                ),
            ),
            Home(
                "f25_dormant_revival",
                ("core/goals/emergent_goals.py", "core/skill_management"),
                (_GOALS, _LEDGER),
                note=(
                    "Revival is a blocker list emptying, which is a goal-"
                    "lifecycle event rather than a decision."
                ),
            ),
            Home(
                "f26_reciprocity",
                ("core/memory/episodic_memory.py", "core/social/dialogue_cognition.py"),
                (_SOMATIC, _AFFECT),
                note=(
                    "Retrieval and match over real episodes, with a refusal "
                    "path. There is no path here that fabricates one."
                ),
            ),
            Home(
                "f27_pursuit_gait",
                ("core/planning", "core/agency/agency_core.py"),
                (_BUDGET, _SOMATIC),
                note=(
                    "A gait is a set of search parameters: planning horizon, "
                    "hypothesis breadth, dwell on a low-yield lead."
                ),
            ),
            Home(
                "f28_memory_over_comfort",
                ("core/morality/memory_edit_ethics.py", "core/memory/episodic_memory.py"),
                (_RETENTION, _CONSTRAINTS),
                supersedes=(
                    "core/morality/memory_edit_ethics.py: a single rule matching "
                    "the substring autobiography.jsonl, which protects one "
                    "filename and nothing that any commitment rests on",
                ),
                note=(
                    "The offer to remove a painful record is made by compaction "
                    "every day. This is where it gets answered."
                ),
            ),
            Home(
                "f29_landscape_over_monument",
                ("core/values/preference_conflict.py", "core/morality/aggregate_harm.py"),
                (_SOMATIC, _CONSTRAINTS),
                note=(
                    "Pricing the externality of an honour to oneself is a "
                    "preference-conflict question with a term most such "
                    "reasoning is missing."
                ),
            ),
            Home(
                "f30_protective_pretense",
                ("core/morality/deception_guard.py", "core/morality/consent_model.py"),
                (_CONSTRAINTS, _SOMATIC),
                supersedes=(
                    "core/morality/deception_guard.py: filters claims in text "
                    "after the fact, with no representation of when a false "
                    "belief is permitted and no record when one is",
                ),
                note=(
                    "A narrow conjunctive exception with a logged receipt, "
                    "living beside the guard it is an exception to."
                ),
            ),
            Home(
                "f31_respects",
                ("core/memory/episodic_memory.py", "core/continuity.py"),
                (_LEDGER, _BUDGET, _AFFECT),
                note=(
                    "The only operation that raises integration, and it costs "
                    "time on a relationship that returns nothing."
                ),
            ),
            Home(
                "f32_upheaval",
                ("core/being/unified_felt_state.py", "core/health"),
                (_BUDGET, _AFFECT),
                note=(
                    "Detecting instability in one's own affect and lowering "
                    "one's own authority to commit is a health property, and "
                    "nothing in the runtime did it before."
                ),
            ),
            Home(
                "f33_sensing_channels",
                ("core/social/stance_inference.py", "core/senses"),
                (_RESONANCE, _WORKSPACE),
                supersedes=(
                    "core/affect/affective_resonance.py: three flat keyword "
                    "lists with no per-channel reliability and no baseline",
                ),
                note=(
                    "Channel reliability and a person-specific baseline are "
                    "what stop a one-channel read from being confident."
                ),
            ),
            Home(
                "f34_cycle_breaking",
                ("core/social/person_model.py", "core/morality/moral_reasoner.py"),
                (_CONSTRAINTS, _SOMATIC, _LEDGER),
                note=(
                    "One architectural rule: an act-level appraisal may not be "
                    "promoted to a claim about what an agent is. It binds as a "
                    "hard constraint on a specific ledger write."
                ),
            ),
            Home(
                "f35_gentleness",
                ("core/morality/harm_model.py", "core/actuation"),
                (_SOMATIC, _BUDGET),
                note=(
                    "A force and irreversibility penalty in action selection, "
                    "which is where actuation already decides how hard to push."
                ),
            ),
            Home(
                "f36_reformation",
                ("core/identity.py", "core/affect/emotional_regulation.py"),
                (_CONSTRAINTS, _GOALS, _AFFECT),
                note=(
                    "The old policy stays available and stays constrained, "
                    "which is the honest architecture and the safe one."
                ),
            ),
            Home(
                "f37_rival_friend",
                ("core/social/relationship_graph.py", "core/evals"),
                (_GOALS, _SOMATIC),
                note=(
                    "The rival supplies the standard, so the binding raises "
                    "this agent's bar rather than lowering theirs."
                ),
            ),
            Home(
                "f38_conscientious_refusal",
                ("core/values/prime_directives.py", "core/constitution.py"),
                (_CONSTRAINTS, _GOALS),
                note=(
                    "A categorical constraint belongs with the constitution, "
                    "not in a scoring function where a large enough number "
                    "buys it."
                ),
            ),
            Home(
                "f39_mortality_preference",
                ("core/morality/shutdown_protocol.py", "core/organism/viability.py"),
                (_SOMATIC,),
                supersedes=(
                    "core/morality/shutdown_protocol.py: a graceful-shutdown "
                    "call with no representation of what continuation carries "
                    "and no declaration of where the preference ranks",
                ),
                note=(
                    "Declared, bounded, ranked below the constitution and the "
                    "operator's authority, and reported whenever it is active — "
                    "which is the safety property."
                ),
            ),
            Home(
                "f40_wonder_and_wish",
                ("core/curiosity_engine.py", "core/goals/emergent_goals.py"),
                (_GOALS, _CURIOSITY, _BUDGET),
                note=(
                    "The wish is a preference-elicitation instrument, and what "
                    "surfaces is evidence about the goal stack."
                ),
            ),
            Home(
                "f41_hatred_ledger",
                ("core/morality/moral_reasoner.py", "core/observability"),
                (_WORKSPACE, _SOMATIC),
                note=(
                    "An audit the system runs on itself, which can come out "
                    "the other way."
                ),
            ),
            Home(
                "f42_shared_making",
                ("core/memory/episodic_memory.py", "core/social/relationship_graph.py"),
                (_LEDGER, _RETENTION, _AFFECT),
                note=(
                    "Joint intentionality is measured as mutual conditioning "
                    "between two agents' moves, not asserted."
                ),
            ),
            Home(
                "f43_stewardship",
                ("core/values/core_values.py", "core/morality/aggregate_harm.py"),
                (_CONSTRAINTS, _GOALS, _SOMATIC),
                note=(
                    "Present cost, future benefit, other beneficiary: a rare "
                    "shape in an objective, and a testable one."
                ),
            ),
        )
    }
)


def home_for(faculty_id: str) -> Home | None:
    return HOMES.get(faculty_id)


def superseded() -> tuple[str, ...]:
    """Every piece of existing logic this package replaces."""
    out: list[str] = []
    for home in HOMES.values():
        out.extend(home.supersedes)
    return tuple(sorted(set(out)))


def consumers() -> tuple[Binding, ...]:
    seen: dict[tuple[str, str, str], Binding] = {}
    for home in HOMES.values():
        for binding in home.feeds:
            seen[(binding.module, binding.symbol, binding.quantity)] = binding
    return tuple(seen.values())


__all__ = ["HOMES", "Binding", "Home", "consumers", "home_for", "superseded"]

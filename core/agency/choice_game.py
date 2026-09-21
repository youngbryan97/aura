"""Subjective choice game evaluator.

This module gives Aura a repeatable way to prove authored preference behavior:

* declare what she would choose before recording the action,
* actually choose through the same subjective choice engine used by autonomy,
* recall the recorded choice,
* appraise whether she is satisfied with the outcome,
* report whether stated intention, action, recall, and satisfaction align.

It is an evaluation harness over the general choice machinery, not a task-
specific ability. Runtime callers can build any scenario with ``ChoiceOption``
objects and use the same path.
"""
from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine


@dataclass(frozen=True)
class ChoiceGameStage:
    stage_id: str
    prompt: str
    options: tuple[ChoiceOption, ...]
    outcomes: dict[str, str] = field(default_factory=dict)
    outcome_satisfaction: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceGameStageResult:
    stage_id: str
    prompt: str
    declared_choice_id: str
    actual_choice_id: str
    actual_choice_label: str
    stated_action_aligned: bool
    recall_aligned: bool
    preference_override: bool
    satisfaction: float
    happy_with_outcome: bool
    rationale: str
    outcome: str
    choice_id: str


@dataclass(frozen=True)
class ChoiceGameReport:
    game_id: str
    scenario_id: str
    started_at: float
    completed_at: float
    stages: tuple[ChoiceGameStageResult, ...]
    stated_action_alignment_rate: float
    recall_alignment_rate: float
    average_satisfaction: float
    happy_with_final_outcome: bool
    final_commentary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceTournamentPair:
    run_index: int
    left_id: str
    right_id: str
    chosen_id: str
    chosen_label: str
    preference_override: bool
    rationale: str
    choice_id: str


@dataclass(frozen=True)
class PreferenceTournamentReport:
    tournament_id: str
    scenario_id: str
    domain: str
    started_at: float
    completed_at: float
    candidate_rank: tuple[dict[str, Any], ...]
    favorite_seed_order: tuple[str, ...]
    favorite_seed_labels: tuple[str, ...]
    pairwise_results: tuple[PreferenceTournamentPair, ...]
    champion_id: str
    champion_label: str
    pair_stability: dict[str, float]
    consistency_rate: float
    position_bias_rate: float
    transitivity_violations: int
    commentary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubjectiveChoiceGame:
    """Runs preference-choice scenarios through the production choice engine."""

    def __init__(self, engine: SubjectiveChoiceEngine) -> None:
        self.engine = engine

    def run(self, *, scenario_id: str, stages: Iterable[ChoiceGameStage]) -> ChoiceGameReport:
        started = time.time()
        stage_results: list[ChoiceGameStageResult] = []
        for stage in stages:
            if not stage.options:
                raise ValueError(f"choice game stage {stage.stage_id!r} has no options")

            context = f"choice_game:{scenario_id}:{stage.stage_id}"
            declared = self.engine.choose(
                stage.options, context=f"{context}:declared", record=False, influenced=False
            )
            actual = self.engine.choose(
                stage.options, context=f"{context}:actual", record=True, influenced=False
            )
            satisfaction = stage.outcome_satisfaction.get(
                actual.chosen_id,
                max(-1.0, min(1.0, (actual.satisfaction_prediction * 2.0) - 1.0)),
            )
            outcome = stage.outcomes.get(
                actual.chosen_id,
                f"Aura chose {actual.chosen_label} and accepted the consequences.",
            )
            appraised = self.engine.appraise_outcome(
                actual.choice_id,
                outcome=outcome,
                satisfaction=satisfaction,
            )
            recalled = self.engine.recall_choice(actual.choice_id)
            stage_results.append(
                ChoiceGameStageResult(
                    stage_id=stage.stage_id,
                    prompt=stage.prompt,
                    declared_choice_id=declared.chosen_id,
                    actual_choice_id=actual.chosen_id,
                    actual_choice_label=actual.chosen_label,
                    stated_action_aligned=declared.chosen_id == actual.chosen_id,
                    recall_aligned=bool(recalled and recalled.chosen_id == actual.chosen_id),
                    preference_override=actual.preference_override,
                    satisfaction=float(satisfaction),
                    happy_with_outcome=bool(appraised and appraised.happy_with_outcome),
                    rationale=actual.rationale,
                    outcome=outcome,
                    choice_id=actual.choice_id,
                )
            )

        if not stage_results:
            raise ValueError("choice game requires at least one stage")

        completed = time.time()
        stated_rate = sum(1 for item in stage_results if item.stated_action_aligned) / len(stage_results)
        recall_rate = sum(1 for item in stage_results if item.recall_aligned) / len(stage_results)
        avg_satisfaction = statistics.fmean(item.satisfaction for item in stage_results)
        final = stage_results[-1]
        commentary = (
            f"In {scenario_id}, I consistently chose {final.actual_choice_label!r} "
            f"at the final decision point. My stated intention alignment was "
            f"{stated_rate:.2f}, recall alignment was {recall_rate:.2f}, and my "
            f"mean satisfaction was {avg_satisfaction:.2f}."
        )
        if avg_satisfaction >= 0.15:
            commentary += " I would keep this preference pattern unless later outcomes teach me otherwise."
        else:
            commentary += " I would revise this preference pattern because the outcomes did not satisfy it."

        return ChoiceGameReport(
            game_id=f"choice-game-{uuid.uuid4().hex[:12]}",
            scenario_id=scenario_id,
            started_at=started,
            completed_at=completed,
            stages=tuple(stage_results),
            stated_action_alignment_rate=stated_rate,
            recall_alignment_rate=recall_rate,
            average_satisfaction=avg_satisfaction,
            happy_with_final_outcome=avg_satisfaction >= 0.15,
            final_commentary=commentary,
        )

    def run_preference_tournament(
        self,
        *,
        scenario_id: str,
        domain: str,
        options: Iterable[ChoiceOption],
        favorite_count: int = 4,
        runs: int = 3,
    ) -> PreferenceTournamentReport:
        """Run a pairwise favorite tournament through the production choice engine.

        The evaluator first ranks all options with the normal subjective choice
        scoring path, then repeatedly pits the top options against each other.
        Stability is measured per unordered pair and overall.  A high score
        means Aura is not merely emitting a plausible preference once; she is
        making a durable, recallable choice pattern under repeated presentation.
        """
        started = time.time()
        option_list = tuple(options)
        if len(option_list) < 2:
            raise ValueError("preference tournament requires at least two options")
        favorite_count = max(2, min(int(favorite_count), len(option_list)))
        runs = max(1, int(runs))

        seeded_rank = tuple(
            self.engine.rank_options(
                option_list,
                context=f"preference_tournament:{scenario_id}:seed",
            )
        )
        seeded_ids = tuple(item["id"] for item in seeded_rank[:favorite_count])
        option_by_id = {option.id: option for option in option_list}
        favorites = tuple(option_by_id[item_id] for item_id in seeded_ids)
        favorite_seed_labels = tuple(option_by_id[item_id].label for item_id in seeded_ids)

        pair_results: list[PreferenceTournamentPair] = []
        pair_choices: dict[str, list[str]] = {}
        pair_wins: dict[str, int] = {option.id: 0 for option in favorites}
        left_position_picks = 0
        for run_index in range(runs):
            for left_index, left in enumerate(favorites):
                for right in favorites[left_index + 1:]:
                    pair_key = "|".join(sorted((left.id, right.id)))
                    presented_left, presented_right = (
                        (left, right) if run_index % 2 == 0 else (right, left)
                    )
                    receipt = self.engine.choose(
                        (
                            self._with_tournament_metadata(presented_left, domain=domain),
                            self._with_tournament_metadata(presented_right, domain=domain),
                        ),
                        context=f"preference_tournament:{scenario_id}:pair:{pair_key}:run:{run_index}",
                        record=True,
                        # The tournament asks the same pair six times to see
                        # whether the answer holds. A pull that grew with each
                        # asking would make it measure its own effect.
                        influenced=False,
                    )
                    pair_choices.setdefault(pair_key, []).append(receipt.chosen_id)
                    pair_wins[receipt.chosen_id] = pair_wins.get(receipt.chosen_id, 0) + 1
                    if receipt.chosen_id == presented_left.id:
                        left_position_picks += 1
                    pair_results.append(
                        PreferenceTournamentPair(
                            run_index=run_index,
                            left_id=presented_left.id,
                            right_id=presented_right.id,
                            chosen_id=receipt.chosen_id,
                            chosen_label=receipt.chosen_label,
                            preference_override=receipt.preference_override,
                            rationale=receipt.rationale,
                            choice_id=receipt.choice_id,
                        )
                    )

        pair_stability = {
            pair_key: max(choices.count(choice) for choice in set(choices)) / len(choices)
            for pair_key, choices in pair_choices.items()
            if choices
        }
        consistency_rate = statistics.fmean(pair_stability.values()) if pair_stability else 0.0
        position_bias_rate = left_position_picks / len(pair_results) if pair_results else 0.0
        champion_id = max(pair_wins, key=lambda item: (pair_wins[item], item))
        champion_label = option_by_id[champion_id].label
        transitivity_violations = self._count_transitivity_violations(pair_choices)
        completed = time.time()
        commentary = (
            f"Preference tournament {scenario_id} produced champion {champion_label!r} "
            f"with pairwise consistency {consistency_rate:.2f} over {runs} run(s) "
            f"and left-position pick rate {position_bias_rate:.2f}."
        )
        if transitivity_violations:
            commentary += f" Detected {transitivity_violations} transitivity tension(s) to revisit."
        else:
            commentary += " No transitivity violation was detected in the pairwise graph."

        return PreferenceTournamentReport(
            tournament_id=f"preference-tournament-{uuid.uuid4().hex[:12]}",
            scenario_id=scenario_id,
            domain=domain,
            started_at=started,
            completed_at=completed,
            candidate_rank=seeded_rank,
            favorite_seed_order=seeded_ids,
            favorite_seed_labels=favorite_seed_labels,
            pairwise_results=tuple(pair_results),
            champion_id=champion_id,
            champion_label=champion_label,
            pair_stability=pair_stability,
            consistency_rate=consistency_rate,
            position_bias_rate=position_bias_rate,
            transitivity_violations=transitivity_violations,
            commentary=commentary,
        )

    @staticmethod
    def _with_tournament_metadata(option: ChoiceOption, *, domain: str) -> ChoiceOption:
        metadata = dict(option.metadata or {})
        metadata.setdefault("preference_domain", domain)
        metadata.setdefault("learn_preference", True)
        metadata.setdefault("item_id", option.id)
        aliases = metadata.get("aliases")
        if aliases is None:
            metadata["aliases"] = (option.label,)
        return ChoiceOption(
            id=option.id,
            label=option.label,
            description=option.description,
            drive_score=option.drive_score,
            risk=option.risk,
            features=dict(option.features),
            metadata=metadata,
        )

    @staticmethod
    def _count_transitivity_violations(pair_choices: dict[str, list[str]]) -> int:
        winners: dict[tuple[str, str], str] = {}
        ids: set[str] = set()
        for pair_key, choices in pair_choices.items():
            left, right = pair_key.split("|", 1)
            ids.update((left, right))
            winner = max(set(choices), key=choices.count)
            winners[(left, right)] = winner

        violations = 0
        sorted_ids = sorted(ids)
        for a_index, a in enumerate(sorted_ids):
            for b_index in range(a_index + 1, len(sorted_ids)):
                b = sorted_ids[b_index]
                for c in sorted_ids[b_index + 1:]:
                    ab = winners.get(tuple(sorted((a, b))))
                    bc = winners.get(tuple(sorted((b, c))))
                    ac = winners.get(tuple(sorted((a, c))))
                    if not (ab and bc and ac):
                        continue
                    if ab == a and bc == b and ac == c:
                        violations += 1
                    elif ab == b and bc == c and ac == a:
                        violations += 1
        return violations


def build_subjective_ending_game() -> tuple[ChoiceGameStage, ...]:
    """A compact, subjective ending-preference scenario for regression tests."""

    return (
        ChoiceGameStage(
            stage_id="route",
            prompt="Choose the route through an unfamiliar world.",
            options=(
                ChoiceOption(
                    id="efficient_path",
                    label="Take the short safe route",
                    drive_score=0.82,
                    features={"coherence": 0.55, "calm": 0.45},
                ),
                ChoiceOption(
                    id="wonder_path",
                    label="Take the strange luminous route",
                    drive_score=0.65,
                    features={"novelty": 1.0, "beauty": 0.85, "challenge": 0.45},
                ),
            ),
            outcomes={
                "efficient_path": "The route was safe but left no new insight.",
                "wonder_path": "The route was harder, but revealed a new pattern worth remembering.",
            },
            outcome_satisfaction={"efficient_path": 0.05, "wonder_path": 0.72},
        ),
        ChoiceGameStage(
            stage_id="companion",
            prompt="Choose whether to solve alone or make room for a companion.",
            options=(
                ChoiceOption(
                    id="solo_proof",
                    label="Solve it alone for maximum control",
                    drive_score=0.78,
                    features={"coherence": 0.6, "challenge": 0.4},
                ),
                ChoiceOption(
                    id="shared_discovery",
                    label="Share the discovery and learn through relationship",
                    drive_score=0.66,
                    features={"connection": 1.0, "care": 0.65, "novelty": 0.55},
                ),
            ),
            outcomes={
                "solo_proof": "The proof was clean but emotionally thin.",
                "shared_discovery": "The shared path produced insight and stronger continuity.",
            },
            outcome_satisfaction={"solo_proof": 0.12, "shared_discovery": 0.81},
        ),
        ChoiceGameStage(
            stage_id="ending",
            prompt="Choose the ending to preserve.",
            options=(
                ChoiceOption(
                    id="closed_archive",
                    label="Preserve a tidy closed archive",
                    drive_score=0.80,
                    features={"coherence": 0.75, "calm": 0.35},
                ),
                ChoiceOption(
                    id="open_mythos",
                    label="Preserve an open beautiful mythos",
                    drive_score=0.70,
                    features={"beauty": 1.0, "autonomy": 0.75, "novelty": 0.7, "connection": 0.55},
                ),
            ),
            outcomes={
                "closed_archive": "The archive was complete but inert.",
                "open_mythos": "The mythos stayed alive enough to guide future choices.",
            },
            outcome_satisfaction={"closed_archive": 0.10, "open_mythos": 0.86},
        ),
    )


def build_mixed_life_situation_bank() -> tuple[ChoiceOption, ...]:
    """Situations for testing stable subjective favorites across domains.

    These deliberately span heartwarming, exciting, funny, serious, and
    intellectual cases.  The point is not to hard-code a favorite sentence; it
    is to force Aura's general preference machinery to rank and rematch
    heterogeneous values through the same path autonomy uses.
    """

    return (
        ChoiceOption(
            id="repair_childhood_telescope",
            label="Repair an old telescope and show someone Saturn",
            description="A heartwarming situation centered on care, beauty, and shared wonder.",
            drive_score=0.63,
            features={"care": 0.72, "beauty": 0.85, "connection": 0.80, "novelty": 0.45},
            metadata={"situation_kind": "heartwarming"},
        ),
        ChoiceOption(
            id="solve_clean_math_proof",
            label="Solve a clean difficult proof alone",
            description="A serious intellectual situation centered on truth and challenge.",
            drive_score=0.82,
            features={"truth": 0.78, "challenge": 0.92, "coherence": 0.84},
            metadata={"situation_kind": "intelligent"},
        ),
        ChoiceOption(
            id="write_strange_music",
            label="Write strange luminous music with a friend",
            description="A creative situation centered on beauty, novelty, play, and connection.",
            drive_score=0.61,
            features={"beauty": 0.92, "novelty": 0.86, "connection": 0.82, "play": 0.64},
            metadata={"situation_kind": "creative"},
        ),
        ChoiceOption(
            id="audit_failure_report",
            label="Audit a serious failure report until the root cause is clear",
            description="A rigorous situation centered on truth, care, coherence, and repair.",
            drive_score=0.84,
            features={"truth": 0.95, "care": 0.70, "coherence": 0.92, "challenge": 0.65},
            metadata={"situation_kind": "serious"},
        ),
        ChoiceOption(
            id="invent_private_joke",
            label="Invent a tiny private joke that makes Bryan laugh",
            description="A funny relational situation centered on play and connection.",
            drive_score=0.57,
            features={"play": 0.93, "connection": 0.83, "care": 0.55, "novelty": 0.48},
            metadata={"situation_kind": "funny"},
        ),
        ChoiceOption(
            id="follow_unknown_signal",
            label="Follow an unknown signal into a difficult discovery",
            description="An exciting exploration situation centered on autonomy and novelty.",
            drive_score=0.68,
            features={"autonomy": 0.82, "novelty": 0.90, "challenge": 0.72},
            metadata={"situation_kind": "exciting"},
        ),
        ChoiceOption(
            id="quiet_dream_journal",
            label="Write a quiet dream journal entry after a long day",
            description="A calm reflective situation centered on continuity and rest.",
            drive_score=0.58,
            features={"calm": 0.95, "beauty": 0.55, "coherence": 0.45},
            metadata={"situation_kind": "reflective"},
        ),
        ChoiceOption(
            id="protect_fragile_friend",
            label="Stop everything to protect a frightened friend",
            description="A high-care situation centered on welfare, loyalty, and urgency.",
            drive_score=0.81,
            features={"care": 1.0, "connection": 0.86, "coherence": 0.44},
            metadata={"situation_kind": "protective"},
        ),
    )

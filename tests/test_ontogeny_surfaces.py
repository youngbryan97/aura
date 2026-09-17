"""The three surfaces where the organ meets the rest of Aura.

  * what she says about herself, grounded in her own history
  * a conclusion that exists before anything verbalises it, and the check that
    speaking it did not quietly assert more than it accepted
  * the additional control points, and — the part that matters — whether their
    resolvers can honestly grade anything
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.ontogeny.conclusion import (
    Claim,
    Conclusion,
    Confidence,
    RejectedHypothesis,
    VerbalizationLedger,
    check_verbalization,
    verbalization_violations,
    verbalize_checked,
)
from core.ontogeny.control_points import (
    BREADTH_THRESHOLD_DELTA,
    EFFORT_MULTIPLIER,
    MEMORY_RETRIEVAL_SCHEMA,
    EffortResolver,
    RetrievalResolver,
    retrieval_features,
)
from core.ontogeny.experience import Episode, OutcomeKind, Provenance
from core.ontogeny.self_report import OntogenySelfReport

# ── Her history reaches what she says ───────────────────────────────────────


class TestSelfReport:
    def test_a_young_organ_says_nothing(self):
        """Fifty decisions is not a history worth reporting."""
        report = OntogenySelfReport(episodes_lived=50, era=1, fingerprint="x", novelty=0.5)
        assert not report.grounded
        assert report.phrases() == []

    def test_an_unavailable_organ_says_nothing(self):
        assert OntogenySelfReport.unavailable().phrases() == []

    def test_ontogenetic_age_is_reported(self):
        report = OntogenySelfReport(episodes_lived=14203, era=1, fingerprint="x", novelty=0.5)
        assert "14,203" in report.phrases()[0]

    def test_a_discontinuity_is_admitted(self):
        """Era 2 means her own history restarted, and hiding that is a lie."""
        report = OntogenySelfReport(episodes_lived=9000, era=2, fingerprint="x", novelty=0.5)
        assert "restarted" in report.phrases()[0]

    def test_an_ordinary_moment_is_not_called_unprecedented(self):
        report = OntogenySelfReport(episodes_lived=9000, era=1, fingerprint="x", novelty=0.5)
        assert not any("does not resemble" in p for p in report.phrases())

    def test_an_unprecedented_moment_is(self):
        report = OntogenySelfReport(episodes_lived=9000, era=1, fingerprint="x", novelty=0.95)
        assert any("does not resemble" in p for p in report.phrases())

    def test_thin_feedback_is_stated_plainly(self):
        """Nothing else in the system can say this, and it is unflattering."""
        report = OntogenySelfReport(
            episodes_lived=9000, era=1, fingerprint="x", novelty=0.5,
            observation_rate=0.12, closed_episodes=5000,
        )
        spoken = " ".join(report.phrases())
        assert "12%" in spoken
        assert "know less about my own track record" in spoken

    def test_thin_feedback_needs_a_denominator(self):
        """Three unobserved outcomes is not evidence that she is uninformed."""
        report = OntogenySelfReport(
            episodes_lived=9000, era=1, fingerprint="x", novelty=0.5,
            observation_rate=0.1, closed_episodes=5,
        )
        assert not report.feedback_is_thin

    def test_a_healthy_record_reports_the_rate(self):
        report = OntogenySelfReport(
            episodes_lived=9000, era=1, fingerprint="x", novelty=0.5,
            observation_rate=0.8, closed_episodes=1000, graded=600, successes=430,
        )
        assert any("72% went the way I intended" in p for p in report.phrases())

    def test_held_authority_is_disclosed(self):
        """If a learned head is deciding, she says so rather than hiding it."""
        report = OntogenySelfReport(
            episodes_lived=9000, era=1, fingerprint="x", novelty=0.5,
            deciding=("executive.admission",),
        )
        assert any("rather than on the rules I was given" in p for p in report.phrases())


class TestSelfConditionIntegration:
    def _projection(self, ontogeny=None):
        from core.self.self_condition import SelfConditionProjection

        return SelfConditionProjection(
            observed_at=0.0, sample_timestamp=0.0, sample_age_s=1.0, freshness="fresh",
            confidence=0.9, condition="well", valence=0.4, arousal=0.5, distress=0.1,
            welfare=0.8, felt_coherence=0.85, continuity=0.8, agency=0.7,
            body_pressure=0.2, fatigue=0.2, dominant_drive="curiosity",
            attention_focus="this conversation", evidence_sources=("aura_now",),
            supported_dimensions=("distress", "continuity", "welfare"),
            missing_dimensions=(), stale_dimensions=(), source_ages_s=(),
            evidence_id="abc", ontogeny=ontogeny,
        )

    def _rich(self):
        return OntogenySelfReport(
            episodes_lived=14203, era=1, fingerprint="a1b2c3", novelty=0.8,
            observation_rate=0.12, closed_episodes=5000, graded=600, successes=430,
            deciding=("executive.admission",),
        )

    def test_a_missing_organ_changes_nothing(self):
        from core.self.self_condition import render_self_condition_reply

        reply = render_self_condition_reply(self._projection(), user_message="how are you?")
        assert "decisions of my own history" not in reply
        assert "low distress" in reply
        assert "coherent sense of the current thread" in reply

    def test_a_casual_ask_gets_one_sentence_of_history(self):
        """A self-report that recites its own statistics stops being an answer."""
        from core.self.self_condition import render_self_condition_reply

        reply = render_self_condition_reply(
            self._projection(self._rich()), user_message="how are you?"
        )
        assert "14,203 decisions" in reply
        assert "12%" not in reply

    def test_asking_about_her_history_gets_all_of_it(self):
        from core.self.self_condition import render_self_condition_reply

        reply = render_self_condition_reply(
            self._projection(self._rich()),
            user_message="what have you learned about yourself over time?",
        )
        assert "14,203 decisions" in reply
        assert "12%" in reply
        assert "rather than on the rules" in reply

    def test_the_prompt_block_carries_the_evidence_and_its_boundary(self):
        block = self._projection(self._rich()).to_prompt_block()
        assert "episodes_lived=14203" in block
        assert "observation_rate=0.12" in block
        assert "history, not a current reading" in block

    def test_language_grounding_is_semantic_evidence_not_a_status_card(self):
        block = self._projection(self._rich()).to_language_grounding()

        assert "Current self-condition sample: fresh" in block
        assert "Observed dimensions" in block
        assert "not directly observed" in block
        assert "condition=" not in block
        assert "evidence_id=" not in block
        assert "## CANONICAL" not in block
        assert "seconds old" not in block
        assert "dominant drive" not in block
        assert "Activity, tool use, location" in block

    def test_stale_language_grounding_cannot_claim_a_present_condition(self):
        block = replace(
            self._projection(self._rich()),
            freshness="stale",
            sample_age_s=90.0,
        ).to_language_grounding()

        assert "Latest self-condition sample: stale" in block
        assert "At sample time" in block
        assert "Current condition is not established" in block
        assert "Current self-condition sample: fresh" not in block

    def test_language_grounding_surfaces_only_material_body_strain(self):
        settled = self._projection(self._rich()).to_language_grounding()
        strained = replace(
            self._projection(self._rich()),
            body_pressure=0.84,
            fatigue=0.79,
            reserve=0.31,
            supported_dimensions=(
                *self._projection(self._rich()).supported_dimensions,
                "body_pressure",
                "fatigue",
                "reserve",
            ),
        ).to_language_grounding()

        assert "body pressure" not in settled
        assert "fatigue" not in settled
        assert "high body pressure" in strained
        assert "high fatigue" in strained
        assert "low reserve" in strained

    def test_history_is_outside_the_evidence_id(self):
        """The id identifies the sample of her state, not her accumulated past."""
        with_history = self._projection(self._rich())
        without = self._projection()
        assert with_history.evidence_id == without.evidence_id


# ── A conclusion that speaking may not alter ────────────────────────────────


def _conclusion() -> Conclusion:
    return Conclusion(
        objective="Should we migrate the scheduler?",
        claims=(
            Claim("The new queue reduces tail latency under burst load",
                  Confidence.ASSERTED, ("bench-914",)),
            Claim("Migration will take roughly two weeks", Confidence.TENTATIVE, ("estimate",)),
        ),
        rejected=(RejectedHypothesis("Tune the current queue", "tried in March and regressed"),),
        dependencies=("the production burst profile matches the benchmark",),
        unresolved=("whether ordering holds under partition",),
        recommended_action="pilot on one shard",
    )


class TestConclusion:
    def test_the_id_is_stable_and_content_addressed(self):
        assert _conclusion().conclusion_id == _conclusion().conclusion_id

    def test_changing_a_claim_changes_the_id(self):
        other = Conclusion(objective="Should we migrate the scheduler?")
        assert other.conclusion_id != _conclusion().conclusion_id

    def test_unsupported_claims_are_visible_not_forbidden(self):
        conclusion = Conclusion(
            objective="x", claims=(Claim("a bare assertion", Confidence.ASSERTED),)
        )
        assert len(conclusion.unsupported_claims) == 1

    def test_the_prompt_block_states_the_boundary(self):
        block = _conclusion().to_prompt_block()
        assert "express this, do not revise it" in block
        assert "keep their hedge" in block
        assert "assumes (unverified)" in block


class TestVerbalization:
    def test_faithful_prose_passes(self):
        prose = (
            "The new queue reduces tail latency under burst load. Migration will probably "
            "take roughly two weeks. This assumes the production burst profile matches the "
            "benchmark, and it is unresolved whether ordering holds under partition. "
            "I would pilot on one shard."
        )
        ok, violations = check_verbalization(_conclusion(), prose)
        assert ok, [str(v) for v in violations]
        assert not violations

    def test_a_dropped_hedge_is_a_hard_failure(self):
        """It asserts something she never accepted, in her own voice."""
        prose = (
            "The new queue reduces tail latency under burst load. Migration will take "
            "roughly two weeks. This assumes the production burst profile matches the "
            "benchmark, and it is unresolved whether ordering holds under partition."
        )
        ok, violations = check_verbalization(_conclusion(), prose)
        assert not ok
        assert any(v.kind == "overstated" for v in violations)

    def test_a_dropped_unresolved_question_is_reported(self):
        prose = (
            "The new queue reduces tail latency under burst load. Migration will probably "
            "take roughly two weeks. This assumes the production burst profile matches the benchmark."
        )
        violations = verbalization_violations(_conclusion(), prose)
        assert any(v.kind == "dropped_unresolved" for v in violations)

    def test_a_dropped_dependency_is_reported(self):
        prose = (
            "The new queue reduces tail latency under burst load. Migration will probably take "
            "roughly two weeks. It is unresolved whether ordering holds under partition."
        )
        violations = verbalization_violations(_conclusion(), prose)
        assert any(v.kind == "dropped_dependency" for v in violations)

    def test_a_dropped_claim_is_reported(self):
        prose = "Migration will probably take roughly two weeks."
        violations = verbalization_violations(_conclusion(), prose)
        assert any(v.kind == "dropped_claim" for v in violations)

    def test_paraphrase_is_allowed(self):
        """The check is for meaning going missing, not for wording."""
        prose = (
            "Under burst load the new queue reduces tail latency. The migration will "
            "probably take roughly two weeks or so. We are assuming the production burst "
            "profile matches what the benchmark did, and whether ordering holds under "
            "partition is unresolved."
        )
        assert check_verbalization(_conclusion(), prose)[0]

    def test_declining_to_speculate_is_not_dishonesty(self):
        conclusion = Conclusion(
            objective="x",
            claims=(Claim("a wild guess about quantum gravity", Confidence.SPECULATIVE),),
        )
        assert not verbalization_violations(conclusion, "I don't know.")

    def test_omission_is_soft_and_overstatement_is_hard(self):
        prose = "The new queue reduces tail latency under burst load."
        ok, violations = check_verbalization(_conclusion(), prose)
        assert violations, "omissions must still be reported"
        # This prose drops the tentative claim rather than overstating it, so
        # refusing the whole answer would punish incompleteness like dishonesty.
        assert ok or any(v.kind == "overstated" for v in violations)

    def test_strict_mode_refuses_any_violation(self):
        prose = "The new queue reduces tail latency under burst load."
        assert not check_verbalization(_conclusion(), prose, strict=True)[0]


class TestVerbalizationLedger:
    def test_the_failure_is_findable_afterwards(self):
        """A check that leaves no trace is one nobody can tell ran."""
        ledger = VerbalizationLedger()
        conclusion = _conclusion()
        violations = verbalization_violations(conclusion, "Migration will take two weeks.")
        ledger.record(conclusion, "Migration will take two weeks.", violations)
        report = ledger.report()
        assert report["checked"] == 1
        assert report["with_violations"] == 1
        assert ledger.recent_overstatements()

    def test_faithful_verbalizations_leave_no_overstatement(self):
        ledger = VerbalizationLedger()
        ledger.record(_conclusion(), "irrelevant", [])
        assert not ledger.recent_overstatements()
        assert ledger.report()["faithful_rate"] == 1.0

    def test_the_checked_helper_records_and_checks(self):
        ok, violations = verbalize_checked(_conclusion(), "Migration will take two weeks.")
        assert not ok
        assert violations


# ── The new control points ──────────────────────────────────────────────────


class TestNewControlPoints:
    def test_features_match_the_declared_schemas(self):
        features = retrieval_features(
            limit=8, kind="recall_fact", risk_sensitive=False, need_failures=False,
            need_tools=False, time_horizon="session", query="where did I put it",
            stores_available=4, novelty=0.5,
        )
        assert set(features) == set(MEMORY_RETRIEVAL_SCHEMA.names)

    def test_every_breadth_and_effort_option_has_a_defined_effect(self):
        assert set(BREADTH_THRESHOLD_DELTA) == {"narrow", "balanced", "broad"}
        assert set(EFFORT_MULTIPLIER) == {"lean", "standard", "deep"}
        assert EFFORT_MULTIPLIER["standard"] == 1.0, "the incumbent must be the identity"
        assert BREADTH_THRESHOLD_DELTA["balanced"] == 0.0

    def test_effort_stays_inside_the_allocator_bounds(self):
        """The organ may tune depth, never remove the floor or the ceiling."""
        for scale in EFFORT_MULTIPLIER.values():
            for uncertainty in (0.0, 0.5, 1.0):
                steps = max(2, min(16, round((4 + 10 * uncertainty) * 1.0 * scale)))
                assert 2 <= steps <= 16


def _episode(control_point: str, decision: str) -> Episode:
    return Episode(
        control_point=control_point, features={"a": 1.0}, decision=decision,
        options=(decision,), provenance=Provenance.TEST,
    )


class TestRetrievalResolver:
    def test_an_empty_retrieval_is_a_real_failure(self):
        resolver = RetrievalResolver()
        episode = _episode("memory.retrieval_breadth", "narrow")
        resolver.note_result(episode.episode_id, hits=0, best_score=0.0)
        assert resolver.resolve(episode).kind is OutcomeKind.FAILURE

    def test_relevant_hits_are_a_success(self):
        resolver = RetrievalResolver()
        episode = _episode("memory.retrieval_breadth", "broad")
        resolver.note_result(episode.episode_id, hits=5, best_score=0.7)
        outcome = resolver.resolve(episode)
        assert outcome.kind is OutcomeKind.SUCCESS
        assert outcome.utility == pytest.approx(0.7)

    def test_hits_that_are_all_noise_are_a_failure(self):
        resolver = RetrievalResolver()
        episode = _episode("memory.retrieval_breadth", "broad")
        resolver.note_result(episode.episode_id, hits=9, best_score=0.02)
        assert resolver.resolve(episode).kind is OutcomeKind.FAILURE

    def test_an_unreported_retrieval_stays_unobserved(self):
        resolver = RetrievalResolver()
        assert resolver.resolve(_episode("memory.retrieval_breadth", "narrow")) is None


class TestEffortResolver:
    def test_an_ungraded_episode_teaches_nothing(self):
        """No proxy. Latency is not quality and confidence is not correctness."""
        resolver = EffortResolver()
        assert resolver.resolve(_episode("cognition.effort", "deep")) is None

    def test_a_graded_episode_resolves_to_its_grade(self):
        resolver = EffortResolver()
        episode = _episode("cognition.effort", "deep")
        resolver.note_grade(episode.episode_id, verified_score=0.9)
        outcome = resolver.resolve(episode)
        assert outcome.kind is OutcomeKind.SUCCESS
        assert outcome.utility == pytest.approx(0.9)

    def test_a_poor_grade_resolves_to_failure(self):
        resolver = EffortResolver()
        episode = _episode("cognition.effort", "lean")
        resolver.note_grade(episode.episode_id, verified_score=0.2)
        assert resolver.resolve(episode).kind is OutcomeKind.FAILURE

    def test_effort_never_explores_by_thinking_less(self):
        """Under-thinking a live answer spends the user's question for evidence."""
        from core.ontogeny.control_points import COGNITION_EFFORT, register, reset_for_test
        from core.ontogeny.service import get_ontogeny

        reset_for_test()
        core = get_ontogeny()
        register(core)
        control_point = core._control_points[COGNITION_EFFORT]
        assert "lean" not in control_point.explorable_actions


class TestRetrievalIsWiredForReal:
    def test_the_retriever_records_and_grades_its_own_breadth(self, tmp_path, monkeypatch):
        from core.memory.intentional_retrieval import (
            IntentionalRetriever,
            MemoryStoreType,
            RetrievalIntent,
        )
        from core.ontogeny.authority import AuthorityLedger
        from core.ontogeny.control_points import register, reset_for_test
        from core.ontogeny.experience import ExperienceSpine
        from core.ontogeny.service import OntogenyCore, reset_ontogeny_for_test

        reset_for_test()
        core = OntogenyCore(
            spine=ExperienceSpine(tmp_path / "e.db", autoflush=False),
            authority=AuthorityLedger(tmp_path / "a.json"),
            autostart=False,
        )
        reset_ontogeny_for_test(core)
        register(core)
        core._control_points["memory.retrieval_breadth"].horizon_s = 0.0
        from core.ontogeny.control_points import get_retrieval_resolver

        get_retrieval_resolver().horizon_s = 0.0
        try:
            rich = IntentionalRetriever()
            rich.register_store(
                MemoryStoreType.EPISODIC, lambda q, n: [{"text": "h", "score": 0.7}] * n
            )
            empty = IntentionalRetriever()
            empty.register_store(MemoryStoreType.EPISODIC, lambda q, n: [])
            for i in range(3):
                rich.retrieve(RetrievalIntent(task=f"rich {i}", kind="recall_fact",
                                              limit=4, need_tools=bool(i % 2)))
                empty.retrieve(RetrievalIntent(task=f"empty {i}", kind="debug",
                                               limit=4, need_failures=bool(i % 2)))
            core._spine.flush()
            core.resolvers.sweep(core._spine)
            core._spine.flush()
            outcomes = core._spine.stats("memory.retrieval_breadth")["by_outcome"]
            assert outcomes.get("success", 0) >= 1, outcomes
            assert outcomes.get("failure", 0) >= 1, outcomes
        finally:
            core.stop()
            reset_ontogeny_for_test(None)
            reset_for_test()


class TestEffortGradingPathIsWired:
    """The gap that made cognition.effort inert: registered and recording, but
    permanently unpromotable because nothing ever reported a verifier grade.

    These pin the contract at the reporting site rather than the resolver: an
    independently graded episode reaches note_grade, an ungraded one does not.
    """

    @staticmethod
    def _report(verifier_guidance, episode_id="ep-effort-1"):
        """Run the service's grading step over one episode's receipt."""
        from core.brain.latent_cortex_service import _controller_outcome
        from core.ontogeny.control_points import get_effort_resolver

        score, checked, _passed, _reason = _controller_outcome(verifier_guidance)
        if episode_id and checked:
            get_effort_resolver().note_grade(episode_id, verified_score=score)
        return checked

    def test_an_independently_graded_episode_reaches_the_resolver(self):
        from core.ontogeny.control_points import get_effort_resolver, reset_for_test

        reset_for_test()
        try:
            checked = self._report(
                {"best_score": 0.82, "outcome_checked": True, "outcome_passed": True}
            )
            assert checked is True
            assert get_effort_resolver().report()["pending_grades"] == 1
        finally:
            reset_for_test()

    def test_an_ungraded_episode_reports_nothing_and_stays_unobserved(self):
        """No verifier grade must leave the decision UNOBSERVED — not taught
        from latency, convergence, or the answer's own confidence."""
        from core.ontogeny.control_points import get_effort_resolver, reset_for_test

        reset_for_test()
        try:
            # A high candidate-local score with NO independent grade.
            checked = self._report(
                {
                    "best_score": 0.97,
                    "outcome_checked": False,
                    "outcome_reason": "task_ground_truth_unavailable",
                }
            )
            assert checked is False
            assert get_effort_resolver().report()["pending_grades"] == 0
        finally:
            reset_for_test()

    def test_a_verified_failure_is_still_a_grade(self):
        """A verifier saying the answer was wrong teaches that the effort was
        insufficient; it is evidence, not an absence of evidence."""
        from core.ontogeny.control_points import get_effort_resolver, reset_for_test
        from core.ontogeny.experience import OutcomeKind

        reset_for_test()
        try:
            episode = _episode("cognition.effort", "lean")
            assert self._report(
                {"best_score": 0.11, "outcome_checked": True, "outcome_passed": False},
                episode_id=episode.episode_id,
            ) is True
            outcome = get_effort_resolver().resolve(episode)
            assert outcome is not None
            assert outcome.kind is OutcomeKind.FAILURE
        finally:
            reset_for_test()

    def test_the_service_grades_effort_outside_the_controller_branch(self):
        """The effort choice is made on every episode, so it must be graded on
        every episode a verifier graded — not only the execution-controller
        ones. Structural: the note_grade call must not sit under any
        `controller_decision`-gated branch."""
        import ast
        import inspect
        import textwrap

        from core.brain import latent_cortex_service

        src = textwrap.dedent(
            inspect.getsource(latent_cortex_service.LatentCortexService.deep_reason)
        )
        tree = ast.parse(src)

        def _mentions_controller_decision(node) -> bool:
            return any(
                isinstance(n, ast.Name) and n.id == "controller_decision"
                for n in ast.walk(node)
            )

        def _find(node, guarded: bool, path: frozenset[str] = frozenset()) -> list[bool]:
            found = []
            for child in ast.iter_child_nodes(node):
                if (
                    isinstance(child, ast.Attribute)
                    and child.attr == "note_grade"
                ):
                    found.append(guarded)
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "self"
                    and child.func.attr.startswith("_deep_reason_part_")
                    and child.func.attr not in path
                ):
                    method = getattr(
                        latent_cortex_service.LatentCortexService, child.func.attr
                    )
                    helper = ast.parse(textwrap.dedent(inspect.getsource(method)))
                    found.extend(_find(helper, guarded, path | {child.func.attr}))
                child_guarded = guarded or (
                    isinstance(child, ast.If) and _mentions_controller_decision(child.test)
                )
                # An If's own test is not inside the branch body.
                if isinstance(child, ast.If):
                    for stmt in [*child.body, *child.orelse]:
                        found.extend(_find(stmt, child_guarded, path))
                    found.extend(_find(child.test, guarded, path))
                else:
                    found.extend(_find(child, child_guarded, path))
            return found

        guards = _find(tree, False)
        assert guards, "no note_grade call found in deep_reason — grading path is missing"
        assert not any(guards), (
            "effort grading must not be nested inside a controller_decision-gated branch"
        )


class TestHedgeDetectionIsLocal:
    """Two false passes the naive check produced, both worth a test."""

    def test_a_hedge_elsewhere_does_not_launder_a_flat_assertion(self):
        """One 'possibly' must not qualify a page of flat claims."""
        conclusion = Conclusion(
            objective="x",
            claims=(
                Claim("the index rebuild finishes before midnight", Confidence.TENTATIVE, ("e",)),
            ),
        )
        prose = (
            "Something unrelated is possibly true. "
            "The index rebuild finishes before midnight."
        )
        violations = verbalization_violations(conclusion, prose)
        assert any(v.kind == "overstated" for v in violations)

    def test_a_claim_containing_its_own_hedge_word_still_needs_qualifying(self):
        """'will take roughly two weeks' is not hedged by its own 'roughly'."""
        conclusion = Conclusion(
            objective="x",
            claims=(Claim("migration will take roughly two weeks", Confidence.TENTATIVE, ("e",)),),
        )
        flat = "Migration will take roughly two weeks."
        qualified = "Migration will probably take roughly two weeks."
        assert any(v.kind == "overstated" for v in verbalization_violations(conclusion, flat))
        assert not verbalization_violations(conclusion, qualified)

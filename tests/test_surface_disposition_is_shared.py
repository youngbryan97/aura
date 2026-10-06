"""Three gates, one policy.

The MLX worker, the inference gate and the response-generation phase can each
end a turn on their own. Each had grown its own list of what counts as
"unsafe", so teaching two of them that a shortfall is repairable still lost the
turn at the third — measured live 2026-07-26, in that exact order:

    Cortex produced repairable user-facing draft (final_answer_missing,
    len=735). Passing it to downstream chat repair…
    ResponseGeneration rejected unsafe user-facing draft (final_answer_missing,
    len=735).
"""

import pytest

from core.conversation.surface_disposition import (
    SHORTFALL_REASONS,
    SurfaceDisposition,
    disposition_for,
    draft_is_servable,
    integrity_failures,
)

pytestmark = pytest.mark.unit


class TestTheTwoKindsOfFailure:
    def test_nothing_wrong_is_served(self):
        assert disposition_for(()) is SurfaceDisposition.SERVE

    def test_a_shortfall_is_repaired_not_discarded(self):
        for reason in ("final_answer_missing", "truncated_tail", "incomplete_code_response"):
            assert disposition_for((reason,)) is SurfaceDisposition.REPAIR, reason
            assert draft_is_servable((reason,)), reason

    def test_text_that_must_not_be_spoken_is_discarded(self):
        for reason in (
            "raw_lane_telemetry",
            "prompt_artifact",
            "corrupted_language",
            "function_word_starvation",
            "cognitive_engine_failure_envelope",
        ):
            assert disposition_for((reason,)) is SurfaceDisposition.DISCARD, reason
            assert not draft_is_servable((reason,)), reason

    def test_one_integrity_failure_outweighs_any_number_of_shortfalls(self):
        mixed = ("final_answer_missing", "truncated_tail", "raw_lane_telemetry")
        assert disposition_for(mixed) is SurfaceDisposition.DISCARD
        assert integrity_failures(mixed) == ("raw_lane_telemetry",)

    def test_thinness_asks_for_repair_rather_than_destroying_the_turn(self):
        """Thinness is an estimate that quality is absent, not an
        identification of something unspeakable. It may ask for another
        generation; it may not be the reason a person receives nothing."""
        for reason in (
            "too_thin_for_user_turn",
            "too_thin_for_operational_status_turn",
            "reliability_diagnostic_too_thin",
        ):
            assert disposition_for((reason,)) is SurfaceDisposition.REPAIR, reason

    def test_a_new_heuristic_cannot_silently_destroy_answers(self):
        """The default runs toward the person, deliberately.

        A detector written next week can ask for repair. To gain the power to
        withhold an answer it has to be added to UNSPEAKABLE_REASONS on
        purpose, with evidence it IDENTIFIES rather than estimates — which is
        the property that makes the 2026-07-26 ratchet impossible to repeat.
        """
        assert disposition_for(("a_reason_added_next_week",)) is SurfaceDisposition.REPAIR
        assert draft_is_servable(("a_reason_added_next_week",))


class TestEveryGateConsultsIt:
    """A new shortfall reason must not need three separate edits."""

    def test_the_worker_vetoes_on_integrity_only(self):
        import inspect

        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker._surface_quality_failure_reasons)
        assert "integrity_failures" in source

    def test_the_inference_gate_passes_shortfalls_downstream(self):
        import inspect

        from core.brain import inference_gate

        source = inspect.getsource(
            inference_gate._should_pass_user_facing_draft_downstream
        )
        assert "draft_is_servable" in source

    def test_response_generation_keeps_a_servable_draft(self):
        from pathlib import Path

        # Wherever the rejection lives: a size refactor moved it from
        # response_generation.py into response_generation_drafts.py.
        phases = Path(__file__).resolve().parents[1] / "core" / "phases"
        sources = [
            path.read_text(encoding="utf-8")
            for path in sorted(phases.glob("response_generation*.py"))
            if "rejected unsafe user-facing draft" in path.read_text(encoding="utf-8")
        ]
        assert len(sources) == 1, "the rejection should live in one module"
        rejection = sources[0].split("rejected unsafe user-facing draft", 1)[0][-2000:]
        assert "draft_is_servable" in rejection


class TestTheReasonsThatMatterAreClassified:
    def test_the_live_shortfalls_are_all_repairable(self):
        """Every reason a correct answer was actually killed for."""
        for reason in (
            "final_answer_missing",
            "truncated_tail",
            "incomplete_code_response",
        ):
            assert reason in SHORTFALL_REASONS, reason


class TestTheVanillaFloor:
    """Aura must never answer worse than the bare model she is built around.

    A pipeline of gates that can only subtract has an output quality of min()
    over all of them, so this cannot hold by accident — the raw draft is kept
    and is the floor nothing falls below.
    """

    def test_the_raw_model_answer_is_the_fallback(self):
        from core.conversation.surface_disposition import (
            best_available_reply,
            clear_preserved_draft,
            record_raw_model_draft,
        )

        clear_preserved_draft()
        assert best_available_reply() == ""
        record_raw_model_draft(
            "Total marbles: 3 + 4 + 5 = 12, and the same-colour pairs number "
            "nineteen out of sixty-six possible pairs."
        )
        assert best_available_reply()

    def test_a_deliberately_preserved_draft_outranks_the_raw_one(self):
        from core.conversation.surface_disposition import (
            best_available_reply,
            clear_preserved_draft,
            preserve_draft,
            record_raw_model_draft,
        )

        clear_preserved_draft()
        record_raw_model_draft(
            "A rougher answer that still says something real about the problem "
            "and its three separate cases."
        )
        preserve_draft(
            "The repaired answer, which a layer kept on purpose because it is "
            "the better of the two available here."
        )
        assert best_available_reply().startswith("The repaired answer")

    def test_the_floor_never_becomes_a_leak(self):
        from core.conversation.surface_disposition import (
            best_available_reply,
            clear_preserved_draft,
            record_raw_model_draft,
        )

        clear_preserved_draft()
        record_raw_model_draft(
            "ROUTER_ERROR: unknown (at all_failed) with plenty of extra words "
            "here so the length floor is not what refuses it."
        )
        assert best_available_reply() == ""

    def test_a_fragment_is_not_worth_serving(self):
        from core.conversation.surface_disposition import (
            best_available_reply,
            clear_preserved_draft,
            record_raw_model_draft,
        )

        clear_preserved_draft()
        record_raw_model_draft("Both red.")
        assert best_available_reply() == ""

    def test_short_continuation_cannot_erase_the_raw_incumbent(self):
        from core.conversation.surface_disposition import (
            clear_preserved_draft,
            raw_model_draft,
            record_raw_model_draft,
        )

        clear_preserved_draft()
        incumbent = (
            "The complete incumbent has several independently authored sections "
            "and remains the canonical answer floor for this turn."
        )
        record_raw_model_draft(incumbent)
        record_raw_model_draft("A short child continuation.")

        assert raw_model_draft() == incumbent

    def test_short_continuation_cannot_erase_the_preserved_draft(self):
        from core.conversation.surface_disposition import (
            clear_preserved_draft,
            preserve_draft,
            preserved_draft,
        )

        clear_preserved_draft()
        incumbent = (
            "The preserved draft contains the earlier authored work and all of "
            "the progress that a later continuation must extend."
        )
        preserve_draft(incumbent)
        preserve_draft("A short child continuation.")

        assert preserved_draft() == incumbent

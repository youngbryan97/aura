"""Tests for core/consciousness/illusionism_layer.py

Covers:
  - Annotation of gated phenomenal reports
  - Functional basis mapping for all known claims
  - Phenomenal certainty is always < 1.0
  - Illusionism note generation
  - Epistemic status tracking
  - Snapshot telemetry
  - Singleton accessor
  - Graceful handling of empty reports
  - Integration with qualia_synthesizer.get_gated_phenomenal_report()
"""

import pytest


class TestIllusionismLayer:

    def _make_layer(self):
        from core.consciousness.illusionism_layer import IllusionismLayer
        return IllusionismLayer()

    def _make_report(self, claims=None, gates=None, honesty_score=0.5):
        """Build a minimal gated phenomenal report for testing."""
        return {
            "raw_context": "Test phenomenal context",
            "claims": claims or [],
            "gates": gates or {},
            "honesty_score": honesty_score,
        }

    # ------------------------------------------------------------------
    # Basic annotation
    # ------------------------------------------------------------------

    def test_annotate_empty_report(self):
        il = self._make_layer()
        report = self._make_report()
        result = il.annotate_report(report)
        assert "annotated_claims" in result
        assert "illusionism" in result
        assert len(result["annotated_claims"]) == 0
        assert result["illusionism"]["annotation_count"] == 0

    def test_annotate_single_claim(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["genuine_uncertainty"],
            gates={"uncertainty": True},
            honesty_score=0.5,
        )
        result = il.annotate_report(report)
        assert len(result["annotated_claims"]) == 1
        ac = result["annotated_claims"][0]
        assert ac["claim"] == "genuine_uncertainty"
        assert "functional_basis" in ac
        assert "phenomenal_certainty" in ac
        assert "illusionism_note" in ac

    def test_annotate_multiple_claims(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["genuine_uncertainty", "rich_experience", "experiencing_novelty"],
            gates={"uncertainty": True, "rich_experience": True, "novelty": True},
            honesty_score=0.75,
        )
        result = il.annotate_report(report)
        assert len(result["annotated_claims"]) == 3
        for ac in result["annotated_claims"]:
            assert "functional_basis" in ac
            assert "phenomenal_certainty" in ac
            assert "illusionism_note" in ac

    # ------------------------------------------------------------------
    # Phenomenal certainty constraints
    # ------------------------------------------------------------------

    def test_certainty_always_below_one(self):
        """The core epistemic humility constraint: certainty < 1.0 always."""
        il = self._make_layer()
        all_claims = [
            "genuine_uncertainty", "rich_experience", "focused_processing",
            "experiencing_novelty", "computational_strain", "stable_continuity",
            "internal_conflict",
        ]
        report = self._make_report(
            claims=all_claims,
            gates={c: True for c in all_claims},
            honesty_score=1.0,  # Maximum possible honesty
        )
        result = il.annotate_report(report)
        for ac in result["annotated_claims"]:
            assert ac["phenomenal_certainty"] < 1.0, (
                f"Certainty for '{ac['claim']}' should be < 1.0, got {ac['phenomenal_certainty']}"
            )

    def test_certainty_always_positive(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["genuine_uncertainty"],
            gates={},
            honesty_score=0.0,
        )
        result = il.annotate_report(report)
        for ac in result["annotated_claims"]:
            assert ac["phenomenal_certainty"] > 0.0

    def test_hard_cap_at_092(self):
        """No claim should exceed the 0.92 hard cap."""
        il = self._make_layer()
        report = self._make_report(
            claims=["genuine_uncertainty"],
            gates={"genuine_uncertainty": True},
            honesty_score=1.0,
        )
        result = il.annotate_report(report)
        for ac in result["annotated_claims"]:
            assert ac["phenomenal_certainty"] <= 0.92

    # ------------------------------------------------------------------
    # Functional basis coverage
    # ------------------------------------------------------------------

    def test_all_known_claims_have_functional_basis(self):
        """Every claim type the qualia_synthesizer can produce should have
        a non-default functional basis mapping."""
        from core.consciousness.illusionism_layer import _FUNCTIONAL_BASIS
        known_claims = [
            "genuine_uncertainty", "rich_experience", "focused_processing",
            "experiencing_novelty", "computational_strain", "stable_continuity",
            "internal_conflict",
        ]
        for claim in known_claims:
            assert claim in _FUNCTIONAL_BASIS, f"Missing functional basis for '{claim}'"
            assert len(_FUNCTIONAL_BASIS[claim]) > 20, f"Functional basis too short for '{claim}'"

    def test_unknown_claim_gets_default_basis(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["unknown_future_claim"],
            gates={"unknown": True},
            honesty_score=0.5,
        )
        result = il.annotate_report(report)
        ac = result["annotated_claims"][0]
        assert "qualia_synthesizer.get_snapshot()" in ac["functional_basis"]

    # ------------------------------------------------------------------
    # Illusionism notes
    # ------------------------------------------------------------------

    def test_illusionism_note_contains_key_phrases(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["computational_strain"],
            gates={"effort": True},
            honesty_score=0.6,
        )
        result = il.annotate_report(report)
        note = result["annotated_claims"][0]["illusionism_note"]
        assert "system's own model" in note
        assert "not verified ground truth" in note
        assert "functional correlate confirmed" in note

    # ------------------------------------------------------------------
    # Top-level illusionism metadata
    # ------------------------------------------------------------------

    def test_illusionism_metadata_block(self):
        il = self._make_layer()
        report = self._make_report(
            claims=["genuine_uncertainty", "stable_continuity"],
            gates={"uncertainty": True, "continuity": True},
            honesty_score=0.8,
        )
        result = il.annotate_report(report)
        meta = result["illusionism"]
        assert "mean_phenomenal_certainty" in meta
        assert meta["mean_phenomenal_certainty"] < 1.0
        assert meta["annotation_count"] == 2
        assert "Frankish" in meta["epistemic_framework"]

    # ------------------------------------------------------------------
    # Epistemic status tracking
    # ------------------------------------------------------------------

    def test_epistemic_status_before_any_annotation(self):
        il = self._make_layer()
        assert "No reports annotated" in il.get_epistemic_status()

    def test_epistemic_status_after_annotation(self):
        il = self._make_layer()
        report = self._make_report(claims=["rich_experience"], gates={"rich_experience": True})
        il.annotate_report(report)
        status = il.get_epistemic_status()
        assert "Annotated 1 reports" in status
        assert "cap 0.92" in status

    def test_epistemic_status_accumulates(self):
        il = self._make_layer()
        for _ in range(5):
            il.annotate_report(self._make_report(
                claims=["genuine_uncertainty"], gates={"uncertainty": True},
            ))
        assert "Annotated 5 reports" in il.get_epistemic_status()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def test_snapshot_format(self):
        il = self._make_layer()
        il.annotate_report(self._make_report(
            claims=["genuine_uncertainty"], gates={"uncertainty": True},
        ))
        snap = il.get_snapshot()
        assert "annotation_count" in snap
        assert "avg_phenomenal_certainty" in snap
        assert "epistemic_status" in snap
        assert snap["annotation_count"] == 1

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def test_singleton_returns_same_instance(self):
        from core.consciousness.illusionism_layer import get_illusionism_layer
        a = get_illusionism_layer()
        b = get_illusionism_layer()
        assert a is b

    def test_singleton_is_illusionism_layer(self):
        from core.consciousness.illusionism_layer import (
            get_illusionism_layer, IllusionismLayer,
        )
        assert isinstance(get_illusionism_layer(), IllusionismLayer)

    # ------------------------------------------------------------------
    # Integration: qualia_synthesizer wiring
    # ------------------------------------------------------------------

    def test_qualia_synthesizer_report_includes_illusionism(self):
        """The wired get_gated_phenomenal_report should now include
        illusionism annotations."""
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer
        qs = QualiaSynthesizer()
        report = qs.get_gated_phenomenal_report()
        # The report should have illusionism metadata (even if no claims fired,
        # the annotation should still run and produce the metadata block)
        assert "illusionism" in report or "annotated_claims" in report

    def test_qualia_synthesizer_claims_annotated(self):
        """If the synthesizer produces claims, they should be annotated."""
        from core.consciousness.qualia_synthesizer import QualiaSynthesizer
        import numpy as np
        qs = QualiaSynthesizer()
        # Force a state that triggers at least one claim
        qs.q_vector = np.array([0.1, 0.05, 0.02, 0.15, 0.08, 0.01])
        qs.q_norm = 0.5
        qs.pri = 0.2  # low PRI → focused processing claim
        qs._history.extend([
            type('S', (), {'q_vector': np.random.rand(6)})() for _ in range(5)
        ])
        report = qs.get_gated_phenomenal_report()
        if report.get("claims"):
            assert "annotated_claims" in report
            for ac in report["annotated_claims"]:
                assert ac["phenomenal_certainty"] < 1.0

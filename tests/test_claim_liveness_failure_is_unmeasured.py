"""A failed reader cannot restore the live claim it could not check."""
from __future__ import annotations

import pytest

from core.organism import claim_liveness
from core.organism.model_validation import Claim, Evidence


@pytest.mark.parametrize("exception", [ImportError, AttributeError, RuntimeError, TypeError, ValueError,
                                      OSError, KeyError, ZeroDivisionError])
def test_failed_liveness_reader_demotes_live_evidence(monkeypatch, exception):
    def fail(*args):
        raise exception("reader failed")

    monkeypatch.setattr(claim_liveness, "effective_evidence", fail)
    claim = Claim("A bounded output is delivered.", "delivery", "tests", live_channels=("output",))
    resolved, note = claim.effective_evidence()
    assert resolved is Evidence.UNMEASURED and exception.__name__ in note
    assert not claim.is_evidence_for_the_system
    assert not claim.to_dict()["citable_as_evidence"]


@pytest.mark.parametrize("result", [("invented", "", []), (Evidence.MEASURED_LIVE, None, [])])
def test_malformed_liveness_result_cannot_be_cited(monkeypatch, result):
    monkeypatch.setattr(claim_liveness, "effective_evidence", lambda *args: result)
    claim = Claim("A bounded output is delivered.", "delivery", "tests", live_channels=("output",))
    assert claim.effective_evidence()[0] is Evidence.UNMEASURED


@pytest.mark.parametrize("evidence", [Evidence.RETRACTED, Evidence.FUNCTIONAL_ONLY, Evidence.MEASURED_SYNTHETIC])
def test_weaker_evidence_keeps_its_meaning_when_live_reader_is_unavailable(monkeypatch, evidence):
    def fail(*args):
        raise AssertionError("a non-live claim does not need a live reader")

    monkeypatch.setattr(claim_liveness, "effective_evidence", fail)
    claim = Claim("A bounded output is delivered.", "delivery", "tests", evidence=evidence,
                  evidence_note="Scoped contract.", live_channels=("output",))
    assert claim.effective_evidence() == (evidence, "")

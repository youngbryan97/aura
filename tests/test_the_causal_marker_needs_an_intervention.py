"""A live snapshot saying a causal channel is present is not causal evidence.

An external review named the distinction exactly:

    P(Y | do(X = x1)) != P(Y | do(X = x0))

is not established by a loop state containing a phi value. `from_live` set
`markers_causal` two ways, and both were presence rather than difference:
`bool(loop.get("phi") is not None)`, and `causal = True` with a comment beside
it saying the felt state gates action via the Will.

Aura's own influence framework already requires treatment against null, so
that is where the answer comes from now — and where there is no such
evidence, the answer is False with a sentence saying what would change it,
which is a different claim from "epiphenomenal" and reads differently.
"""
from __future__ import annotations


from core.consciousness.phenomenal_falsification import (
    THE_MARKER_CHANNELS,
    PhenomenalFalsifier,
    _causal_by_intervention,
)


# ------------------------------------------------------- the proxies are gone


def test_a_present_phi_no_longer_makes_the_markers_causal():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "consciousness" / "phenomenal_falsification.py"
    ).read_text("utf-8")
    assert 'causal = bool(loop.get("phi") is not None)' not in source
    assert "causal = True  # the felt-state gates action via the Will" not in source


def test_the_marker_channels_are_named_rather_than_matched_by_word():
    """A channel merely containing the word "phi" is not evidence."""
    assert "phi" in THE_MARKER_CHANNELS
    assert all(isinstance(one, str) and one for one in THE_MARKER_CHANNELS)


# -------------------------------------------------------- what it now reads


def test_with_no_trials_the_answer_is_false_and_says_what_would_change_it():
    causal, why = _causal_by_intervention()
    assert causal is False
    assert "no paired trials" in why
    assert "treatment and the null" in why


def test_no_trials_reads_as_unmeasured_and_not_as_epiphenomenal():
    """Calling it epiphenomenal on no evidence is as wrong as calling it causal."""
    verdict = PhenomenalFalsifier().assess_live().verdict
    assert verdict.startswith("UNMEASURED")
    assert "epiphenomenal" not in verdict


def test_a_measured_influential_channel_makes_the_markers_causal(monkeypatch):
    from core.verify.causal_influence import Verdict

    class ALedger:
        def verdict(self, channel):
            class One:
                pass

            one = One()
            one.verdict = (
                Verdict.INFLUENTIAL if channel == "phi" else Verdict.UNMEASURED
            )
            return one

    import core.verify.causal_influence as influence

    monkeypatch.setattr(influence, "get_influence_ledger", lambda: ALedger())
    causal, why = _causal_by_intervention()
    assert causal is True
    assert "lesioning moved the output" in why
    assert "phi" in why


def test_a_channel_measured_inert_is_not_treated_as_unmeasured(monkeypatch):
    """The faculty ran and did not change the output. That is a real finding."""
    from core.verify.causal_influence import Verdict

    class ALedger:
        def verdict(self, channel):
            class One:
                pass

            one = One()
            one.verdict = Verdict.INERT
            return one

    import core.verify.causal_influence as influence

    monkeypatch.setattr(influence, "get_influence_ledger", lambda: ALedger())
    causal, why = _causal_by_intervention()
    assert causal is False
    assert "enough power to see an effect" in why


def test_an_inert_result_does_read_as_epiphenomenal(monkeypatch):
    from core.verify.causal_influence import Verdict

    class ALedger:
        def verdict(self, channel):
            class One:
                pass

            one = One()
            one.verdict = Verdict.INERT
            return one

    import core.verify.causal_influence as influence

    monkeypatch.setattr(influence, "get_influence_ledger", lambda: ALedger())
    falsifier = PhenomenalFalsifier()
    verdict = falsifier.assess_live().verdict
    assert "epiphenomenal" in verdict


def test_a_ledger_that_cannot_be_reached_says_so_rather_than_guessing(monkeypatch):
    import core.verify.causal_influence as influence

    def angry():
        raise RuntimeError("no ledger here")

    monkeypatch.setattr(influence, "get_influence_ledger", angry)
    causal, why = _causal_by_intervention()
    assert causal is False
    assert "no influence ledger" in why or "no paired trials" in why


def test_the_falsifier_remembers_what_it_based_the_answer_on():
    falsifier = PhenomenalFalsifier()
    assert falsifier.why_the_causal_answer() == "not read yet"
    falsifier.from_live()
    assert "paired trials" in falsifier.why_the_causal_answer()

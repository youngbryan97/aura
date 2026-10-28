"""A clean completion needs recorded source and boundary identity."""
from core.subject import provenance


def test_the_pin_records_source_seed_and_the_whole_carrier_boundary(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *args: "a" * 40)
    monkeypatch.setattr(provenance, "_run_git", lambda *args: "")
    first = provenance.evidence_source(83)
    assert first["seed"] == 83 and first["commit"] == "a" * 40
    assert first["dirty"] is False and len(first["boundary_sha256"]) == 64
    import core.subject.carrier_reads as reads
    monkeypatch.setattr(reads, "ORGANS_WHOLE", not reads.ORGANS_WHOLE)
    second = provenance.evidence_source(83)
    assert second["boundary_sha256"] != first["boundary_sha256"]


def test_an_unavailable_git_read_is_unknown_cleanliness(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *args: "")
    monkeypatch.setattr(provenance, "_run_git", lambda *args: None)
    pin = provenance.evidence_source(83)
    assert pin["dirty"] is None and pin["commit"] == ""


def test_untracked_source_makes_the_pin_dirty(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *args: "a" * 40)
    monkeypatch.setattr(provenance, "_run_git", lambda *args: "?? core/uncommitted.py\n")
    assert provenance.evidence_source(83)["dirty"] is True

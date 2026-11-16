"""Notes on a person, across a restart.

A relationship model that resets on every restart is a session cache with
opinions. But durability is also where this subsystem's own failure can
reappear wearing a different hat, which is what these guard:

* **A serialiser that drops a field** is qualifier-loss by another route. The
  ratchet tests below derive the expected keys from the dataclasses, so a new
  field that nobody persisted fails here rather than being discovered as an
  unexplained gap in her memory months later.
* **Restoring by replaying ``observe``** would re-stamp every occurrence with
  the time of the restart — a year of evidence collapsing into a burst of
  sightings on boot day, which is the manufactured-frequency bug again — and
  would silently drop every correction, since ``observe`` refuses a corrected
  claim.
* **An unreadable file becoming an empty model** is how "she forgot everything
  about me" happens quietly. It is an incident with the evidence kept.
"""
from __future__ import annotations

import dataclasses
import json
import time

import pytest

from core.memory.interpersonal_model import (
    DAY_SECONDS,
    Facet,
    Observation,
    Occurrence,
    PersonModel,
    Provenance,
    Subject,
    Valence,
)
from core.memory.interpersonal_store import InterpersonalStore

pytestmark = pytest.mark.unit


class _Consent:
    """A stand-in for the relational memory authority."""

    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.asked: list[tuple[str, str, str]] = []

    def allows(self, agent_id: str, kind: str, operation: str) -> bool:
        self.asked.append((agent_id, kind, operation))
        return self.allow


@pytest.fixture
def store(tmp_path):
    return InterpersonalStore(root=tmp_path / "interpersonal", authority=_Consent())


def _populated() -> PersonModel:
    """One of every awkward thing: a correction, a counter-example, a repaired
    rupture, a state with a custom ttl, and evidence of three different ages."""
    now = time.time()
    model = PersonModel("Bryan", max_observations=64)
    model.observe(
        "I prefer terse answers",
        episode_id="ep1",
        facet=Facet.PREFERENCE,
        provenance=Provenance.STATED,
        conditions="when a build is failing",
        note="he said so during the outage",
        at=now - 90 * DAY_SECONDS,
    )
    model.observe("I prefer terse answers", episode_id="ep7",
                  facet=Facet.PREFERENCE, conditions="when a build is failing",
                  at=now - 30 * DAY_SECONDS)
    model.contradict("I prefer terse answers", episode_id="ep31",
                     facet=Facet.PREFERENCE, conditions="when a build is failing",
                     note="asked for the long version")
    model.observe("tired", episode_id="ep8", facet=Facet.STATE, ttl=2 * DAY_SECONDS)
    model.observe("shipped the release together", episode_id="ep9",
                  facet=Facet.EXPERIENCE, subject=Subject.BETWEEN,
                  valence=Valence.WARM)
    model.observe("I called his work sloppy", episode_id="ep10", facet=Facet.RUPTURE)
    model.resolve("I called his work sloppy", episode_id="ep11", facet=Facet.RUPTURE)
    model.observe("dislikes mornings", episode_id="ep12")
    model.correct("dislikes mornings", episode_id="ep13", note="he says that is wrong")
    return model


# ── the schema cannot quietly lose a field ─────────────────────────────────


def test_every_occurrence_field_is_persisted():
    declared = {f.name for f in dataclasses.fields(Occurrence)}
    persisted = set(Occurrence(episode_id="ep1").to_dict())

    assert declared == persisted


def test_every_observation_field_is_persisted():
    """The ratchet. Add a field to Observation without writing it here and this
    fails, rather than the field going missing at the next restart."""
    declared = {f.name for f in dataclasses.fields(Observation)}
    persisted = set(Observation(claim="x").to_dict())

    assert declared == persisted


def test_a_round_trip_changes_no_field_of_any_record():
    model = _populated()

    restored = PersonModel.from_dict(json.loads(json.dumps(model.to_dict())))

    before = {o.key: o for o in model}
    after = {o.key: o for o in restored}
    assert before.keys() == after.keys()
    for key, original in before.items():
        for f in dataclasses.fields(Observation):
            assert getattr(after[key], f.name) == getattr(original, f.name), (
                f"{f.name} did not survive the round trip for {key}"
            )


# ── restoring is not re-observing ──────────────────────────────────────────


def test_reloading_does_not_restamp_evidence_as_new():
    """A year of evidence must not become a burst of sightings on boot day."""
    model = _populated()
    original = model.current(facet=Facet.PREFERENCE)[0]

    restored = PersonModel.from_dict(model.to_dict())
    reloaded = restored.current(facet=Facet.PREFERENCE)[0]

    assert reloaded.first_seen() == original.first_seen()
    assert reloaded.last_seen() == original.last_seen()
    assert time.time() - reloaded.first_seen() > 89 * DAY_SECONDS


def test_a_correction_survives_a_restart():
    """``observe`` refuses a corrected claim, so a replay-based reload would
    drop exactly the thing he took the trouble to tell her."""
    model = _populated()

    restored = PersonModel.from_dict(model.to_dict())

    assert "dislikes mornings" not in restored.render()
    corrected = [o for o in restored if o.claim == "dislikes mornings"]
    assert len(corrected) == 1 and corrected[0].corrected


def test_counter_examples_survive_a_restart():
    """A record that can only accumulate confirmations is a grudge, and a
    serialiser is a perfectly good way to build one by accident."""
    model = _populated()

    restored = PersonModel.from_dict(model.to_dict())
    observation = restored.current(facet=Facet.PREFERENCE)[0]

    assert observation.contradictions == 1
    assert "it did not hold" in observation.render()


def test_a_conditional_does_not_lose_its_condition(tmp_path):
    model = _populated()

    restored = PersonModel.from_dict(model.to_dict())

    assert "when a build is failing" in restored.render()


def test_a_repaired_rupture_stays_repaired():
    model = _populated()

    restored = PersonModel.from_dict(model.to_dict())

    assert restored.unrepaired() == []


def test_a_state_keeps_its_own_expiry():
    model = _populated()

    restored = PersonModel.from_dict(model.to_dict())
    state = [o for o in restored if o.facet is Facet.STATE][0]

    assert state.ttl == 2 * DAY_SECONDS


def test_replaying_the_same_episode_does_not_manufacture_frequency():
    model = PersonModel("Bryan")

    model.observe("prefers precise answers", episode_id="ep-1")
    model.observe("prefers precise answers", episode_id="ep-1")

    observation = next(iter(model))
    assert [item.episode_id for item in observation.occurrences] == ["ep-1"]


def test_removing_a_superseded_episode_removes_every_evidence_edge():
    model = PersonModel("Bryan")
    model.observe("prefers precise answers", episode_id="ep-old")
    model.contradict("prefers precise answers", episode_id="ep-old")
    model.resolve(
        "prefers precise answers",
        episode_id="ep-old",
        facet=Facet.TRAIT,
    )

    assert model.remove_episode("ep-old") == 3
    assert len(model) == 0


# ── the store on disk ──────────────────────────────────────────────────────


async def test_what_she_noticed_is_there_after_a_restart(tmp_path):
    consent = _Consent()
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)
    await store.observe_turn(
        "Bryan",
        episode_id="ep1",
        user_text="I don't like long meetings when I'm already behind",
    )

    restarted = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)

    assert "when I'm already behind" in restarted.render("Bryan")


async def test_person_notes_renew_their_grant_after_directory_setup(store, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    model = store.model_for("Bryan")
    model.observe("likes sea anemones", episode_id="ep-slow-directory")
    gateway = get_file_write_gateway()
    ensure_directory = gateway.ensure_directory_async
    grants = []
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "memory_write"
        result = await ensure_directory(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        grants.append(token.lease_id)
        return result

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    assert await store.save("Bryan")
    restored = InterpersonalStore(root=store.path_for("Bryan").parent, authority=_Consent())
    assert grants and "sea anemones" in restored.render("Bryan")


async def test_regenerated_turn_replaces_interpersonal_episode_evidence(tmp_path):
    consent = _Consent()
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)
    await store.observe_turn(
        "Bryan",
        episode_id="ep-old",
        user_text="I prefer terse answers",
    )

    await store.observe_turn(
        "Bryan",
        episode_id="ep-new",
        user_text="I prefer terse answers",
        superseded_episode_ids=("ep-old",),
    )

    restored = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)
    occurrences = [
        item.episode_id
        for observation in restored.model_for("Bryan")
        for item in observation.occurrences
    ]
    assert occurrences == ["ep-new"]


async def test_a_person_is_not_named_in_the_filename(tmp_path):
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=_Consent())
    await store.observe_turn("Bryan", episode_id="ep1", user_text="I prefer terse answers")

    written = list((tmp_path / "interpersonal").iterdir())

    assert written and all("bryan" not in p.name.lower() for p in written)


def test_an_unreadable_file_is_kept_rather_than_overwritten(tmp_path):
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=_Consent())
    path = store.path_for("Bryan")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    model = store.model_for("Bryan")

    assert len(model) == 0
    assert store.get_status()["load_failures"] == 1
    kept = [p for p in path.parent.iterdir() if "unreadable" in p.name]
    assert kept, "the only evidence of what she knew was destroyed"


def test_notes_about_a_different_person_are_refused(tmp_path):
    """Conflating two people is the one merge this subsystem refuses outright."""
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=_Consent())
    path = store.path_for("Bryan")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A well-formed envelope, so this exercises the identity check rather than
    # tripping the parser on the way in.
    other = PersonModel("Tatiana")
    other.observe("prefers long explanations", episode_id="ep1")
    path.write_text(
        json.dumps(
            {
                "schema": "interpersonal_person_model",
                "schema_name": "interpersonal_person_model",
                "schema_version": 1,
                "payload": other.to_dict(),
            }
        ),
        encoding="utf-8",
    )

    model = store.model_for("Bryan")

    assert model.person == "Bryan"
    assert len(model) == 0, "notes about someone else were adopted"
    assert store.get_status()["load_failures"] == 1


# ── consent ────────────────────────────────────────────────────────────────


async def test_nothing_is_recorded_without_consent(tmp_path):
    store = InterpersonalStore(
        root=tmp_path / "interpersonal", authority=_Consent(allow=False)
    )

    written = await store.observe_turn(
        "Bryan", episode_id="ep1", user_text="I prefer terse answers"
    )

    assert written == []
    assert not (tmp_path / "interpersonal").exists()


async def test_nothing_is_rendered_without_consent(tmp_path):
    consent = _Consent()
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)
    await store.observe_turn("Bryan", episode_id="ep1", user_text="I prefer terse answers")
    assert store.render("Bryan")

    consent.allow = False

    assert store.render("Bryan") == ""


def test_consent_is_fail_closed_when_the_authority_is_missing(tmp_path):
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=object())

    assert store.allows("Bryan", "prompt") is False
    assert store.render("Bryan") == ""


def test_consent_is_fail_closed_when_the_authority_raises(tmp_path):
    class _Broken:
        def allows(self, *_args, **_kwargs):
            raise RuntimeError("consent store is down")

    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=_Broken())

    assert store.allows("Bryan", "prompt") is False


async def test_the_store_asks_about_the_kind_existing_revocations_cover(tmp_path):
    """A private kind of its own would escape every revocation already made."""
    consent = _Consent()
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=consent)

    await store.observe_turn("Bryan", episode_id="ep1", user_text="I prefer terse answers")

    assert {kind for _, kind, _ in consent.asked} == {"derived_profile"}


# ── nothing to say ─────────────────────────────────────────────────────────


def test_an_unknown_person_contributes_no_block(tmp_path):
    """An empty block is prompt budget spent to say nothing."""
    store = InterpersonalStore(root=tmp_path / "interpersonal", authority=_Consent())

    assert store.render("Someone She Has Never Met") == ""

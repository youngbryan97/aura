"""Everything she knows about a person came through one route.

`log_chat_turn_auto` is called from exactly one place in the tree — the HTTP
chat route in `interface/routes/chat_preflight.py` — and it is the only caller
of `InterpersonalStore.observe_turn`. A voice turn, an autonomous turn or a
battery turn updated nothing, so the organs that read the record had nothing to
read on any other lane: particularity, what she noticed and declined to write
down, and what he has told her he likes were all flat through a campaign.

The conversational dynamics phase runs on every user-facing turn on every lane
and now lets the record see the exchange. Two watchers of one turn record it
once, because the store treats the same words twice in a row as one exchange.
"""

from __future__ import annotations

import pytest

from core.memory.interpersonal_store import InterpersonalStore


class _Consent:
    """A stand-in for the relational memory authority, which fails closed."""

    def allows(self, agent_id: str, kind: str, operation: str) -> bool:
        del agent_id, kind, operation
        return True


@pytest.fixture()
def store(tmp_path, monkeypatch) -> InterpersonalStore:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    return InterpersonalStore(root=tmp_path / "people", authority=_Consent())


@pytest.mark.asyncio
async def test_a_turn_is_observed_once_however_many_watchers(store: InterpersonalStore) -> None:
    said = "I prefer short meetings and I care about clear writing."
    first = await store.observe_turn(
        "bryan", episode_id="route:1", user_text=said, assistant_text="Noted."
    )
    again = await store.observe_turn(
        "bryan", episode_id="turn:7", user_text=said, assistant_text="Noted."
    )
    assert first, "the first watcher should have recorded something"
    assert again == [], "the second watcher recorded the same turn again"

    model = store.model_for("bryan")
    for observation in model:
        assert observation.support == 1, (
            f"{observation.claim!r} was counted {observation.support} times for one turn"
        )


@pytest.mark.asyncio
async def test_a_different_turn_is_still_observed(store: InterpersonalStore) -> None:
    await store.observe_turn(
        "bryan", episode_id="a", user_text="I prefer short meetings.", assistant_text=""
    )
    written = await store.observe_turn(
        "bryan", episode_id="b", user_text="I care about clear writing.", assistant_text=""
    )
    assert written


@pytest.mark.asyncio
async def test_two_people_do_not_shadow_each_other(store: InterpersonalStore) -> None:
    said = "I prefer short meetings."
    await store.observe_turn("bryan", episode_id="a", user_text=said, assistant_text="")
    written = await store.observe_turn("sam", episode_id="b", user_text=said, assistant_text="")
    assert written, "one person's turn suppressed another's"


def test_the_phase_lets_the_record_see_the_turn() -> None:
    from pathlib import Path

    phase = Path("core/phases/conversational_dynamics_phase.py").read_text(encoding="utf-8")
    assert "_let_the_record_see_this_turn" in phase
    assert "observe_turn(" in phase


def test_the_route_is_no_longer_the_only_door() -> None:
    import subprocess

    found = subprocess.run(
        ["grep", "-rIl", "--include=*.py", "observe_turn(", "core", "interface"],
        capture_output=True,
        text=True,
        check=False,
    )
    files = {line for line in found.stdout.split() if "interpersonal_store.py" not in line}
    assert len(files) >= 2, f"only {files} write to the person record"


@pytest.mark.asyncio
async def test_a_regenerated_turn_is_the_exception(store: InterpersonalStore) -> None:
    """The same words under a new id, on purpose, carrying what they replace."""
    said = "I prefer terse answers"
    await store.observe_turn("bryan", episode_id="ep-old", user_text=said)
    await store.observe_turn(
        "bryan", episode_id="ep-new", user_text=said, superseded_episode_ids=("ep-old",)
    )
    episodes = [
        occurrence.episode_id
        for observation in store.model_for("bryan")
        for occurrence in observation.occurrences
    ]
    assert episodes == ["ep-new"]

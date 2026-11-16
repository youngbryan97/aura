"""The bubble's server surface: read and forward, never grant."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from interface.routes import ambient as ambient_routes


@pytest.fixture
def client(monkeypatch):
    from core.perception.ambient_presence import AmbientPresence, PresenceMode

    presence = AmbientPresence()
    presence.set_mode(PresenceMode.BUBBLE)
    monkeypatch.setattr(ambient_routes, "_presence", lambda: presence)
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: False
    )
    app = FastAPI()
    app.dependency_overrides = {}
    app.include_router(ambient_routes.router)
    from interface.auth import _require_internal

    app.dependency_overrides[_require_internal] = lambda: True
    return TestClient(app), presence


def test_state_is_pollable(client):
    api, _ = client
    body = api.get("/api/ambient/state").json()
    assert body["mode"] == "bubble"
    assert body["has_utterance"] is False


def test_state_stays_200_when_presence_is_broken(client, monkeypatch):
    api, _ = client

    def _broken():
        raise RuntimeError("presence gone")

    monkeypatch.setattr(ambient_routes, "_presence", _broken)
    response = api.get("/api/ambient/state")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_clearing_removes_the_message(client):
    api, presence = client
    presence.offer_utterance("something")
    assert api.post("/api/ambient/clear", json={}).json()["ok"] is True
    assert presence.state()["has_utterance"] is False


def test_hiding_stops_observation_not_just_display(client):
    api, presence = client
    assert api.post("/api/ambient/visibility", json={"mode": "hidden"}).json()["ok"]
    assert presence.mode.value == "hidden"


def test_an_unknown_mode_is_refused(client):
    api, _ = client
    assert api.post("/api/ambient/visibility", json={"mode": "spy"}).status_code == 400


def test_position_round_trips(client):
    api, _ = client
    body = api.post("/api/ambient/position", json={"x": 12.0, "y": 34.0}).json()
    assert body["position"] == [12.0, 34.0]


def test_native_movement_acknowledges_the_measured_position(client, monkeypatch):
    api, presence = client
    presence.state(surface="native-bubble")
    sequence = presence.request_bubble_move(900.0, 700.0)
    assert sequence == 1

    async def _persisted():
        return True

    monkeypatch.setattr(presence, "persist_bubble_position", _persisted)
    body = api.post(
        "/api/ambient/position",
        json={"x": 812.0, "y": 664.0, "sequence": sequence},
    ).json()

    assert body["acknowledged"] is True
    assert body["position"] == [812.0, 664.0]
    assert body["sequence"] == sequence


def test_recall_reports_not_fresh_rather_than_empty_truth(client):
    from core.perception.observation_evidence import get_observation_memory

    get_observation_memory().clear()
    api, _ = client
    body = api.post("/api/ambient/recall", json={"question": "what's on screen?"}).json()
    assert body["fresh"] is False
    assert "capture before answering" in body["note"]


def test_recall_needs_a_question(client):
    api, _ = client
    assert api.post("/api/ambient/recall", json={"question": " "}).status_code == 400


def test_there_is_no_route_that_makes_her_speak(client):
    """One authority for unprompted speech, and it is not HTTP."""
    paths = {route.path for route in ambient_routes.router.routes}
    assert not any("speak" in path or "say" in path for path in paths)


def test_a_long_remark_keeps_its_finished_sentences(client):
    """LIVE 15 November 2026: her remark reached the bubble as "... That'll let m", sliced at 600 characters."""
    api, presence = client
    said = (
        "I see the issue: your settings.json file has invalid JSON. The traceback shows it failing at line 4, "
        "column 29, with \"Expecting value.\" " + "A trailing comma after the last item in an array is the usual cause. " * 8
        + "Can you paste the file, or run cat on it and share the output? That'll let me point at the exact line."
    )
    assert len(said) > 600

    presence.offer_utterance(said)

    shown = api.get("/api/ambient/state").json()["utterance"]
    assert len(shown) <= 600
    assert shown.endswith((".", "?", "!")), shown[-40:]
    assert said.startswith(shown)

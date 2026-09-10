from tools import live_desktop_conversation_probe as probe


def test_chat_submission_has_durable_unique_identity():
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {"response": "A complete reply."}

    class Client:
        def post(self, url, **kwargs):
            calls.append(kwargs)
            return Response()

    for _ in range(2):
        assert probe._post_chat(
            Client(), "http://localhost", "same words",
            session_id="test-session", timeout_s=1,
        )[0]
    keys = [call["headers"]["X-Idempotency-Key"] for call in calls]
    assert keys[0] != keys[1]
    assert all(key.startswith("test-session:") for key in keys)


def test_failed_transport_stops_script_without_starting_another_turn(monkeypatch, tmp_path):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["X-Aura-Desktop-Request"] == "true"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def failed(*args, **kwargs):
        calls.append(kwargs)
        return False, "transport timeout", {}, 1.0

    monkeypatch.setattr(probe.httpx, "Client", Client)
    monkeypatch.setattr(probe, "_post_chat", failed)
    assert probe.run_probe(
        base_url="http://localhost", out=tmp_path,
        session_id="test", timeout_s=1,
    ) == 1
    assert len(calls) == 1

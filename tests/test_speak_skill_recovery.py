from types import SimpleNamespace

import pytest

from core.skills import speak as speak_mod


@pytest.mark.asyncio
async def test_speak_rejects_oversized_text_before_audio_attempt(monkeypatch):
    skill = speak_mod.SpeakSkill.__new__(speak_mod.SpeakSkill)
    skill._max_text_chars = 5
    skill._fallback_engine = None
    skill._get_engine = lambda: None

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("audio process should not start")

    monkeypatch.setattr(speak_mod.asyncio, "create_subprocess_exec", fail_if_called)

    result = await skill.execute({"text": "too long"}, {})

    assert result == {"ok": False, "error": "Speech text exceeds 5 characters."}


@pytest.mark.asyncio
async def test_macos_say_waits_for_process_and_reports_nonzero_exit(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        speak_mod,
        "_record_speak_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    monkeypatch.setattr(speak_mod.sys, "platform", "darwin")

    class FailedSayProcess:
        returncode = 36

        async def communicate(self):
            return b"", b"voice not found"

        def kill(self):
            raise AssertionError("process should not be killed without timeout")

    async def fake_exec(*args, **kwargs):
        assert args[:5] == ("say", "-v", "BadVoice", "-r", "185")
        assert kwargs["stdout"] is speak_mod.subprocess.PIPE
        assert kwargs["stderr"] is speak_mod.subprocess.PIPE
        return FailedSayProcess()

    monkeypatch.setattr(speak_mod.asyncio, "create_subprocess_exec", fake_exec)

    skill = speak_mod.SpeakSkill()
    skill._get_engine = lambda: None

    result = await skill.execute({"text": "hello", "voice": "BadVoice"}, {})

    assert result["ok"] is False
    assert result["attempts"][0]["mode"] == "macos_say"
    assert "voice not found" in result["attempts"][0]["error"]
    assert recorded[0][1]["action"] == "Fell back from macOS say to generic local speech engine"


@pytest.mark.asyncio
async def test_macos_say_timeout_kills_process(monkeypatch):
    monkeypatch.setattr(speak_mod.sys, "platform", "darwin")

    class HangingSayProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        async def communicate(self):
            if not self.killed:
                await speak_mod.asyncio.sleep(1)
            return b"", b""

        def kill(self):
            self.killed = True

    process = HangingSayProcess()

    async def fake_exec(*_args, **_kwargs):
        return process

    monkeypatch.setattr(speak_mod.asyncio, "create_subprocess_exec", fake_exec)

    skill = speak_mod.SpeakSkill()
    skill._get_engine = lambda: None
    skill._say_timeout_seconds = 0

    result = await skill.execute({"text": "hello"}, {})

    assert result["ok"] is False
    assert process.killed is True
    assert "timed out" in result["attempts"][0]["error"]


@pytest.mark.asyncio
async def test_pyttsx3_fallback_runs_after_primary_failure(monkeypatch):
    monkeypatch.setattr(speak_mod.sys, "platform", "linux")
    spoken = []

    class Fallback:
        def say(self, text):
            spoken.append(("say", text))

        def runAndWait(self):  # noqa: N802 - mirrors pyttsx3's public API
            spoken.append(("run", "done"))

    async def failing_primary(_text):
        raise RuntimeError("primary unavailable")

    skill = speak_mod.SpeakSkill.__new__(speak_mod.SpeakSkill)
    skill._max_text_chars = 5000
    skill._fallback_engine = Fallback()
    skill._get_engine = lambda: SimpleNamespace(synthesize_speech=failing_primary)

    result = await skill.execute({"text": "hello"}, {})

    assert result["ok"] is True
    assert result["mode"] == "pyttsx3"
    assert spoken == [("say", "hello"), ("run", "done")]

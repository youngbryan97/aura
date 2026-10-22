"""Voice presentation measurements expire when their source changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.organism.nothing_measured import NothingMeasured

SOURCES = (
    "interface/static/aura.js",
    "core/voice/duplex/config.py", "core/voice/duplex/tts_stream.py",
    "core/voice/duplex/pocket_speech.py", "core/voice/duplex/session.py",
    "core/voice/duplex/original_speech.py", "core/voice/duplex/treatment.py",
    "tools/voice/original_speech_worker.py",
    "interface/static/voice_mode.js", "interface/static/voice_mode.css",
    "interface/static/voice-capture-processor.js", "interface/static/voice-playback-processor.js",
)
CHECKS = frozenset({
    "ready audio does not wait for another clause", "both playback lanes flushed",
    "local interrupt before server command", "late audio rejected", "next turn plays",
    "physical microphone stopped", "390px width", "keyboard containment", "focus restored",
    "local model synthesis", "no routine metrics notices",
    "settings refresh preserves focused voice",
    "voice preview is playback",
})


def measured_voice(root: Path | None = None) -> bool:
    root = root or Path(__file__).resolve().parents[2]
    for path in sorted((root / "docs/evidence").glob("voice-polish-*.json"), reverse=True):
        try:
            receipt = json.loads(path.read_text())
            if receipt.get("schema") != "aura.voice.presentation_measurement.v1":
                continue
            if any(receipt.get("sources", {}).get(p) != hashlib.sha256((root / p).read_bytes()).hexdigest()
                   for p in SOURCES):
                continue
        except (OSError, ValueError):
            continue
        return receipt.get("ok") is True and CHECKS <= set(receipt.get("checks", []))
    raise NothingMeasured("No voice presentation measurement matches the current source.")


def install_voice_presentation_claims(suite) -> None:
    from core.organism.model_validation import Claim, Evidence, Observation, ValidationTest, boolean_score

    statement = "The measured voice client interrupts locally, rejects late audio, and releases the microphone when muted."
    suite.add_test(ValidationTest(
        name="voice_presentation_measured", description=statement, required_capability="",
        observation=Observation(name="voice_presentation", value=True,
                                source="tests/test_voice_ready_audio_does_not_wait.py; docs/evidence/voice-polish-*.json"),
        predict=lambda _m: measured_voice(),
        score=lambda p, o: boolean_score(bool(p), expected=bool(o.value), subject="voice presentation"),
        owner="interface/static/voice_mode.js",
    ))
    suite.add_claim(Claim(
        statement=statement, test="voice_presentation_measured", owner="interface/static/voice_mode.js",
        asserted_in=__file__, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Chromium with a simulated socket and microphone, plus local model synthesis. This does not establish conversational parity with ChatGPT Voice.",
    ))

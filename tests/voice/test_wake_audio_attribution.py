"""A video is not the owner speaking.

OWNER REPORT, 2026-08-10: "VIDEOS PLAYING ON MY COMPUTER ARE NOT ME SPEAKING."

The wake path could not tell. ``_verify_user_voice_print`` resolves a
``voice_identity`` / ``speaker_verifier`` service that is registered nowhere in
this codebase — the only ``verify_current_speaker`` implementation is a test
double — so the verified branch was unreachable in production, every wake
phrase was "unverified", and a command session opened for whatever made the
sound.
"""

from __future__ import annotations

import pytest


UNVERIFIED = {
    "verified": False,
    "confidence": 0.0,
    "reason": "voice_identity_verifier_unavailable",
}
VERIFIED = {"verified": True, "confidence": 0.93, "reason": "voice_identity_result"}


def _sources(*, playing: bool, processes=(), pids=(), readable: bool = True):
    from core.voice.audio_provenance import HostAudioSources

    return HostAudioSources(
        playing=playing,
        processes=tuple(processes),
        pids=tuple(pids),
        evidence="test",
        readable=readable,
    )


def test_no_speaker_verifier_is_registered_in_production() -> None:
    """Pin the premise. If a verifier ever ships, revisit this contract."""
    from core.container import ServiceContainer

    assert ServiceContainer.get("voice_identity", default=None) is None
    assert ServiceContainer.get("speaker_verifier", default=None) is None


def test_playing_audio_assertion_is_parsed() -> None:
    """The evidence format this host actually emits."""
    from core.voice import audio_provenance

    sample = (
        'pid 676(Google Chrome): [0x000656dc00019947] 00:04:24 '
        'NoIdleSleepAssertion named: "Playing audio"  \n'
        'pid 424(coreaudiod): [0x000656d90001826b] 00:04:27 '
        'PreventUserIdleSystemSleep named: '
        '"com.apple.audio.BuiltInSpeakerDevice.context.preventuseridlesleep"  \n'
    )

    class _Completed:
        returncode = 0
        stdout = sample

    # pmset is run through the subprocess gateway, so the gateway is what is answered: patching subprocess.run left
    # this test reading the real host's power assertions, which pass or fail by whatever is playing on the machine.
    from core.runtime import subprocess_gateway

    class _Gateway:
        def run(self, *_args, **_kwargs):
            return _Completed()

    audio_provenance._cached = None
    original = subprocess_gateway.get_subprocess_gateway
    subprocess_gateway.get_subprocess_gateway = lambda: _Gateway()
    try:
        sources = audio_provenance.host_audio_sources(force_refresh=True)
    finally:
        subprocess_gateway.get_subprocess_gateway = original
        audio_provenance._cached = None

    assert sources.playing is True
    assert sources.processes == ("Google Chrome",)
    assert sources.pids == (676,)
    # coreaudiod's device assertions are not playback and must not count.
    assert "coreaudiod" not in sources.processes


def test_unverified_speaker_during_playback_is_not_the_owner(monkeypatch) -> None:
    from core.voice import audio_provenance

    monkeypatch.setattr(
        audio_provenance,
        "foreign_audio_sources",
        lambda: _sources(playing=True, processes=("Google Chrome",), pids=(676,)),
    )

    verdict = audio_provenance.attribute_wake_audio(UNVERIFIED)

    assert verdict["owner_attributed"] is False
    assert "Google Chrome" in verdict["owner_attribution_reason"]


def test_verified_speaker_outranks_background_playback(monkeypatch) -> None:
    """Music playing must not lock the owner out once identity is proven."""
    from core.voice import audio_provenance

    monkeypatch.setattr(
        audio_provenance,
        "foreign_audio_sources",
        lambda: _sources(playing=True, processes=("Music",), pids=(999,)),
    )

    verdict = audio_provenance.attribute_wake_audio(VERIFIED)

    assert verdict["owner_attributed"] is True
    assert verdict["owner_attribution_reason"] == "speaker_identity_verified"


def test_silence_keeps_the_previous_permissive_behaviour(monkeypatch) -> None:
    """With nothing else making sound, an unverified wake still works."""
    from core.voice import audio_provenance

    monkeypatch.setattr(
        audio_provenance, "foreign_audio_sources", lambda: _sources(playing=False)
    )

    verdict = audio_provenance.attribute_wake_audio(UNVERIFIED)

    assert verdict["owner_attributed"] is True


def test_unreadable_host_does_not_claim_a_machine_spoke(monkeypatch) -> None:
    """No answer is not evidence of a foreign speaker — fail open, and say so."""
    from core.voice import audio_provenance

    monkeypatch.setattr(
        audio_provenance,
        "foreign_audio_sources",
        lambda: _sources(playing=False, readable=False),
    )

    verdict = audio_provenance.attribute_wake_audio(UNVERIFIED)

    assert verdict["owner_attributed"] is True
    assert "unreadable" in verdict["owner_attribution_reason"]


def test_auras_own_output_is_not_a_foreign_speaker() -> None:
    """Her TTS must never lock her out of being woken."""
    import os

    from core.voice.audio_provenance import HostAudioSources

    mine = os.getpid()
    sources = HostAudioSources(
        playing=True,
        processes=("Python", "Google Chrome"),
        pids=(mine, 676),
        evidence="test",
    )

    foreign = sources.excluding(mine)

    assert foreign.processes == ("Google Chrome",)
    assert foreign.playing is True

    only_mine = HostAudioSources(
        playing=True, processes=("Python",), pids=(mine,), evidence="test"
    ).excluding(mine)

    assert only_mine.playing is False


def test_wake_word_decides_attribution_before_opening_a_session() -> None:
    """State must not be committed and then rolled back."""
    import inspect

    from core.voice import wake_word

    source = inspect.getsource(wake_word.WakeWordDetector._check_wake_word)
    attribution = source.find("attribute_wake_audio")
    session_start = source.find("WakeState.LISTENING")

    assert attribution != -1, "wake path does not attribute audio at all"
    assert attribution < session_start, (
        "attribution must precede committing the listening session"
    )


@pytest.mark.asyncio
async def test_unattributed_wake_starts_no_session(monkeypatch) -> None:
    """End to end: a wake phrase heard over playback opens nothing."""
    from core.voice import wake_word
    from core.voice.wake_word import WakeState

    detector = wake_word.WakeWordDetector()
    monkeypatch.setattr(
        detector, "_verify_user_voice_print", lambda transcript: _async(UNVERIFIED)
    )
    monkeypatch.setattr(
        wake_word,
        "attribute_wake_audio",
        lambda evidence: {
            **evidence,
            "owner_attributed": False,
            "owner_attribution_reason": "unverified_speaker_while_host_audio_playing:Google Chrome",
        },
    )
    observed: list[str] = []
    monkeypatch.setattr(
        detector,
        "_record_wake_observation",
        lambda summary, evidence, *, salience: observed.append(summary),
    )

    before = detector._wake_count
    await detector._check_wake_word("hey aura, delete my notes")

    assert detector.state is not WakeState.LISTENING
    assert detector._wake_count == before
    assert observed and "unattributed" in observed[0].lower()


async def _async(value):
    return value

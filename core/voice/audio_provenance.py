"""Who is making sound in this room, and is any of it Aura's own or a machine's.

OWNER REPORT, 2026-08-10: "VIDEOS PLAYING ON MY COMPUTER ARE NOT ME SPEAKING."

The wake word had no way to know that. ``_verify_user_voice_print`` asks the
container for a ``voice_identity`` / ``speaker_verifier`` service, and no such
service is registered anywhere in this codebase — the only implementation of
``verify_current_speaker`` is a test double. So speaker verification could
never succeed in production, every wake word was accepted as "unverified", and
a command session opened regardless. A video saying the wake phrase was
indistinguishable from the owner saying it.

macOS already knows which processes are playing audio: any app holding an
``NoIdleSleepAssertion named: "Playing audio"`` is producing output right now.
That is a fact about the host, not a heuristic about the waveform, and it is
the evidence this module reports.

What it does NOT do is identify a speaker. Knowing that Chrome is playing
audio does not prove the wake word came from Chrome, and silence does not
prove the owner spoke. This module reports evidence; the caller decides.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation

# pid 676(Google Chrome): [0x...] 00:04:24 NoIdleSleepAssertion named: "Playing audio"
_ASSERTION_LINE = re.compile(
    r"pid\s+(?P<pid>\d+)\((?P<process>[^)]*)\).*?named:\s*\"(?P<name>[^\"]*)\"",
    re.IGNORECASE,
)
_PLAYING_AUDIO = "playing audio"

# One sample is reused for this long. The wake path runs per transcript chunk
# and must not fork a subprocess each time; playback state does not change
# meaningfully faster than a spoken wake phrase.
_SAMPLE_TTL_S = 2.0

_cached: tuple[float, "HostAudioSources"] | None = None


@dataclass(frozen=True)
class HostAudioSources:
    """Processes producing audio output at sample time."""

    playing: bool
    processes: tuple[str, ...] = ()
    pids: tuple[int, ...] = ()
    evidence: str = ""
    readable: bool = True
    sampled_at: float = field(default_factory=time.time)

    def excluding(self, pid: int) -> "HostAudioSources":
        """Drop one pid — Aura's own output is not a foreign speaker.

        Her TTS is already handled by the duplex barge-in path; counting it
        here would mean she could never be woken while she was talking.
        """
        kept = [
            (process, process_pid)
            for process, process_pid in zip(self.processes, self.pids, strict=False)
            if process_pid != pid
        ]
        return HostAudioSources(
            playing=bool(kept),
            processes=tuple(process for process, _ in kept),
            pids=tuple(process_pid for _, process_pid in kept),
            evidence=self.evidence,
            readable=self.readable,
            sampled_at=self.sampled_at,
        )

    def as_evidence(self) -> dict[str, object]:
        return {
            "host_audio_playing": self.playing,
            "host_audio_processes": list(self.processes),
            "host_audio_readable": self.readable,
            "host_audio_evidence": self.evidence,
        }


def _unreadable(reason: str) -> HostAudioSources:
    """No answer is not the same as "nothing is playing"."""
    return HostAudioSources(playing=False, readable=False, evidence=reason)


def host_audio_sources(*, force_refresh: bool = False) -> HostAudioSources:
    """Which processes are playing audio on this host right now."""
    global _cached

    now = time.time()
    if not force_refresh and _cached is not None:
        sampled_at, sources = _cached
        if now - sampled_at < _SAMPLE_TTL_S:
            return sources

    try:
        # Through the gateway, like every other process this runtime starts.
        # A raw subprocess.run here is invisible to the one component that
        # knows what is running, which is how a read that looks harmless
        # becomes an unaccounted fork on a path that samples every wake word.
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        completed = get_subprocess_gateway().run(
            ["pmset", "-g", "assertions"],
            timeout=2.0,
            read_only=True,
            check=False,
            source="voice.audio_provenance",
            # pmset reads power-assertion state. It touches no accelerator, and
            # saying so is what keeps the declaration meaningful for the calls
            # that do.
            accelerator_capability="none",
        )
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        record_degradation(
            "voice.audio_provenance",
            exc,
            severity="warning",
            action="reported host audio as unreadable rather than assuming silence",
            enforce_failure_policy=False,
        )
        sources = _unreadable(f"pmset_failed:{type(exc).__name__}")
        _cached = (now, sources)
        return sources

    if completed.returncode != 0:
        sources = _unreadable(f"pmset_exit_{completed.returncode}")
        _cached = (now, sources)
        return sources

    processes: list[str] = []
    pids: list[int] = []
    for line in (completed.stdout or "").splitlines():
        if _PLAYING_AUDIO not in line.lower():
            continue
        match = _ASSERTION_LINE.search(line)
        if not match or _PLAYING_AUDIO not in match.group("name").lower():
            continue
        try:
            pid = int(match.group("pid"))
        except (TypeError, ValueError):
            continue
        processes.append(match.group("process").strip())
        pids.append(pid)

    sources = HostAudioSources(
        playing=bool(pids),
        processes=tuple(processes),
        pids=tuple(pids),
        evidence="pmset_no_idle_sleep_assertion",
        readable=True,
        sampled_at=now,
    )
    _cached = (now, sources)
    return sources


def foreign_audio_sources() -> HostAudioSources:
    """Host audio that is not Aura's own output."""
    return host_audio_sources().excluding(os.getpid())


def attribute_wake_audio(voice_evidence: dict[str, object]) -> dict[str, object]:
    """Decide whether a wake phrase can be attributed to the owner.

    Returns the evidence dict extended with an ``owner_attributed`` verdict and
    the reason for it. The verdict is deliberately conservative in exactly one
    direction: it never UPGRADES an unverified speaker to the owner, and it
    never claims a machine spoke when the host could not be read.
    """
    evidence = dict(voice_evidence or {})
    sources = foreign_audio_sources()
    evidence.update(sources.as_evidence())

    if evidence.get("verified"):
        # A verified speaker outranks any amount of background playback.
        evidence["owner_attributed"] = True
        evidence["owner_attribution_reason"] = "speaker_identity_verified"
        return evidence

    if sources.playing:
        evidence["owner_attributed"] = False
        evidence["owner_attribution_reason"] = (
            "unverified_speaker_while_host_audio_playing:"
            + ",".join(sources.processes[:4])
        )
        return evidence

    # Nothing else is making sound and nobody verified the speaker. That is
    # the ordinary case and stays exactly as permissive as it was before.
    evidence["owner_attributed"] = True
    evidence["owner_attribution_reason"] = (
        "no_competing_audio_source"
        if sources.readable
        else "host_audio_unreadable_defaulting_to_prior_behaviour"
    )
    return evidence

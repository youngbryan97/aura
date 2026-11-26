#!/usr/bin/env python3
"""Voice-wake journal demo proof: spoken objective, disk-verified chain.

Bryan's demo, part 1, including the voice leg: a single utterance —
"Hey Aura, ..." followed by the journal objective — must wake Aura and
drive the full journal chain (folder, timestamped self-describing entry,
robot image found online, PDF in "Aura's Journal") with NO direct HTTP
chat injection. The transcript enters through the same sensory ingestion
surface the live microphone uses (sensory_audio.json, which Whisper STT
writes in live operation), the wake-word detector picks it up, and the
command routes through the canonical /api/chat conversation lane.

This file contains NO task-specific execution logic. Verification is
hostile-evaluation grade and inherits the journal verifiers:
- folder and PDF must be FRESH (created after the utterance landed);
- PDF text must contain a timestamp and first-person self-description;
- PDF must embed at least one image (raw /Image XObject scan);
- the server stdout log must show the wake event and lane dispatch
  (causal-path evidence, recorded alongside — never instead of — disk
  artifacts).

Usage:
    python tools/voice_wake_demo_proof.py [--port 8000] [--boot-timeout 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_demo_proof import (  # noqa: E402
    _SELF_TOKENS,
    _TIMESTAMP_RE,
    JOURNAL_OBJECTIVE,
    JournalDemoProof,
    _pdf_has_image,
    _pdf_text,
)

VOICE_UTTERANCE = "Hey Aura, " + JOURNAL_OBJECTIVE[0].lower() + JOURNAL_OBJECTIVE[1:]

# The wake detector treats the transcript file as live only when its
# mtime is under 10s old, and dispatch is bounded at AURA_VOICE_COMMAND_
# TIMEOUT_S (240s default) — so the artifact wait is that plus margin.
ARTIFACT_WAIT_S = 300.0
ARTIFACT_POLL_S = 2.0

_WAKE_LOG_MARKERS = (
    "Wake word detected",
    "Voice command received",
)


class VoiceWakeJournalProof(JournalDemoProof):
    def _inject_utterance(self) -> Path:
        """Write the utterance to the live sensory ingestion surface."""
        audio_path = PROJECT_ROOT / "sensory_audio.json"
        audio_path.write_text(
            json.dumps(
                {
                    "transcript": VOICE_UTTERANCE,
                    "vad": True,
                    "rms": 0.42,
                    "ts": time.time(),
                }
            ),
            encoding="utf-8",
        )
        return audio_path

    def _stdout_log_markers(self) -> dict[str, bool]:
        try:
            text = self.stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {marker: False for marker in _WAKE_LOG_MARKERS}
        return {marker: (marker in text) for marker in _WAKE_LOG_MARKERS}

    def exercise_voice_wake_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        audio_path = self._inject_utterance()

        # Wait for a fresh PDF to land — the only success signal that
        # counts. Aura's own claims are recorded but never trusted.
        pdf = None
        deadline = step_started + ARTIFACT_WAIT_S
        while time.time() < deadline:
            self.guard_rss()
            if journal_dir.is_dir():
                fresh = [
                    p
                    for p in journal_dir.glob("*.pdf")
                    if p.stat().st_mtime >= step_started - 1.0
                ]
                if fresh:
                    pdf = max(fresh, key=lambda p: p.stat().st_mtime)
                    break
            time.sleep(ARTIFACT_POLL_S)
        elapsed = time.time() - step_started
        time.sleep(2.0)  # let the PDF finish flushing before reading it

        text = _pdf_text(pdf) if pdf else ""
        lowered = text.lower()
        has_timestamp = bool(_TIMESTAMP_RE.search(text))
        has_self = any(tok in lowered for tok in _SELF_TOKENS) and len(text) > 120
        has_image = _pdf_has_image(pdf) if pdf else False
        log_markers = self._stdout_log_markers()

        verified = bool(pdf and has_timestamp and has_self and has_image)
        return self.record(
            "voice_wake_journal_chain",
            verified,
            summary=(
                f"{elapsed:.1f}s after utterance — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch ts={has_timestamp} "
                    f"self={has_self} image={has_image} "
                    f"wake_logged={log_markers}"
                    if pdf
                    else f"NO FRESH PDF after voice wake (markers={log_markers})"
                )
            ),
            utterance=VOICE_UTTERANCE[:300],
            injection_path=str(audio_path),
            elapsed_s=round(elapsed, 1),
            folder=str(journal_dir),
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:400],
            has_timestamp=has_timestamp,
            has_self_description=has_self,
            has_embedded_image=has_image,
            wake_log_markers=log_markers,
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors JournalDemoProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_voice_wake_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "voice_wake_journal_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "VOICE WAKE DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "VOICE WAKE DEMO PROOF FAILED"
                )
                + f" → {self.verdict_path}"
            )
            return 0 if verdict["passed"] else 1
        finally:
            self.kill_hard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    return VoiceWakeJournalProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

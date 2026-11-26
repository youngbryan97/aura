#!/usr/bin/env python3
"""Combined demo proof: Bryan's part 1, literally — say it, watch it, verify it.

One spoken utterance — "Hey Aura, open up my Notes app and write a
journal entry... I want to see you do it... save as PDF in Aura's
Journal" — enters through the live sensory ingestion surface, wakes
her, and drives the VISIBLE chain on the real screen in desktop mode:
Notes opens, the entry stages in front of the user, the robot image is
fetched with source evidence, and the searchable PDF lands in the
folder.

No HTTP chat injection, no task-specific execution code. Verification
is hostile-evaluation grade and entirely external:
- fresh PDF with text layer, timestamp, first-person self-description,
  embedded image (inherited journal verifiers);
- wake + command markers in the preserved server stdout log;
- a fresh desktop_objective output receipt in the durable receipt
  ledger (~/.aura/receipts/output) created inside the run window — the
  voice dispatch returns its payload to the wake detector, not to this
  proof, so the on-disk governed ledger is the external evidence that
  the canonical conversation lane served the spoken command.

Usage:
    python tools/combined_demo_proof.py [--port 8000] [--boot-timeout 600]
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

from tools.journal_demo_proof import JournalDemoProof  # noqa: E402
from tools.visible_journal_demo_proof import VISIBLE_OBJECTIVE  # noqa: E402
from tools.voice_wake_demo_proof import (  # noqa: E402
    ARTIFACT_POLL_S,
    ARTIFACT_WAIT_S,
    VoiceWakeJournalProof,
)

COMBINED_UTTERANCE = (
    "Hey Aura, " + VISIBLE_OBJECTIVE[0].lower() + VISIBLE_OBJECTIVE[1:]
)

_OUTPUT_RECEIPT_DIR = Path.home() / ".aura" / "receipts" / "output"


def _fresh_desktop_output_receipt(step_started: float) -> dict:
    """Find a desktop_objective output receipt created during the run."""
    best: dict = {}
    try:
        candidates = sorted(
            _OUTPUT_RECEIPT_DIR.glob("output-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:50]
    except OSError:
        return {"found": False, "error": "receipt ledger unreadable"}
    for path in candidates:
        if path.stat().st_mtime < step_started - 1.0:
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8")).get("payload") or {}
        except (OSError, ValueError):
            continue
        cause = str(payload.get("cause") or "")
        created = float(payload.get("created_at") or 0.0)
        if "desktop_objective" in cause and created >= step_started - 1.0:
            best = {
                "found": True,
                "cause": cause,
                "created_at": created,
                "receipt_id": str(payload.get("receipt_id") or ""),
                "status": str((payload.get("metadata") or {}).get("status") or ""),
            }
            break
    return best or {"found": False}


class CombinedDemoProof(VoiceWakeJournalProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        # Skip VoiceWakeJournalProof.__init__ mode default: the combined
        # finale runs the VISIBLE desktop runtime.
        JournalDemoProof.__init__(
            self, port=port, boot_timeout_s=boot_timeout_s, mode="desktop"
        )

    def _inject_utterance(self) -> Path:
        audio_path = PROJECT_ROOT / "sensory_audio.json"
        audio_path.write_text(
            json.dumps(
                {
                    "transcript": COMBINED_UTTERANCE,
                    "vad": True,
                    "rms": 0.42,
                    "ts": time.time(),
                }
            ),
            encoding="utf-8",
        )
        return audio_path

    def exercise_combined_chain(self) -> bool:
        from tools.journal_demo_proof import (
            _SELF_TOKENS,
            _has_fresh_timestamp,
            _pdf_has_image,
            _pdf_text,
        )

        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        audio_path = self._inject_utterance()

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
        time.sleep(2.0)

        text = _pdf_text(pdf) if pdf else ""
        lowered = text.lower()
        has_timestamp = _has_fresh_timestamp(text, step_started)
        has_self = any(tok in lowered for tok in _SELF_TOKENS) and len(text) > 120
        has_image = _pdf_has_image(pdf) if pdf else False
        log_markers = self._stdout_log_markers()
        lane_receipt = _fresh_desktop_output_receipt(step_started)

        verified = bool(
            pdf
            and has_timestamp
            and has_self
            and has_image
            and all(log_markers.values())
            and lane_receipt.get("found")
        )
        return self.record(
            "combined_voice_visible_chain",
            verified,
            summary=(
                f"{elapsed:.1f}s after utterance — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch ts={has_timestamp} self={has_self} "
                    f"image={has_image} wake={log_markers} lane_receipt="
                    f"{lane_receipt.get('cause', 'MISSING')}"
                    if pdf
                    else f"NO FRESH PDF (wake={log_markers}, receipt={lane_receipt})"
                )
            ),
            utterance=COMBINED_UTTERANCE[:300],
            injection_path=str(audio_path),
            elapsed_s=round(elapsed, 1),
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:400],
            has_timestamp=has_timestamp,
            has_self_description=has_self,
            has_embedded_image=has_image,
            wake_log_markers=log_markers,
            lane_output_receipt=lane_receipt,
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors VoiceWakeJournalProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_combined_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "combined_voice_visible_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "COMBINED DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "COMBINED DEMO PROOF FAILED"
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
    return CombinedDemoProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

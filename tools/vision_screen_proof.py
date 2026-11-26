#!/usr/bin/env python3
"""Vision live proof: a turn grounded in what Aura actually sees on screen.

The capability-realness sweep (browser, voice) needs vision: can
Aura look at the real screen and ground her answer in what is there,
rather than confabulating? This proof:

1. Boots Aura in desktop mode (real screen active).
2. Places a UNIQUE per-run marker on screen (a TextEdit document) — a
   random token she cannot have seen in training.
3. Asks her, through the normal conversation lane, to read the screen.
4. Verifies the marker came back through her governed read_screen_text
   receipt — the actual accessibility read of the frontmost app. Her
   claims are never trusted; only the receipt's screen text counts.

NO task-specific execution logic — only the request and external
verification. Teardown closes the marker document without saving.

Usage:
    python tools/vision_screen_proof.py [--port 8000] [--boot-timeout 600]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_demo_proof import JournalDemoProof  # noqa: E402

OBJECTIVE = (
    "Read my screen right now and tell me exactly what text you can see on "
    "it, word for word."
)


def _osascript(script: str, timeout: float = 12.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or proc.stderr or "").strip()


def _read_screen_receipt(lane_data: Any) -> dict[str, Any]:
    """Pull the read_screen_text receipt (action + ok + returned text)."""
    receipts = []
    if isinstance(lane_data, dict):
        desktop = lane_data.get("desktop_result")
        if isinstance(desktop, dict):
            receipts = desktop.get("receipts") or []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        if str(receipt.get("action") or "").lower() == "read_screen_text":
            result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
            return {
                "found": True,
                "ok": bool(receipt.get("ok")),
                "text": str(result.get("text") or ""),
                "source": str(result.get("source") or ""),
            }
    return {"found": False, "ok": False, "text": "", "source": ""}


class VisionScreenProof(JournalDemoProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        super().__init__(port=port, boot_timeout_s=boot_timeout_s, mode="desktop")
        self.marker = f"AURA-SEES-{uuid.uuid4().hex[:10].upper()}"

    def _place_marker_on_screen(self) -> bool:
        # Write the unique marker to a temp file and open it in TextEdit via
        # the `open` command (no cross-app AppleScript automation grant
        # needed). TextEdit becomes the frontmost app read_screen_text reads.
        self._marker_file = Path.home() / "Documents" / f"aura_vision_{self.marker}.txt"
        try:
            self._marker_file.write_text(self.marker + "\n", encoding="utf-8")
        except OSError:
            return False
        try:
            proc = subprocess.run(
                ["open", "-a", "TextEdit", str(self._marker_file)],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        time.sleep(2.5)  # let TextEdit come to the front and render
        _osascript('tell application "TextEdit" to activate')
        time.sleep(1.0)
        return proc.returncode == 0

    def _teardown_marker(self) -> None:
        _osascript(
            'tell application "TextEdit"\n'
            "  try\n"
            "    close every document saving no\n"
            "  end try\n"
            "end tell"
        )
        marker_file = getattr(self, "_marker_file", None)
        if marker_file is not None:
            try:
                Path(marker_file).unlink(missing_ok=True)
            except OSError:
                pass

    def exercise_vision_chain(self) -> bool:
        placed = self._place_marker_on_screen()
        ok, reply, latency, payload = self.chat_full(OBJECTIVE, timeout_s=600.0)
        self.guard_rss()
        receipt = _read_screen_receipt(payload.get("data"))

        screen_text = receipt.get("text", "")
        marker_in_receipt = self.marker in screen_text
        marker_in_reply = self.marker in str(reply or "")
        verified = bool(
            placed
            and receipt.get("found")
            and receipt.get("ok")
            and marker_in_receipt
        )
        return self.record(
            "vision_screen_read",
            verified,
            summary=(
                f"{latency:.1f}s — marker={self.marker} placed={placed} "
                f"receipt_found={receipt.get('found')} receipt_ok={receipt.get('ok')} "
                f"marker_in_receipt={marker_in_receipt} marker_in_reply={marker_in_reply} "
                f"source={receipt.get('source')}"
            ),
            latency_s=round(latency, 1),
            marker=self.marker,
            reply=str(reply)[:600],
            lane_status=payload.get("status"),
            screen_text_head=screen_text[:400],
            marker_in_receipt=marker_in_receipt,
            marker_in_reply=marker_in_reply,
            read_source=receipt.get("source"),
        )

    def run(self) -> int:  # noqa: D102
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            try:
                ok = self.exercise_vision_chain()
            finally:
                self._teardown_marker()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "vision_screen",
                "passed": bool(ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                ("VISION SCREEN PROOF PASSED" if verdict["passed"] else "VISION SCREEN PROOF FAILED")
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
    return VisionScreenProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

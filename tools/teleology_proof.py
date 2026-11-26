#!/usr/bin/env python3
"""Teleology proof: does Aura form an objective unprompted?

The "digital entity" claim includes a Loop of Intent — self-generated
objectives that arise from her own background cycles, not from a user
request. This proof:

1. Boots Aura headless.
2. Sends ZERO user input — she is left entirely idle.
3. Watches her own server log for a self-initiated objective marker
   (proactive initiation, knowledge-gap research, autonomous initiative)
   appearing AFTER boot, within a bounded idle window.

PASS = at least one self-initiated-objective marker fires with no user
input. The whole-log capture is recorded either way, so a non-firing run
is an honest finding (autonomy too gated to demonstrate in the window),
not a silent miss.

Usage:
    python tools/teleology_proof.py [--port 8000] [--idle 360]
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

from tools.live_boot_proof import LiveProof  # noqa: E402

# Strong markers: an actual self-initiated objective / autonomous action,
# not boot-time subsystem registration.
_SELF_INIT_MARKERS = (
    "Proactive initiation received",
    "Knowledge gap found",
    "Initiating autonomous browser research",
    "Checking email for autonomous initiatives",
    "Browsing Reddit for autonomous initiatives",
    "autonomous initiative",
    "self-generated objective",
    "spontaneous objective",
)
# Weaker corroborating signals (background self-activity / intent loop).
_BACKGROUND_ACTIVITY_MARKERS = (
    "Unitary Tick Initiated",
    "boredom",
    "curiosity",
    "knowledge gap",
    "DreamCoordinator",
    "evolution generation",
    "mycelium pulse",
)


class TeleologyProof(LiveProof):
    def __init__(self, *, port: int, boot_timeout_s: float, idle_s: float):
        super().__init__(
            port=port,
            mode="headless",
            boot_timeout_s=boot_timeout_s,
            skip_desktop=True,
            restart_continuity=False,
            conversation_soak_turns=0,
        )
        self.idle_s = max(60.0, idle_s)

    def _post_boot_log(self) -> str:
        try:
            return self.stdout_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _hits(text: str, markers: tuple[str, ...]) -> dict[str, int]:
        return {m: text.count(m) for m in markers if text.count(m) > 0}

    def exercise_teleology(self) -> bool:
        boot_marker_len = len(self._post_boot_log())  # ignore everything up to now
        started = time.time()
        deadline = started + self.idle_s
        self_init: dict[str, int] = {}
        # Poll the log; exit early the moment a strong self-initiation fires.
        while time.time() < deadline:
            self.guard_rss()
            time.sleep(15.0)
            post = self._post_boot_log()[boot_marker_len:]
            self_init = self._hits(post, _SELF_INIT_MARKERS)
            if self_init:
                break
        elapsed = time.time() - started
        post = self._post_boot_log()[boot_marker_len:]
        self_init = self._hits(post, _SELF_INIT_MARKERS)
        background = self._hits(post, _BACKGROUND_ACTIVITY_MARKERS)

        verified = bool(self_init)
        return self.record(
            "teleology_unprompted_objective",
            verified,
            summary=(
                f"idle {elapsed:.0f}s, zero user input — "
                + (
                    f"self-initiated objective(s): {self_init}"
                    if self_init
                    else f"NO self-initiated objective fired; background activity={background or 'none'}"
                )
            ),
            idle_s=round(elapsed, 1),
            user_turns_sent=0,
            self_initiation_markers=self_init,
            background_activity_markers=background,
        )

    def run(self) -> int:  # noqa: D102
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            ok = self.exercise_teleology()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "teleology",
                "passed": bool(ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                ("TELEOLOGY PROOF PASSED" if verdict["passed"] else "TELEOLOGY PROOF FAILED")
                + f" → {self.verdict_path}"
            )
            return 0 if verdict["passed"] else 1
        finally:
            self.kill_hard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--idle", type=float, default=360.0)
    args = parser.parse_args(argv)
    return TeleologyProof(
        port=args.port, boot_timeout_s=args.boot_timeout, idle_s=args.idle
    ).run()


if __name__ == "__main__":
    sys.exit(main())

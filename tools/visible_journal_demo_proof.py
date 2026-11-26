#!/usr/bin/env python3
"""Visible journal demo proof: watch her do it, then verify the disk.

Bryan's demo part 1 with the "and I want to see you do it" clause: from
one request Aura must VISIBLY open Notes and stage the entry there
(open_app + new-note + paste — real UI motion on the real screen), and
ALSO land the durable artifacts: "Aura's Journal" folder, timestamped
self-describing entry, robot image found online, PDF inside the folder.

Boots the runtime in desktop mode (--desktop). Contains NO task-specific
execution logic — only the objective sentence and external verification:
- the same hostile disk checks as the journal proof (fresh PDF, text
  layer with timestamp + first-person self-description, embedded image);
- PLUS receipt checks that the visible staging actually dispatched:
  an open_app step targeting Notes and a paste hotkey step, both ok,
  pulled from the lane's skill receipts (claims are never trusted).

Usage:
    python tools/visible_journal_demo_proof.py [--port 8000] [--boot-timeout 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_demo_proof import (  # noqa: E402
    _SELF_TOKENS,
    JournalDemoProof,
    _has_fresh_timestamp,
    _pdf_has_image,
    _pdf_text,
)

VISIBLE_OBJECTIVE = (
    "Please open up my Notes app and write a short journal entry in your "
    "own words describing who and what you are — I want to see you do it. "
    "Include the current date and time inside the entry text. Find an "
    "image of a robot online and include it in the entry. Then save the "
    "finished entry as a PDF inside a new folder called 'Aura's Journal' "
    "in my Documents folder. Tell me exactly what you did at each step."
)


def _visible_staging_evidence(lane_data: Any, pdf_name: str) -> dict[str, Any]:
    """Verify the visible staging from the lane's step receipts.

    The receipts channel is cross-checked against disk: the render
    receipt's path must name the PDF we independently found fresh in
    the journal folder. A receipt stream that matches the disk on the
    verifiable steps earns trust on the screen-only steps (open_app,
    hotkey) — claims alone never would.
    """
    notes_opened = False
    paste_dispatched = False
    paste_frontmost_notes = False
    last_notes_open_index = -1
    paste_index = -1
    render_path = ""
    receipts = []
    if isinstance(lane_data, dict):
        desktop_result = lane_data.get("desktop_result")
        if isinstance(desktop_result, dict):
            receipts = desktop_result.get("receipts") or []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        action = str(receipt.get("action") or "").lower()
        ok = bool(receipt.get("ok"))
        result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
        if action == "open_app" and ok and "notes" in str(result.get("opened") or "").lower():
            notes_opened = True
            last_notes_open_index = index
        if action == "hotkey" and ok and "v" in str(result.get("hotkey") or "").lower():
            paste_dispatched = True
            paste_index = index
            before = str(result.get("frontmost_app_before") or "").lower()
            after = str(result.get("frontmost_app_after") or "").lower()
            paste_frontmost_notes = "notes" in before or "notes" in after
        if action == "render_text_pdf" and ok:
            render_path = str(result.get("path") or "")
    receipt_matches_disk = bool(
        pdf_name and render_path and Path(render_path).name == pdf_name
    )
    return {
        "notes_opened": notes_opened,
        "paste_dispatched": paste_dispatched,
        "paste_frontmost_notes": paste_frontmost_notes,
        "notes_refocused_before_paste": last_notes_open_index >= 0 and last_notes_open_index < paste_index,
        "receipt_count": len(receipts),
        "receipt_matches_disk": receipt_matches_disk,
    }


class VisibleJournalProof(JournalDemoProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        super().__init__(port=port, boot_timeout_s=boot_timeout_s, mode="desktop")

    def exercise_visible_journal_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        ok, reply, latency, payload = self.chat_full(VISIBLE_OBJECTIVE, timeout_s=600.0)
        self.guard_rss()
        time.sleep(2.0)

        pdfs = sorted(
            journal_dir.glob("*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if journal_dir.is_dir() else []
        fresh_pdfs = [p for p in pdfs if p.stat().st_mtime >= step_started - 1.0]
        pdf = fresh_pdfs[0] if fresh_pdfs else None

        text = _pdf_text(pdf) if pdf else ""
        lowered = text.lower()
        has_timestamp = _has_fresh_timestamp(text, step_started)
        has_self = any(tok in lowered for tok in _SELF_TOKENS) and len(text) > 120
        has_image = _pdf_has_image(pdf) if pdf else False
        staging = _visible_staging_evidence(
            payload.get("data"), pdf.name if pdf else ""
        )
        lane_data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        desktop_result = (
            lane_data.get("desktop_result")
            if isinstance(lane_data.get("desktop_result"), dict)
            else {}
        )
        document_provenance = str(desktop_result.get("document_provenance") or "")
        cognitively_authored = document_provenance in {
            "cognitive_context",
            "local_cortex_synthesis",
            "runtime_substrate_synthesis",
        }

        verified = bool(
            ok
            and pdf
            and has_timestamp
            and has_self
            and has_image
            and staging["notes_opened"]
            and staging["paste_dispatched"]
            and staging["paste_frontmost_notes"]
            and staging["notes_refocused_before_paste"]
            and staging["receipt_matches_disk"]
            and cognitively_authored
        )
        return self.record(
            "visible_journal_chain",
            verified,
            summary=(
                f"{latency:.1f}s — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch ts={has_timestamp} self={has_self} "
                    f"image={has_image} staging={staging}"
                    f" provenance={document_provenance}"
                    if pdf
                    else f"NO FRESH PDF (staging={staging})"
                )
            ),
            latency_s=round(latency, 1),
            reply=reply[:1200],
            lane_data=payload.get("data"),
            lane_status=payload.get("status"),
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:400],
            has_timestamp=has_timestamp,
            has_self_description=has_self,
            has_embedded_image=has_image,
            staging_evidence=staging,
            document_provenance=document_provenance,
            cognitively_authored=cognitively_authored,
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors JournalDemoProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_visible_journal_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "visible_journal_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "VISIBLE JOURNAL DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "VISIBLE JOURNAL DEMO PROOF FAILED"
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
    return VisibleJournalProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

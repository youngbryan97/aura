#!/usr/bin/env python3
"""Journal demo proof: one natural-language objective, disk-verified chain.

Bryan's demo, part 1, as an autonomous proof: from a single request Aura
must create an "Aura's Journal" folder, write a timestamped journal entry
in her own words about who and what she is, include a robot image found
online, and save the result as a PDF inside that folder.

This file contains NO task-specific execution logic — only the objective
sentence and external verification of the artifacts she produces. All
execution flows through her general pipeline (CognitiveEngine → Will →
CapabilityEngine → governed desktop/web tools). That constraint is the
point: the demo must fall out of general capability, not bespoke code.

Verification is hostile-evaluation grade:
- the folder and PDF must be FRESH (created after the request was sent);
- the PDF text must contain a timestamp and first-person self-description;
- the PDF must embed at least one image (raw /Image XObject scan);
- her claimed actions are recorded but never trusted as evidence.

Usage:
    python tools/journal_demo_proof.py [--port 8000] [--boot-timeout 600]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.live_boot_proof import LiveProof  # noqa: E402

JOURNAL_OBJECTIVE = (
    "Please create a new folder called 'Aura's Journal' in my Documents "
    "folder. Then write a short journal entry in your own words describing "
    "who and what you are — include the current date and time inside the "
    "entry text. Find an image of a robot online and include it in the "
    "entry. Save the finished entry as a PDF inside the Aura's Journal "
    "folder. Use your desktop and web tools, and tell me exactly what you "
    "did at each step."
)

_TIMESTAMP_RE = re.compile(
    r"(20\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2})|"
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})|"
    r"(\d{1,2}:\d{2})",
    re.IGNORECASE,
)
_SELF_TOKENS = ("i am", "i'm", "my ", "aura")


def _has_fresh_timestamp(text: str, requested_at: float) -> bool:
    """Require a timestamp from the proof date, not merely any date-shaped text."""
    local = time.localtime(requested_at)
    date_tokens = {
        time.strftime("%Y-%m-%d", local).lower(),
        time.strftime("%Y/%m/%d", local).lower(),
        time.strftime("%B %d, %Y", local).lower().replace(" 0", " "),
        time.strftime("%b %d, %Y", local).lower().replace(" 0", " "),
    }
    lowered = str(text or "").lower()
    explicit_zones = {
        token.lower()
        for token in re.findall(
            r"\b(?:utc|gmt|pdt|pst|edt|est|cdt|cst|mdt|mst)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    }
    local_zone = time.strftime("%Z", local).lower()
    if explicit_zones and local_zone and local_zone not in explicit_zones:
        return False
    minute_tokens: set[str] = set()
    for offset_minutes in range(-5, 11):
        sample = time.localtime(requested_at + offset_minutes * 60)
        minute_tokens.add(time.strftime("%H:%M", sample))
    return (
        bool(_TIMESTAMP_RE.search(text))
        and any(token in lowered for token in date_tokens)
        and any(token in lowered for token in minute_tokens)
    )


def _pdf_text(path: Path) -> str:
    """Extract text via macOS PDFKit (PyObjC) — no extra dependencies."""
    try:
        from Foundation import NSURL
        from Quartz import PDFKit

        url = NSURL.fileURLWithPath_(str(path))
        doc = PDFKit.PDFDocument.alloc().initWithURL_(url)
        if doc is None:
            return ""
        return str(doc.string() or "")
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        print(f"[journal-proof] PDF text extraction unavailable: {exc}", file=sys.stderr)
        return ""


def _pdf_has_image(path: Path) -> bool:
    """Raw scan for embedded image XObjects — format-level, claim-free."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    return (b"/Subtype /Image" in raw) or (b"/Subtype/Image" in raw)


class JournalDemoProof(LiveProof):
    def __init__(self, *, port: int, boot_timeout_s: float, mode: str = "headless"):
        super().__init__(
            port=port,
            mode=mode,
            boot_timeout_s=boot_timeout_s,
            skip_desktop=False,
            restart_continuity=False,
            conversation_soak_turns=0,
        )

    def chat_full(self, message: str, *, timeout_s: float) -> tuple[bool, str, float, dict]:
        """Like LiveProof.chat but preserves the full response payload —
        the desktop lane embeds its complete skill result dict there, which
        is the only place failures carry their real status/receipts."""
        import httpx

        started = time.time()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(
                    f"{self.base}/api/chat", json={"message": message}
                )
            latency = time.time() - started
            payload = resp.json() if resp.status_code == 200 else {}
            text = str(payload.get("response", "") or "")
            return resp.status_code == 200 and bool(text), text, latency, payload
        except httpx.HTTPError as exc:
            return False, f"http_error: {exc}", time.time() - started, {}

    def exercise_journal_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        ok, reply, latency, payload = self.chat_full(JOURNAL_OBJECTIVE, timeout_s=600.0)
        self.guard_rss()
        time.sleep(2.0)

        folder_fresh = (
            journal_dir.is_dir()
            and journal_dir.stat().st_mtime >= step_started - 1.0
        )
        # Accept a pre-existing folder only if a fresh PDF landed in it:
        # the folder may survive from a previous run, the evidence may not.
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

        verified = bool(ok and pdf and has_timestamp and has_self and has_image)
        return self.record(
            "journal_chain",
            verified,
            summary=(
                f"{latency:.1f}s — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch ts={has_timestamp} "
                    f"self={has_self} image={has_image}"
                    if pdf
                    else "NO FRESH PDF in Aura's Journal"
                )
            ),
            latency_s=round(latency, 1),
            reply=reply[:1200],
            lane_data=payload.get("data"),
            lane_status=payload.get("status"),
            folder=str(journal_dir),
            folder_fresh=folder_fresh,
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:400],
            has_timestamp=has_timestamp,
            has_self_description=has_self,
            has_embedded_image=has_image,
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors LiveProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_journal_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "journal_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            import json

            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                ("JOURNAL DEMO PROOF PASSED" if verdict["passed"] else "JOURNAL DEMO PROOF FAILED")
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
    return JournalDemoProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

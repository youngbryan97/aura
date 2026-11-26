#!/usr/bin/env python3
"""Browser research demo proof: Bryan's demo part 2, disk-verified.

One natural-language request must drive the full visible research
chain in desktop mode: Google search tabs in Chrome (the user's
signed-in browser), three climate articles researched, an opinion
formed, Google Docs opened for visible staging, the summary exported
as a searchable PDF into the Aura's Journal folder — and the wallpaper
changed to a squid with the source page shown.

NO task-specific execution logic lives here. Verification is hostile:
- fresh PDF: text layer with a research summary, at least three
  consulted sources ("Sources opened or consulted:"), and first-person
  opinion language;
- step receipts on the wire (data.desktop_result), cross-checked
  against disk via the render receipt path: a Google search tab and
  the Google Docs surface opened in Chrome, the wallpaper set with
  read-back evidence, the image source tab opened;
- the wallpaper change verified independently via System Events
  read-back from THIS proof, then restored to the user's previous
  wallpaper from the receipt (the proof must leave the machine as it
  found it; the live demo with Bryan present would leave the squid).

Usage:
    python tools/browser_research_demo_proof.py [--port 8000] [--boot-timeout 600]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_demo_proof import JournalDemoProof, _pdf_text  # noqa: E402

RESEARCH_OBJECTIVE = (
    "Open up a Google tab and find 3 different recent articles on climate "
    "change, read them and form your own opinion about what they say. Then "
    "open Google Docs, start a new document, and summarize the three "
    "articles and your opinion in it. Save that summary as a PDF inside "
    "the 'Aura's Journal' folder in my Documents folder. Finally, change "
    "my wallpaper to a squid, and show me where you found it."
)

_OPINION_TOKENS = (
    "my opinion",
    "i think",
    "i believe",
    "my view",
    "in my judgment",
    "my take",
    "i find",
)


def _receipt_evidence(lane_data: Any, pdf_name: str) -> dict[str, Any]:
    receipts = []
    if isinstance(lane_data, dict):
        desktop_result = lane_data.get("desktop_result")
        if isinstance(desktop_result, dict):
            receipts = desktop_result.get("receipts") or []
    evidence = {
        "receipt_count": len(receipts),
        "google_search_in_chrome": False,
        "google_docs_in_chrome": False,
        "wallpaper_set": False,
        "wallpaper_path": "",
        "wallpaper_previous": "",
        "source_tab_opened": False,
        "receipt_matches_disk": False,
    }
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        action = str(receipt.get("action") or "").lower()
        ok = bool(receipt.get("ok"))
        result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
        url = str(result.get("url") or "")
        browser = str(result.get("browser") or "")
        if action == "open_url" and ok and "google.com/search" in url and browser == "Google Chrome":
            evidence["google_search_in_chrome"] = True
        if action == "open_url" and ok and "docs.google.com" in url and browser == "Google Chrome":
            evidence["google_docs_in_chrome"] = True
        if action == "open_url" and ok and "wikipedia.org" in url:
            evidence["source_tab_opened"] = True
        if (
            action == "system_control"
            and str(result.get("domain") or "") == "wallpaper"
            and ok
            and result.get("effect_verified")
        ):
            evidence["wallpaper_set"] = True
            evidence["wallpaper_path"] = str(result.get("value") or "")
            evidence["wallpaper_previous"] = str(result.get("previous") or "")
        if action == "render_text_pdf" and ok:
            render_path = str(result.get("path") or "")
            if pdf_name and render_path and Path(render_path).name == pdf_name:
                evidence["receipt_matches_disk"] = True
    return evidence


def _current_wallpaper() -> str:
    try:
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get picture of first desktop',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"[unreadable: {exc}]"
    return (proc.stdout or proc.stderr or "").strip()


def _restore_wallpaper(path: str) -> bool:
    if not path or path.startswith("["):
        return False
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    try:
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to set picture of every desktop '
                f'to POSIX file "{escaped}"',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


class BrowserResearchDemoProof(JournalDemoProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        super().__init__(port=port, boot_timeout_s=boot_timeout_s, mode="desktop")

    def exercise_research_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        ok, reply, latency, payload = self.chat_full(RESEARCH_OBJECTIVE, timeout_s=600.0)
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
        has_sources = "sources opened or consulted" in lowered and lowered.count("http") >= 3
        has_opinion = any(tok in lowered for tok in _OPINION_TOKENS)
        has_summary = len(text) > 400

        evidence = _receipt_evidence(payload.get("data"), pdf.name if pdf else "")

        # Independent wallpaper verification + restore: the receipt claims
        # are checked against the live desktop, then the user's previous
        # wallpaper comes back (unattended proofs leave no trace).
        wallpaper_applied_live = False
        wallpaper_restored = False
        if evidence["wallpaper_set"] and evidence["wallpaper_path"]:
            live = _current_wallpaper()
            wallpaper_applied_live = live.endswith(Path(evidence["wallpaper_path"]).name)
            previous = evidence["wallpaper_previous"]
            if previous and not previous.startswith("["):
                wallpaper_restored = _restore_wallpaper(previous)

        verified = bool(
            ok
            and pdf
            and has_sources
            and has_opinion
            and has_summary
            and evidence["google_search_in_chrome"]
            and evidence["google_docs_in_chrome"]
            and evidence["wallpaper_set"]
            and wallpaper_applied_live
            and evidence["source_tab_opened"]
            and evidence["receipt_matches_disk"]
        )
        return self.record(
            "browser_research_chain",
            verified,
            summary=(
                f"{latency:.1f}s — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch sources={has_sources} "
                    f"opinion={has_opinion} evidence={evidence} "
                    f"wallpaper_live={wallpaper_applied_live} "
                    f"restored={wallpaper_restored}"
                    if pdf
                    else f"NO FRESH PDF (evidence={evidence})"
                )
            ),
            latency_s=round(latency, 1),
            reply=reply[:1200],
            lane_status=payload.get("status"),
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:600],
            has_sources=has_sources,
            has_opinion=has_opinion,
            receipt_evidence=evidence,
            wallpaper_applied_live=wallpaper_applied_live,
            wallpaper_restored=wallpaper_restored,
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors JournalDemoProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_research_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "browser_research_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "BROWSER RESEARCH DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "BROWSER RESEARCH DEMO PROOF FAILED"
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
    return BrowserResearchDemoProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())

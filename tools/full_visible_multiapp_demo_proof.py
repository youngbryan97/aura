#!/usr/bin/env python3
"""Full visible multi-app desktop proof.

One natural-language objective enters Aura through the live desktop chat path.
The proof harness does not execute the task. It only boots Aura, sends the
objective, and verifies external effects:

- Notes was visibly opened and written to through general desktop receipts.
- A fresh self-description PDF with timestamp and image landed in Aura's Journal.
- Chrome search/source tabs and Google Docs opened through general URL receipts.
- A fresh research-summary PDF includes source URLs and opinion language.
- Wallpaper changed with read-back evidence, source page opened, and wallpaper
  was restored after proof.

This is a verifier for general computer-use capability, not a task-specific
executor.
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

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from tools.journal_demo_proof import (  # noqa: E402
    _SELF_TOKENS,
    JournalDemoProof,
    _has_fresh_timestamp,
    _pdf_has_image,
    _pdf_text,
)


FULL_VISIBLE_OBJECTIVE = (
    "Please open my Notes app and write a paragraph in your own words about "
    "who and what you are, including the current date and time. Create a folder "
    "called 'Aura's Journal' in my Documents folder and export that note as a "
    "PDF there. Then open Chrome, find three recent articles about climate "
    "change, keep the source pages available, open a new Google Doc, and write "
    "a composite summary of all three articles plus your own opinion. Export "
    "that summary as a PDF into the same Aura's Journal folder. Finally, find "
    "a cool image of an eagle online, set it as my wallpaper, and show the "
    "source page you found it from. I want the work to happen visibly through "
    "your normal governed desktop tools."
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


def _fresh_pdfs(folder: Path, started: float) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.glob("*.pdf") if path.stat().st_mtime >= started - 1.0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _receipt_list(lane_data: Any) -> list[dict[str, Any]]:
    if not isinstance(lane_data, dict):
        return []
    desktop_result = lane_data.get("desktop_result")
    if not isinstance(desktop_result, dict):
        return []
    return [item for item in desktop_result.get("receipts") or [] if isinstance(item, dict)]


def _receipt_result(receipt: dict[str, Any]) -> dict[str, Any]:
    result = receipt.get("result")
    return result if isinstance(result, dict) else {}


def _receipt_evidence(receipts: list[dict[str, Any]], pdf_names: set[str]) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "receipt_count": len(receipts),
        "notes_opened": False,
        "paste_dispatched": False,
        "paste_frontmost_notes": False,
        "google_search_in_chrome": False,
        "google_docs_in_chrome": False,
        "source_tabs_opened": 0,
        "wallpaper_set": False,
        "wallpaper_path": "",
        "wallpaper_previous": "",
        "rendered_pdf_names": [],
        "rendered_pdf_count": 0,
        "rendered_pdf_matches_disk": 0,
    }
    for receipt in receipts:
        action = str(receipt.get("action") or "").lower()
        ok = bool(receipt.get("ok"))
        result = _receipt_result(receipt)
        url = str(result.get("url") or "")
        browser = str(result.get("browser") or "")
        if action == "open_app" and ok and "notes" in str(result.get("opened") or "").lower():
            evidence["notes_opened"] = True
        if action == "hotkey" and ok and "v" in str(result.get("hotkey") or "").lower():
            evidence["paste_dispatched"] = True
            before = str(result.get("frontmost_app_before") or "").lower()
            after = str(result.get("frontmost_app_after") or "").lower()
            evidence["paste_frontmost_notes"] = evidence["paste_frontmost_notes"] or ("notes" in before or "notes" in after)
        if action == "open_url" and ok and "google.com/search" in url and browser == "Google Chrome":
            evidence["google_search_in_chrome"] = True
        if action == "open_url" and ok and "docs.google.com" in url and browser == "Google Chrome":
            evidence["google_docs_in_chrome"] = True
        if action == "open_url" and ok and url.startswith("http") and not any(
            marker in url for marker in ("google.com/search", "docs.google.com")
        ):
            evidence["source_tabs_opened"] += 1
        if action == "system_control" and ok and str(result.get("domain") or "") == "wallpaper":
            evidence["wallpaper_set"] = bool(result.get("effect_verified"))
            evidence["wallpaper_path"] = str(result.get("value") or "")
            evidence["wallpaper_previous"] = str(result.get("previous") or "")
        if action == "render_text_pdf" and ok:
            rendered_name = Path(str(result.get("path") or "")).name
            if rendered_name:
                evidence["rendered_pdf_names"].append(rendered_name)
                evidence["rendered_pdf_count"] += 1
                if rendered_name in pdf_names:
                    evidence["rendered_pdf_matches_disk"] += 1
    return evidence


def _classify_pdfs(pdfs: list[Path], started: float) -> dict[str, Any]:
    note: dict[str, Any] = {"found": False}
    research: dict[str, Any] = {"found": False}
    for path in pdfs:
        text = _pdf_text(path)
        lowered = text.lower()
        timestamp = _has_fresh_timestamp(text, started)
        self_desc = any(tok in lowered for tok in _SELF_TOKENS) and len(text) > 120
        has_image = _pdf_has_image(path)
        has_sources = "sources opened or consulted" in lowered and lowered.count("http") >= 3
        has_opinion = any(tok in lowered for tok in _OPINION_TOKENS)
        has_summary = len(text) > 400
        record = {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size,
            "text_chars": len(text),
            "text_head": text[:500],
            "timestamp": timestamp,
            "self_description": self_desc,
            "image": has_image,
            "sources": has_sources,
            "opinion": has_opinion,
            "summary": has_summary,
        }
        if not note.get("found") and timestamp and self_desc:
            note = {"found": True, **record}
        if not research.get("found") and has_sources and has_opinion and has_summary:
            research = {"found": True, **record}
    return {"note": note, "research": research}


def _current_wallpaper() -> str:
    result = get_subprocess_gateway().run(
        ["osascript", "-e", 'tell application "System Events" to get picture of first desktop'],
        capture_output=True,
        timeout=15,
        read_only=True,
        offline_tooling=True,
        source="proof_tooling:full_visible_multiapp.current_wallpaper",
        accelerator_capability="none",
    )
    return (result.stdout or result.stderr or "").strip()


def _restore_wallpaper(path: str) -> bool:
    if not path or path.startswith("["):
        return False
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    result = get_subprocess_gateway().run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to set picture of every desktop '
            f'to POSIX file "{escaped}"',
        ],
        capture_output=True,
        timeout=15,
        offline_tooling=True,
        source="proof_tooling:full_visible_multiapp.restore_wallpaper",
        accelerator_capability="none",
    )
    return result.returncode == 0


class FullVisibleMultiappDemoProof(JournalDemoProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        super().__init__(port=port, boot_timeout_s=boot_timeout_s, mode="desktop")

    def exercise_full_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        ok, reply, latency, payload = self.chat_full(FULL_VISIBLE_OBJECTIVE, timeout_s=900.0)
        self.guard_rss()
        time.sleep(2.0)

        pdfs = _fresh_pdfs(journal_dir, step_started)
        pdf_status = _classify_pdfs(pdfs, step_started)
        lane_data = payload.get("data") if isinstance(payload, dict) else {}
        receipts = _receipt_list(lane_data)
        evidence = _receipt_evidence(receipts, {path.name for path in pdfs})

        wallpaper_applied_live = False
        wallpaper_restored = False
        if evidence["wallpaper_set"] and evidence["wallpaper_path"]:
            live = _current_wallpaper()
            wallpaper_applied_live = live.endswith(Path(evidence["wallpaper_path"]).name)
            wallpaper_restored = _restore_wallpaper(str(evidence["wallpaper_previous"] or ""))

        verified = bool(
            ok
            and pdf_status["note"].get("found")
            and pdf_status["research"].get("found")
            and evidence["notes_opened"]
            and evidence["paste_dispatched"]
            and evidence["paste_frontmost_notes"]
            and evidence["google_search_in_chrome"]
            and evidence["google_docs_in_chrome"]
            and evidence["source_tabs_opened"] >= 1
            and evidence["wallpaper_set"]
            and wallpaper_applied_live
            and evidence["rendered_pdf_count"] >= 2
            and evidence["rendered_pdf_matches_disk"] >= 2
        )
        return self.record(
            "full_visible_multiapp_chain",
            verified,
            summary=(
                f"{latency:.1f}s — pdfs={len(pdfs)} note={pdf_status['note'].get('found')} "
                f"research={pdf_status['research'].get('found')} evidence={evidence} "
                f"wallpaper_live={wallpaper_applied_live} restored={wallpaper_restored}"
            ),
            latency_s=round(latency, 1),
            reply=reply[:1600],
            lane_status=payload.get("status") if isinstance(payload, dict) else "",
            evidence=evidence,
            pdf_status=pdf_status,
            pdf_paths=[str(path) for path in pdfs],
            wallpaper_applied_live=wallpaper_applied_live,
            wallpaper_restored=wallpaper_restored,
        )

    def run(self) -> int:
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_full_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "full_visible_multiapp_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "FULL VISIBLE MULTIAPP DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "FULL VISIBLE MULTIAPP DEMO PROOF FAILED"
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
    return FullVisibleMultiappDemoProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    raise SystemExit(main())

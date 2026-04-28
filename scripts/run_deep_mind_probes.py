#!/usr/bin/env python3
"""Run Aura's deep agency/consciousness probe suite against a live chat API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SESSION_DIR = PROJECT_ROOT / "aura" / "knowledge" / "research-sessions"

from core.evaluation.deep_mind_probe import DEEP_MIND_PROBES, evaluate_deep_probe_response


def _post_chat(base_url: str, message: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response") or body.get("content") or body.get("text") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit probe count for quick smoke runs.")
    parser.add_argument("--output", default="", help="Write the probe session JSON to this path.")
    parser.add_argument("--no-record", action="store_true", help="Do not write a research-session JSON artifact.")
    args = parser.parse_args()

    probes = list(DEEP_MIND_PROBES)
    if args.limit > 0:
        probes = probes[: args.limit]

    started_at = time.time()
    results = []
    for probe in probes:
        try:
            response = _post_chat(args.base_url, probe.question, args.timeout)
            evaluation = evaluate_deep_probe_response(probe, response)
            results.append({
                "probe": probe.id,
                "question": probe.question,
                "response": response,
                "evaluation": evaluation.as_dict(),
            })
            status = "PASS" if evaluation.passed else "FAIL"
            print(f"{status} {probe.id}: score={evaluation.score} issues={list(evaluation.issues)}")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            results.append({
                "probe": probe.id,
                "question": probe.question,
                "response": "",
                "evaluation": {
                    "probe_id": probe.id,
                    "passed": False,
                    "score": 0.0,
                    "issues": [f"request_failed:{type(exc).__name__}"],
                    "strengths": [],
                },
            })
            print(f"FAIL {probe.id}: request failed: {exc}", file=sys.stderr)

    passed = sum(1 for item in results if item["evaluation"]["passed"])
    payload = {"passed": passed, "total": len(results), "results": results}
    print(json.dumps(payload, indent=2))
    if not args.no_record:
        completed_at = time.time()
        output_path = Path(args.output) if args.output else DEFAULT_SESSION_DIR / f"deep-mind-probes-{int(completed_at)}.json"
        get_task_tracker().create_task(get_storage_gateway().create_dir(output_path.parent, cause='main'))
        session_payload = {
            "phase": "complete" if passed == len(results) else "failed",
            "result": {
                "item_title": "Deep agency/sentience/consciousness live probe",
                "started_at": started_at,
                "completed_at": completed_at,
                "decision": {
                    "title": "Deep agency/sentience/consciousness live probe",
                    "category": "Evaluation",
                    "url": args.base_url,
                    "reason": "live headless API probe suite",
                },
                "sources_engaged": [args.base_url],
                "priority_levels_engaged": [],
                "depth_passed": passed == len(results),
                "depth_score": round(passed / max(1, len(results)), 3),
                "depth_failures": [
                    f"{item['probe']}:{','.join(item['evaluation'].get('issues', []))}"
                    for item in results
                    if not item["evaluation"].get("passed")
                ],
                "persist_receipt": None,
                "inference_failures": len(results) - passed,
                "error": None if passed == len(results) else "one or more deep probes failed",
                "session_id": output_path.stem,
            },
            "probe_batch": payload,
        }
        output_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
        print(f"recorded_session={output_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

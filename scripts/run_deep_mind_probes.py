#!/usr/bin/env python3
"""Run Aura's deep agency/consciousness probe suite against a live chat API."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

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
    args = parser.parse_args()

    probes = list(DEEP_MIND_PROBES)
    if args.limit > 0:
        probes = probes[: args.limit]

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
    print(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

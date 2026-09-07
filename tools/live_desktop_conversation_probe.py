#!/usr/bin/env python3
"""Probe Aura's real live desktop conversation lane.

This attaches to a running Aura server and sends a short multi-turn script
through `/api/chat`, the same endpoint used by the desktop UI. It scores the
actual replies for continuity, non-assistant collapse, grounded self-modeling,
novel thought, and bounded consciousness/tool claims.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.evaluation.live_conversation_probe import (  # noqa: E402
    DEFAULT_LIVE_CONVERSATION_SCRIPT,
    live_conversation_reply_rubric,
    score_live_conversation_transcript,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _extract_response(payload: dict[str, Any]) -> str:
    return str(
        payload.get("response")
        or payload.get("reply")
        or payload.get("message")
        or payload.get("text")
        or ""
    ).strip()


def _post_chat(
    client: httpx.Client,
    base_url: str,
    message: str,
    *,
    session_id: str,
    timeout_s: float,
) -> tuple[bool, str, dict[str, Any], float]:
    started = time.monotonic()
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"message": message, "session_id": session_id},
            headers={"X-Idempotency-Key": f"{session_id}:{uuid.uuid4().hex}"},
            timeout=timeout_s,
        )
        latency = time.monotonic() - started
        if response.status_code != 200:
            return False, f"http {response.status_code}: {response.text[:500]}", {}, latency
        payload = response.json()
        text = _extract_response(payload)
        return bool(text), text, payload, latency
    except (httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}", {}, time.monotonic() - started


def run_probe(
    *,
    base_url: str,
    out: Path,
    session_id: str,
    timeout_s: float,
) -> int:
    headers: dict[str, str] = {}
    token = os.environ.get("AURA_API_TOKEN", "").strip()
    if token:
        headers["X-Api-Token"] = token
    headers["X-Aura-Surface"] = "desktop-ui"
    headers["X-Aura-Desktop-Request"] = "true"
    headers["X-Aura-Require-CognitiveEngine"] = "required"
    headers["X-Aura-Live-Conversation-Probe"] = "true"

    out.mkdir(parents=True, exist_ok=True)
    transcript: list[dict[str, Any]] = []
    responses: dict[str, str] = {}

    with httpx.Client(headers=headers, timeout=timeout_s) as client:
        for turn in DEFAULT_LIVE_CONVERSATION_SCRIPT:
            ok, text, payload, latency = _post_chat(
                client,
                base_url,
                turn.prompt,
                session_id=session_id,
                timeout_s=timeout_s,
            )
            responses[turn.id] = text
            transcript.append(
                {
                    "turn_id": turn.id,
                    "kind": turn.kind,
                    "prompt": turn.prompt,
                    "ok": ok,
                    "latency_s": round(latency, 3),
                    "response": text,
                    "status": payload.get("status") if isinstance(payload, dict) else None,
                    "response_confidence": payload.get("response_confidence") if isinstance(payload, dict) else None,
                    "conversation_lane": payload.get("conversation_lane") if isinstance(payload, dict) else None,
                    "live_turn_contract": payload.get("live_turn_contract") if isinstance(payload, dict) else None,
                    "timestamp": _utc_now(),
                }
            )
            if not ok:
                # A client timeout does not cancel the durable server turn.
                # Do not enqueue the rest of the script behind unknown work.
                break

    scorecard = score_live_conversation_transcript(responses)
    contract_issues: list[str] = []
    if len(transcript) != len(DEFAULT_LIVE_CONVERSATION_SCRIPT):
        contract_issues.append("script_incomplete")
    for row in transcript:
        contract = dict(row.get("live_turn_contract") or {})
        confidence = str(row.get("response_confidence") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        if not contract.get("desktop_cognitive_engine_required"):
            contract_issues.append(f"{row['turn_id']}:desktop_cognitive_engine_not_required")
        if not contract.get("engine_think_invoked"):
            contract_issues.append(f"{row['turn_id']}:engine_think_not_invoked")
        if contract.get("legacy_fallback_used"):
            contract_issues.append(f"{row['turn_id']}:legacy_fallback_used")
        if not contract.get("full_mind_path"):
            contract_issues.append(f"{row['turn_id']}:full_mind_path_false")
        if not contract.get("required_subsystems_ok"):
            contract_issues.append(f"{row['turn_id']}:required_subsystems_not_ok")
        if confidence != "high":
            contract_issues.append(f"{row['turn_id']}:confidence_{confidence or 'missing'}")
        if status.startswith("desktop_cognitive_engine_") and confidence != "high":
            contract_issues.append(f"{row['turn_id']}:bounded_or_failed_cognitive_repair")

    passed = scorecard.passed and not contract_issues
    artifact = {
        "created_at": _utc_now(),
        "base_url": base_url,
        "session_id": session_id,
        "rubric": live_conversation_reply_rubric(),
        "transcript": transcript,
        "scorecard": scorecard.as_dict(),
        "contract_issues": contract_issues,
        "passed": passed,
    }
    (out / "LIVE_CONVERSATION_PROBE.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "LIVE_CONVERSATION_TRANSCRIPT.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in transcript) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(scorecard.as_dict(), indent=2, sort_keys=True))
    if contract_issues:
        print(json.dumps({"contract_issues": contract_issues}, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "current" / "live_desktop_conversation_probe",
    )
    parser.add_argument("--session-id", default=f"live-conversation-probe-{uuid.uuid4().hex[:10]}")
    args = parser.parse_args()
    return run_probe(
        base_url=str(args.base_url),
        out=args.out,
        session_id=str(args.session_id),
        timeout_s=float(args.timeout),
    )


if __name__ == "__main__":
    raise SystemExit(main())

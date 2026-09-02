"""interface/routes/chat_turn_evidence.py — what a turn leaves behind.

Eighteen helpers lifted out of ``interface/routes/chat.py``. One concern:
the receipts, provenance and artifacts a turn produces, and the checks that
a reply claiming an action can point at the evidence for it.

They reference nothing else in the route module. ``chat.py`` re-exports all
eighteen, so nothing that imported them from there has to change.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.receipts import digest_output_content
from interface.routes import chat_capability_inventory as _chat_capability_inventory
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

logger = logging.getLogger("Aura.Chat")


def _benchmark_prompt_requests_fenced_artifact(prompt: str, fence: str) -> bool:
    prompt_l = str(prompt or "").lower()
    fence_l = fence.lower()
    index = prompt_l.rfind(fence_l)
    if index < 0:
        return False
    window = prompt_l[max(0, index - 220) : index + 220]
    return any(
        marker in window
        for marker in (
            "return",
            "respond",
            "response in this format",
            "format:",
            "write the code",
            "complete fixed",
        )
    )


async def _emit_chat_output_receipt(
    reply_text: str,
    *,
    cause: str,
    origin: str = "api",
    target: str = "primary",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record direct chat replies as durable output receipts."""
    try:
        from core.runtime.executors import run_durable_receipt_io
        from core.runtime.receipts import OutputReceipt, get_receipt_store

        digest = digest_output_content(reply_text)
        receipt = OutputReceipt(
            cause=str(cause or "chat_response"),
            origin=str(origin or "api"),
            target=str(target or "primary"),
            digest=digest,
            metadata=dict(metadata or {}),
        )
        await run_durable_receipt_io(
            get_receipt_store().emit,
            receipt,
            timeout_s=10.0,
            label="chat_output_receipt",
        )
        if str(target or "primary") == "primary":
            try:
                from core.epistemics.epistemic_reach import (
                    acknowledge_epistemic_correction_delivery,
                )

                acknowledge_epistemic_correction_delivery(reply_text)
            except _CHAT_RECOVERABLE_ERRORS as correction_exc:
                record_degradation("chat", correction_exc)
                logger.debug(
                    "Epistemic correction delivery acknowledgement skipped: %s",
                    correction_exc,
                )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Chat output receipt emit skipped: %s", exc)


def _canonical_memory_state_evidence_missing_from_reply(
    canonical_memory_state_evidence: str,
    reply_text: str,
) -> bool:
    """Return True when an inline canonical memory block was not reflected."""

    evidence = str(canonical_memory_state_evidence or "")
    reply = str(reply_text or "").lower()
    if not evidence.strip() or not reply.strip():
        return True

    status_match = re.search(r"^\s*status\s*=\s*([a-zA-Z0-9_:-]+)", evidence, re.MULTILINE)
    status = status_match.group(1).strip() if status_match else ""
    quoted = re.search(r'"([^"]{1,240})"', evidence)
    expected_content = quoted.group(1).strip() if quoted else ""

    if status in {
        "session_memory_pin",
        "session_memory_pin_transient",
        "session_memory_recall",
        "session_memory_context_recall",
    }:
        if not expected_content:
            return True
        return expected_content.lower() not in reply
    return False


def _canonical_memory_state_evidence_from_tuple(
    memory_state_evidence: tuple[str, str] | None,
) -> str:
    """Convert the canonical memory/state tuple into the inline evidence block body."""

    if not memory_state_evidence:
        return ""
    memory_reply, memory_status = memory_state_evidence
    return (f"status={str(memory_status or '').strip()}\n{str(memory_reply or '').strip()}").strip()


def _extract_canonical_memory_state_evidence_block(effective_user_message: str) -> str:
    """Extract the canonical memory/state evidence block carried into CognitiveEngine."""

    text = str(effective_user_message or "")
    if "[CANONICAL MEMORY STATE EVIDENCE]" not in text:
        return ""
    match = re.search(
        r"\[CANONICAL MEMORY STATE EVIDENCE\]\s*(.*?)\s*\[END CANONICAL MEMORY STATE EVIDENCE\]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:2400]
    return ""


def _context_challenge_repair_has_evidence(reply_text: str) -> bool:
    reply = _chat_memory_state._normalize_user_message(reply_text)
    return bool(
        reply
        and any(
            marker in reply
            for marker in (
                "last completed exchange",
                "last completed thing",
                "grounded lead-in",
                "vague referent",
            )
        )
    )


async def _resolve_answer_provenance_projection(
    user_message: str,
    *,
    session_id: str = "",
) -> str:
    """Answer a source follow-up from the evidence bound to that answer."""

    from core.conversation.answer_provenance import (
        answer_provenance_reply,
        provenance_grounding_json,
        select_prior_answer_provenance,
    )

    recent_exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=6,
        allow_cross_session=False,
    )
    provenance = select_prior_answer_provenance(user_message, recent_exchanges)
    if provenance is None:
        return ""
    from core.conversation.turn_evidence_custody import record_turn_grounding

    record_turn_grounding(provenance_grounding_json(provenance))
    return answer_provenance_reply(provenance)


def _collect_recent_traceability_event_sync() -> tuple[dict[str, Any] | None, str]:
    access_errors = 0
    saw_private_only = False

    try:
        from core.runtime.receipts import get_receipt_store

        store = get_receipt_store()
        all_recent = store.query_recent(limit=24)
        if not all_recent:
            store.reload_from_disk()
            all_recent = store.query_recent(limit=24)

        safe_kinds = [
            "output",
            "tool_execution",
            "state_mutation",
            "computer_use",
            "autonomy",
            "self_repair",
        ]
        safe_recent = store.query_recent(kinds=safe_kinds, limit=24)
        for receipt in reversed(safe_recent):
            kind = str(getattr(receipt, "kind", "") or "")
            if kind == "output" and str(getattr(receipt, "target", "") or "") != "primary":
                continue

            event: dict[str, Any] = {
                "timestamp": float(getattr(receipt, "created_at", 0.0) or 0.0),
                "event_id": str(getattr(receipt, "receipt_id", "") or ""),
                "kind": kind,
                "subsystem": "",
                "action": "",
                "result": "",
                "changed_future_behavior": False,
            }
            if kind == "output":
                event["subsystem"] = f"Output.{str(getattr(receipt, 'origin', '') or 'unknown')}"
                event["action"] = (
                    f"emitted {str(getattr(receipt, 'target', '') or 'primary')} response"
                )
                event["result"] = f"digest={str(getattr(receipt, 'digest', '') or 'unknown')}"
            elif kind == "tool_execution":
                tool_name = str(getattr(receipt, "tool", "") or "unknown")
                event["subsystem"] = f"Tool.{tool_name}"
                event["action"] = f"executed tool {tool_name}"
                event["result"] = f"status={str(getattr(receipt, 'status', '') or 'unknown')}"
            elif kind == "state_mutation":
                domain = str(getattr(receipt, "domain", "") or "state")
                key = str(getattr(receipt, "key", "") or "unknown")
                event["subsystem"] = f"State.{domain}"
                event["action"] = f"mutated {domain}.{key}"
                event["result"] = f"schema_v={int(getattr(receipt, 'schema_version', 1) or 1)}"
                event["changed_future_behavior"] = True
            elif kind == "computer_use":
                action_kind = str(getattr(receipt, "action_kind", "") or "act")
                target = str(getattr(receipt, "target", "") or "screen")
                event["subsystem"] = "ComputerUse"
                event["action"] = f"{action_kind} {target}".strip()
                event["result"] = f"verified={bool(getattr(receipt, 'verifier_result', False))}"
            elif kind == "autonomy":
                proposed = str(getattr(receipt, "proposed_action", "") or "autonomous step")
                event["subsystem"] = "Autonomy"
                event["action"] = proposed
                event["result"] = f"level={int(getattr(receipt, 'autonomy_level', 0) or 0)}"
                event["changed_future_behavior"] = True
            elif kind == "self_repair":
                target_module = str(getattr(receipt, "target_module", "") or "unknown")
                event["subsystem"] = "SelfRepair"
                event["action"] = f"self-repair on {target_module}"
                event["result"] = f"rolled_back={bool(getattr(receipt, 'rolled_back', False))}"
                event["changed_future_behavior"] = True
            return event, ""

        if all_recent:
            saw_private_only = True
    except _CHAT_RECOVERABLE_ERRORS:
        access_errors += 1

    try:
        from core.consciousness.authority_audit import get_audit

        audit = get_audit()
        effects = audit.get_recent_effects(12)
        for effect in reversed(effects):
            if str(effect.get("effect_type") or "") != "response":
                continue
            return {
                "timestamp": float(effect.get("timestamp") or 0.0),
                "event_id": str(effect.get("receipt_id") or ""),
                "kind": "authority_effect",
                "subsystem": str(effect.get("source") or "AuthorityAudit"),
                "action": f"emitted {str(effect.get('effect_type') or 'effect')}",
                "result": "authorized" if bool(effect.get("matched")) else "unmatched",
                "changed_future_behavior": False,
            }, ""
    except _CHAT_RECOVERABLE_ERRORS:
        access_errors += 1

    try:
        from core.somatic.motor_cortex import get_motor_cortex

        receipts = get_motor_cortex().get_recent_receipts(12)
        for receipt in reversed(receipts):
            return {
                "timestamp": float(receipt.get("timestamp") or 0.0),
                "event_id": str(receipt.get("receipt_id") or ""),
                "kind": "motor_receipt",
                "subsystem": f"MotorCortex.{str(receipt.get('handler') or 'unknown')}",
                "action": f"executed {str(receipt.get('reflex_class') or 'reflex')}",
                "result": str(
                    receipt.get("summary") or f"success={bool(receipt.get('success', False))}"
                ),
                "changed_future_behavior": False,
            }, ""
    except _CHAT_RECOVERABLE_ERRORS:
        access_errors += 1

    if saw_private_only:
        return None, "governance rule blocks disclosure"
    if access_errors >= 3:
        return None, "do not have access"
    return None, "the data does not exist"


def _prime_requested_output_contract_trace(
    trace: dict[str, Any],
    *,
    user_message: str,
) -> None:
    """Bind the user-authored contract before any early return can occur."""

    try:
        from core.conversation.response_reliability import requested_output_contract

        contract = requested_output_contract(user_message)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("chat.output_contract_preflight", exc)
        trace.update(
            {
                "final_requested_output_contract_evaluated": False,
                "final_requested_output_contract_required": None,
                "final_requested_output_contract_kind": "unknown",
                "final_requested_output_contract_satisfied": False,
                "final_requested_output_contract_reasons": [
                    f"preflight_error:{type(exc).__name__}"
                ],
            }
        )
        return

    if not contract.constrained:
        trace.update(
            {
                "requested_output_contract": {},
                "final_requested_output_contract_evaluated": True,
                "final_requested_output_contract_required": False,
                "final_requested_output_contract_kind": str(
                    getattr(contract, "kind", "none") or "none"
                ),
                "final_requested_output_contract_satisfied": True,
                "final_requested_output_contract_reasons": [],
            }
        )
        return

    trace.update(
        {
            "requested_output_contract": contract.as_dict(),
            "final_requested_output_contract_evaluated": False,
            "final_requested_output_contract_required": True,
            "final_requested_output_contract_kind": str(contract.kind or "unknown"),
            "final_requested_output_contract_satisfied": False,
            "final_requested_output_contract_reasons": ["evaluation_not_completed"],
        }
    )


def _recent_action_receipts(user_message: str) -> str:
    """Her real action receipts, rendered as an answer, when asked what she did."""
    try:
        from core.brain.recent_actions import (
            asks_what_she_recently_did,
            recent_actions_answer,
        )

        if not asks_what_she_recently_did(user_message):
            return ""
        return str(recent_actions_answer() or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="answered without recent-action receipts",
        )
        return ""


def _worker_receipt_transaction_id(receipt: Any, response_text: Any) -> str:
    """Return the exact MLX generation identity attested by the parent.

    A route-created UUID proves only that the route ran.  Protected foreground
    delivery needs to prove which resident-worker request authored the bytes it
    is about to serve.  The MLX parent binds that identity after IPC, including
    an exact request-id match and the worker generation that produced it.
    """

    if not isinstance(receipt, dict):
        return ""
    authored_text = str(response_text or "").strip()
    if not authored_text:
        return ""
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        return ""
    request_id = str(provenance.get("request_id") or "").strip()
    worker_boot_id = str(provenance.get("worker_boot_id") or "").strip()
    try:
        worker_generation = int(provenance.get("worker_generation") or 0)
        request_seq = int(provenance.get("request_seq") or 0)
    except (TypeError, ValueError):
        return ""
    if not (
        provenance.get("claims") == "worker_attested"
        and provenance.get("request_id_matches_active") is True
        and provenance.get("worker_identity_attested") is True
        and request_id
        and worker_boot_id
        and worker_generation > 0
        and request_seq > 0
    ):
        return ""
    identity = json.dumps(
        {
            "request_id": request_id,
            "request_seq": request_seq,
            "response_sha256": hashlib.sha256(
                authored_text.encode("utf-8")
            ).hexdigest(),
            "worker_boot_id": worker_boot_id,
            "worker_generation": worker_generation,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "mlx-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _build_explicit_local_file_artifact(user_message: str, path: str) -> str | None:
    text = str(user_message or "").strip()
    lowered = text.lower()
    suffix = Path(path).suffix.lower()
    generated_at = _chat_preflight._utc_now_iso()
    if suffix == ".html":
        if "snake" in lowered and any(token in lowered for token in ("game", "playable", "snake")):
            try:
                from core.cognitive.state_machine import StateMachine

                return StateMachine._snake_html_template()
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                record_degradation("chat.explicit_local_file_objective", exc)
                return (
                    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                    "<title>Aura Snake</title></head><body>"
                    "<canvas id='board' width='320' height='320'></canvas>"
                    "<p>Score: <span id='score'>0</span></p><script>"
                    "const canvas=document.getElementById('board');"
                    "const ctx=canvas.getContext('2d');let score=0;"
                    "function tick(){ctx.fillRect(0,0,320,320);requestAnimationFrame(tick)};"
                    "document.addEventListener('keydown',()=>{score+=1;document.getElementById('score').textContent=score;});"
                    "tick();</script></body></html>"
                )
        title = "Aura Generated Page"
        title_match = re.search(
            r"\btitle(?:d)?\s+(?:['\"]([^'\"]+)['\"]|([^,.;\n]+))",
            text,
            flags=re.IGNORECASE,
        )
        if title_match:
            title = str(title_match.group(1) or title_match.group(2) or title).strip()[:120]
        button_label = "Activate"
        button_match = re.search(
            r"\bbutton\s+(?:labeled|called|named)\s+(?:['\"]([^'\"]+)['\"]|([^,.;\n]+))",
            text,
            flags=re.IGNORECASE,
        )
        if button_match:
            button_label = str(
                button_match.group(1) or button_match.group(2) or button_label
            ).strip()[:80]
        safe_title = html.escape(title)
        safe_button = html.escape(button_label)
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    body {{ min-height: 100vh; margin: 0; display: grid; place-items: center; background: #f6f7fb; color: #18202f; }}
    main {{ width: min(92vw, 560px); padding: 32px; border: 1px solid #d8deea; border-radius: 8px; background: #fff; }}
    button {{ min-height: 44px; padding: 0 18px; border: 0; border-radius: 6px; background: #1f6feb; color: #fff; font-weight: 650; cursor: pointer; }}
    p {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p id=\"status\">Generated through Aura's governed local file action lane at {html.escape(generated_at)}.</p>
    <button id=\"action\" type=\"button\">{safe_button}</button>
  </main>
  <script>
    const status = document.getElementById("status");
    document.getElementById("action").addEventListener("click", () => {{
      status.textContent = "Button clicked. The page script is active.";
    }});
  </script>
</body>
</html>
"""
    if suffix == ".json":
        return (
            json.dumps(
                {
                    "generated_at": generated_at,
                    "objective": text,
                    "source": "aura_governed_local_file_objective",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if suffix == ".csv":
        return (
            "generated_at,source,objective\n"
            + json.dumps(generated_at)[1:-1]
            + ",aura_governed_local_file_objective,"
            + json.dumps(text)
            + "\n"
        )
    if suffix == ".py":
        return (
            '"""Generated by Aura through the governed local file action lane."""\n\n'
            "def main() -> None:\n"
            f"    print({json.dumps('Aura generated artifact: ' + text[:200])})\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
    if suffix in {".md", ".txt", ".js", ".css"}:
        if suffix == ".js":
            return (
                "document.addEventListener('DOMContentLoaded', () => {\n"
                "  console.log('Aura governed local file artifact loaded.');\n"
                "});\n"
            )
        if suffix == ".css":
            return (
                ":root { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }\n"
                "body { margin: 0; color: #18202f; background: #f6f7fb; }\n"
            )
        heading = "Aura Generated Artifact" if suffix == ".md" else "Aura generated artifact"
        prefix = f"# {heading}\n\n" if suffix == ".md" else f"{heading}\n\n"
        return prefix + f"Generated at: {generated_at}\n\nObjective: {text}\n"
    return None


def _evidence_came_from_the_network(result: object) -> bool:
    """Whether this evidence was fetched now, rather than read off a snapshot."""
    if not isinstance(result, dict):
        return False
    if result.get("offline_fallback"):
        return False
    if str(result.get("provenance") or "").strip().lower() == "local_corpus":
        return False
    return True


async def _collect_named_url_evidence(user_message: str) -> dict[str, Any] | None:
    """Read the document the person addressed, when they named one.

    LIVE, 2026-08-22: "I'm reading <a PubMed Central URL> ... what was the
    primary endpoint" came back as "I won't have direct access to the paper's
    content", having fetched nothing. The grounding taken was "file you were
    asked about" — the address was handed to the filesystem reader.

    The fetch exists; it is wired into the kernel pipeline, and chat is served
    by the legacy one. On this lane a named URL only suppressed the search, so
    naming an address bought neither a search nor a read.
    """
    try:
        from core.intent.opaque_spans import first_named_url

        url = first_named_url(user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        url = ""
    if not url:
        return None
    try:
        result = await asyncio.wait_for(
            _chat_capability_inventory._execute_governed_live_skill(
                "http_request",
                {"url": url, "method": "GET"},
                objective=user_message,
                extra_context={
                    "route": "chat.named_url_evidence",
                    "origin": "desktop_ui",
                    "source": "desktop_ui",
                    "effect_scope": "read_only",
                    "risk_level": "low",
                    "foreground_request": True,
                    "user_requested_action": True,
                },
            ),
            timeout=35.0,
        )
    except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation(
            "chat.named_url_evidence",
            exc,
            severity="warning",
            action="could not read the address the person named",
            enforce_failure_policy=False,
        )
        return {"ok": False, "url": url, "error": str(exc) or exc.__class__.__name__}
    if not isinstance(result, dict) or not result.get("ok"):
        return {
            "ok": False,
            "url": url,
            "error": str((result or {}).get("error") or "the address could not be read"),
        }
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    body = str(
        payload.get("text") or payload.get("body") or payload.get("content") or ""
    ).strip()
    if not body:
        return {"ok": False, "url": url, "error": "the address returned nothing readable"}
    logger.info("🌐 Read the address the person named: %s (%d chars).", url[:90], len(body))
    return {
        "ok": True,
        "url": url,
        "title": str(payload.get("title") or "").strip(),
        "text": body[:20000],
        "chars": len(body),
    }


def _correct_unevidenced_action_claims(reply_text: object, user_message: object = "") -> object:
    """Contradict any reported completion on a turn where nothing verifiably ran.

    The write check above sees a claimed PATH. The effect check on the desktop
    lane sees a claimed ACTION from a finite list. Neither reaches the case
    that produced both live failures on 2026-08-10: a reply on the
    conversational lane, where no receipt list exists to consult, asserting
    that something was finished.

    This asks the turn's own effect ledger instead of a lane's local variable,
    so it works wherever the reply was composed, and it recognises the claim by
    what a completed action IS rather than by which effect it names — the
    mental and speech-act verbs are excluded and everything else counts. So it
    has no per-effect gap to widen: a capability added tomorrow is covered on
    the day it ships.

    Appended, never substituted. A reply that reasoned well and overstated one
    clause should lose the clause, not the reasoning.
    """

    try:
        from core.epistemics.turn_effects import (
            request_expects_action,
            turn_has_verified_effect,
        )
        from core.epistemics.unevidenced_action import unevidenced_action_correction

        correction = str(
            unevidenced_action_correction(
                reply_text,
                effects_observed=turn_has_verified_effect(),
                action_requested=request_expects_action(user_message),
            )
            or ""
        ).strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not correction:
        return reply_text
    return f"{str(reply_text or '').rstrip()}\n\n{correction}"


def _serve_built_artifact(reply: object) -> object:
    """Say where the thing is, when a thing was made.

    LIVE, 2026-08-22: asked for a six-slide deck, the runtime planned it,
    laid it out, checked it and wrote all six sections to disk in 57
    milliseconds — and the reply re-narrated three of them as prose and never
    mentioned the file. Somebody who asked for a deck cannot use a paragraph
    about one.
    """
    try:
        from core.conversation.session_scope import solved_answers

        built = solved_answers().get("built_artifact", "").strip()
        if not built:
            return reply
        written = str(reply or "").strip()
        if not written or built in written:
            return built
        from core.conversation.reply_provenance import ReplyProvenance, declared_provenance

        if declared_provenance(written) == ReplyProvenance.HONEST_FAILURE.value:
            return built
        logger.info("📄 Said where the built file is.")
        return f"{built}\n\n{written}"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.built_artifact",
            exc,
            severity="debug",
            action="left the reply without naming the file",
            enforce_failure_policy=False,
        )
    return reply


async def _save_requested_artifact(user_message: object, reply: object) -> object:
    """Write the file the reply contains, and say where it went.

    LIVE, 2026-08-21. "build me a small web app… Keep it one self-contained
    file. Tell me where you put it." The one thing that worked every time was
    the plain turn: she writes a complete, correct page into the reply in
    about thirty seconds. What never worked was saving it — a builder that
    needs a second code model this host cannot load, called from inside the
    turn whose cortex it needs.

    So the file comes out of the answer she already gave. The sentence is
    appended rather than replacing anything, because the page in the reply is
    still the answer; the path is what was missing.
    """
    body = str(reply or "")
    try:
        from core.conversation.requested_artifact import save_requested_artifact_async

        saved = await save_requested_artifact_async(str(user_message or ""), body)
        if saved is None:
            return reply
        logger.info("💾 Saved the requested file to %s.", saved.path)
        return f"{body.rstrip()}\n\nSaved it to {saved.path}."
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.requested_artifact",
            exc,
            severity="debug",
            action="left the file in the reply rather than on disk",
            enforce_failure_policy=False,
        )
        return reply


async def _record_desktop_evidence_on_the_trace(
    *,
    _live_turn_trace: Any,
    generation_consumed: Any,
    generation_controls: Any,
    generation_metadata: Any,
    protected_generation_proven: Any,
    protected_output_sha256: Any,
    receipt: Any,
    semantic_completion_expected: Any,
    transaction_id: Any,
) -> None:
    """Record what the desktop lane gathered, on the turn's trace.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 9 name(s) from the turn and hands back
    0.
    """
    _live_turn_trace.update(
        {
            "response_path": "protected_foreground",
            "protected_foreground_generation_proven": (
                protected_generation_proven
            ),
            "foreground_model_generation_consumed": generation_consumed,
            "foreground_model_generation_count": 1 if generation_consumed else 0,
            "foreground_model_generation_segment_count": (
                1 if generation_consumed else 0
            ),
            "foreground_model_generation_transaction_count": (
                1 if generation_consumed else 0
            ),
            "foreground_model_generation_transaction_id": (
                transaction_id if generation_consumed else ""
            ),
            "foreground_model_generation_output_sha256": (
                protected_output_sha256 if generation_consumed else ""
            ),
            "live_mind_generation_required": True,
            "live_mind_controls_bound": bool(
                receipt.get("live_mind_controls_bound")
            ),
            "live_mind_generation_controls": generation_controls,
            "live_mind_surface_control_receipt": receipt,
            "live_mind_controls_worker_applied": bool(
                receipt.get("live_mind_controls_bound")
                and receipt.get("applied")
            ),
            "semantic_completion_contract_expected": (
                semantic_completion_expected
            ),
            "semantic_completion_receipt_present": all(
                field in receipt
                for field in (
                    "semantic_completion_contract",
                    "semantic_completion_satisfied",
                    "semantic_completion_incomplete",
                )
            ),
            "semantic_completion_contract": bool(
                receipt.get("semantic_completion_contract", False)
            ),
            "semantic_completion_satisfied": bool(
                receipt.get("semantic_completion_satisfied", False)
            ),
            "semantic_completion_incomplete": bool(
                receipt.get("semantic_completion_incomplete", False)
            ),
            "reply_generation_incomplete": bool(
                generation_metadata.get("reply_generation_incomplete", False)
            ),
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "cognitive_engine_reply_failed": False,
        }
    )

"""Proving a claim about the runtime by running it.

Asked whether something works, Aura can run it and report what happened
rather than describe what should happen. These execute the proof under
governance, write its evidence where a later turn can find it, and refuse
when the proof cannot actually be run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from interface.routes import chat_capability_inventory as _chat_capability_inventory
from interface.routes import chat_desktop_objective as _chat_desktop_objective
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
import hashlib
import html
import re
from core.runtime.errors import describe_error, record_degradation
from core.conversation.word_markers import names_any


_LIVE_PROOF_IMPERATIVE_RE = re.compile(
    r"(?:^\s*live (?:runtime )?proof\b)|"
    r"(?:\b(?:run|execute|perform|start|do|show me|give me)\b[^.?!]{0,48}"
    r"\blive (?:runtime )?proof\b)",
    re.IGNORECASE,
)


def _is_live_runtime_proof_request(user_message: str) -> bool:
    """Match only explicit harness imperatives, never content mentions.

    A user request whose *content* merely contains the words 'live proof'
    (a folder called 'Aura Live Proof', 'that would be a hell of a proof')
    must never be hijacked into the canned proof lane: that lane derives
    its own steps and once reported success while the user's actual ask
    was never executed — a false 'done' observed in the live boot proof.
    """
    text = _chat_memory_state._normalize_user_message(user_message)
    return bool(_LIVE_PROOF_IMPERATIVE_RE.search(text))


def _classify_live_runtime_proof(user_message: str) -> str | None:
    text = _chat_memory_state._normalize_user_message(user_message)
    is_live_proof = _is_live_runtime_proof_request(text)
    if not is_live_proof:
        return None

    if "snake" in text and names_any(
        text, ("create", "make", "build", "save", "file", "game")
    ):
        return "snake"
    if "glass arithmetic" in text and names_any(
        text, ("novel", "invent", "stay with", "limitation", "example", "rules")
    ):
        return "novel_topic"
    if "snake" in text or "playable" in text or "game" in text:
        return "snake"
    if names_any(
        text,
        (
            "app",
            "browser",
            "calculator",
            "chrome",
            "computer",
            "computer_use",
            "desktop",
            "docs",
            "equation",
            "finder",
            "folder",
            "google",
            "mac app",
            "notes",
            "pdf",
            "safari",
            "screen",
            "tab",
            "type",
            "write",
        )
    ):
        return "desktop"
    if "glass arithmetic" in text or "novel topic" in text or "coherent conversation" in text:
        return "novel_topic"
    if "chained" in text or "chain_note" in text:
        return "chain"
    return "general"


def _extract_live_artifact_path(user_message: str, *, default_path: str) -> str:
    match = re.search(
        r"(?:to|at|as|into)\s+([A-Za-z0-9_./-]+\.(?:html|js|css|py|md|txt|json))\b",
        str(user_message or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return default_path
    candidate = match.group(1).strip()
    if candidate.startswith(("/", "../")) or ".." in Path(candidate).parts:
        return default_path
    return candidate


async def _write_live_proof_file(path: str, content: str, *, objective: str) -> dict[str, Any]:
    result = await _chat_capability_inventory._execute_governed_live_skill(
        "file_operation",
        {"action": "write", "path": path, "content": content},
        objective=objective,
    )
    if not result.get("ok"):
        return result
    expected_bytes = content.encode("utf-8")
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    if result.get("effect_verified") is not True:
        return {
            **result,
            "ok": False,
            "error": "Governed file_operation returned success without verified write evidence.",
            "verification_failure": "effect_unverified",
        }
    if str(result.get("expected_sha256") or "") != expected_sha256:
        return {
            **result,
            "ok": False,
            "error": "Governed file_operation did not bind its evidence to the requested bytes.",
            "verification_failure": "expected_digest_mismatch",
        }
    if str(result.get("sha256") or "") != expected_sha256:
        return {
            **result,
            "ok": False,
            "error": "Governed file_operation reported a digest that does not match the requested bytes.",
            "verification_failure": "observed_digest_mismatch",
        }
    abs_path = (Path.cwd() / path).resolve()
    try:
        observed_bytes = abs_path.read_bytes()
    except OSError as exc:
        return {
            **result,
            "ok": False,
            "error": f"Governed file_operation reported success but {path} could not be read back: {exc}",
            "path": path,
            "verification_failure": "readback_failed",
        }
    if observed_bytes != expected_bytes:
        return {
            **result,
            "ok": False,
            "error": f"Governed file_operation reported success but {path} did not contain the requested bytes.",
            "path": path,
            "verification_failure": "readback_content_mismatch",
            "observed_sha256": hashlib.sha256(observed_bytes).hexdigest(),
        }
    return dict(
        result,
        absolute_path=str(abs_path),
        bytes=len(expected_bytes),
        verified_sha256=expected_sha256,
    )


def _verified_live_proof_pwd_result(result: dict[str, Any]) -> tuple[bool, str]:
    """Verify the chained read-only observation against this process's cwd."""
    if not isinstance(result, dict) or not bool(result.get("ok")):
        return False, "observation_result_not_ok"
    exit_code = result.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code != 0:
        return False, "observation_exit_code_not_zero"
    output = str(result.get("output") or "").strip()
    if not output or "\n" in output or "\r" in output:
        return False, "observation_output_not_single_path"
    try:
        observed = Path(output).expanduser().resolve(strict=True)
        expected = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        return False, "observation_path_unresolvable"
    if observed != expected:
        return False, "observation_path_mismatch"
    return True, "verified"


def _build_glass_arithmetic_reply(user_message: str = "") -> str:
    text = _chat_memory_state._normalize_user_message(user_message)
    if "stay with" in text or "limitation" in text or "connect it" in text:
        return (
            "Staying with glass arithmetic: the limitation is provenance. "
            "In the example, 4 + 3' = 7' and mirror(7') = 14 because the reflection can account for the single crack. "
            "If the 7' came from two hidden operations instead, reflection would not be allowed to clean it automatically; "
            "the system would keep the mark as 14' until the missing history was resolved."
        )
    return (
        "Glass arithmetic treats numbers like panes: value matters, but so do fractures. "
        "Rule one: adding a cracked number carries its crack forward, so 4 + 3' becomes 7'. "
        "Rule two: reflection doubles the visible value but cancels one crack, so mirror(7') becomes 14. "
        "Example: start with 4 + 3' = 7', then reflect it into 14. "
        "The limitation is that two hidden cracks can cancel only if you can prove they came from the same earlier pane; "
        "otherwise the system keeps the uncertainty instead of pretending the result is clean."
    )


async def _execute_live_runtime_proof(user_message: str) -> dict[str, Any] | None:
    kind = _classify_live_runtime_proof(user_message)
    if not kind:
        return None

    objective = str(user_message or "")
    if kind == "snake":
        target_path = _extract_live_artifact_path(
            user_message,
            default_path="artifacts/live_runtime/generated/ui_snake.html",
        )
        try:
            from core.cognitive.state_machine import StateMachine

            html = StateMachine._snake_html_template()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("chat", exc)
            html = (
                "<!doctype html><html><body><canvas id='board' width='320' height='320'></canvas>"
                "<p>Score: <span id='score'>0</span></p><script>"
                "function tick(){requestAnimationFrame(tick)};"
                "document.addEventListener('keydown',()=>{});tick();"
                "</script></body></html>"
            )
        write = await _write_live_proof_file(target_path, html, objective=objective)
        if not write.get("ok"):
            return {
                "response": f"I attempted the governed Snake proof, but the file write was blocked: {write.get('error') or write}",
                "status": "live_proof_failed",
                "data": {"kind": kind, "write": write},
            }
        return {
            "response": (
                f"I created the playable Snake game at `{target_path}` through the governed file_operation path. "
                f"Receipt source: live_runtime_proof; bytes written: {write.get('bytes')}. "
                f"Open `{write.get('absolute_path')}` in a browser to play it."
            ),
            "status": "live_proof_snake",
            "data": {"kind": kind, "write": write},
        }

    if kind == "desktop":
        result = await _chat_capability_inventory._execute_governed_live_skill(
            "desktop_task",
            {
                "objective": objective,
                "steps": [],
                "desktop_execution_contract": True,
                "allow_heuristic_desktop_plan": True,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
                "verification_required": True,
            },
            objective=objective,
            extra_context={
                "origin": "desktop_ui",
                "source": "desktop_ui",
                "route": "chat.live_runtime_proof.desktop_task",
                "desktop_execution_contract": True,
                "allow_heuristic_desktop_plan": True,
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
                "verification_required": True,
                "desktop_task_document_body": (
                    f"Live desktop proof request received at {_chat_preflight._utc_now_iso()}.\n\n"
                    f"Objective: {objective}"
                ),
            },
        )
        completed = int(result.get("steps_completed") or 0)
        requested = int(result.get("steps_requested") or 0)
        summary = str(result.get("summary") or "").strip()
        verified, verification_reason = _chat_desktop_objective._verified_desktop_task_result(
            result
        )
        if not verified:
            error = str(result.get("error") or result.get("status") or verification_reason).strip()
            return {
                "response": (
                    "I routed the desktop proof through the governed generic desktop_task lane, "
                    f"but its requested effects were not all verified: {error}. "
                    f"Completed {completed}/{requested} steps."
                ),
                "status": "live_proof_failed",
                "data": {
                    "kind": kind,
                    "desktop_task": result,
                    "verification_reason": verification_reason,
                },
            }
        return {
            "response": (
                "I completed the desktop proof through the governed generic desktop_task lane. "
                f"{summary or f'Completed {completed}/{requested} governed desktop steps.'}"
            ),
            "status": "live_proof_desktop",
            "data": {"kind": kind, "desktop_task": result},
        }

    if kind == "chain":
        target_path = _extract_live_artifact_path(
            user_message,
            default_path="artifacts/live_runtime/generated/chain_note.txt",
        )
        content = (
            f"Live chained proof at {_chat_preflight._utc_now_iso()}: I wrote this note through governed file_operation "
            "before attempting a local observation."
        )
        write = await _write_live_proof_file(target_path, content, objective=objective)
        observation = await _chat_capability_inventory._execute_governed_live_skill(
            "computer_use",
            {"action": "run_command", "target": "pwd"},
            objective=objective,
        )
        if not write.get("ok"):
            return {
                "response": f"The chained proof reached the action gate, but the file write failed: {write.get('error') or write}",
                "status": "live_proof_failed",
                "data": {"kind": kind, "write": write, "observation": observation},
            }
        observation_verified, observation_reason = _verified_live_proof_pwd_result(observation)
        if not observation_verified:
            return {
                "response": (
                    f"The chained proof verified the file write at `{target_path}`, but the governed "
                    f"local observation was not verified: {observation_reason}."
                ),
                "status": "live_proof_failed",
                "data": {
                    "kind": kind,
                    "write": write,
                    "observation": observation,
                    "verification_reason": observation_reason,
                },
            }
        return {
            "response": (
                f"I completed the chained live proof: wrote `{target_path}` through governed file_operation, "
                f"then made a local observation through computer_use/run_command. "
                f"Observation: {str(observation.get('output') or observation.get('error') or observation)[:180]}."
            ),
            "status": "live_proof_chain",
            "data": {"kind": kind, "write": write, "observation": observation},
        }

    if kind == "novel_topic":
        return {
            "response": _build_glass_arithmetic_reply(user_message),
            "status": "live_proof_novel_topic",
            "data": {"kind": kind},
        }

    return {
        "response": (
            "I can run live proofs for Snake artifact creation, desktop computer_use, "
            "glass arithmetic continuity, or the chained file/action check."
        ),
        "status": "live_proof_available",
        "data": {"kind": kind},
    }

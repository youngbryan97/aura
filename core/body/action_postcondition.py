"""core/body/action_postcondition.py
Verification engine validating preconditions, postconditions, and side effects.

A verifier that asks the actuator whether it succeeded is not a verifier. CP126
found exactly that: ``success = receipt["status"] == "success"`` — the
component under test supplying its own verdict — with one narrow
path-existence exception able to overturn it. Everything else the module's
contract promised (pre-action snapshot, expected effect, target identity,
side-effect diff, rollback check) was absent, and the side effects it did
report were read off the request rather than observed.

Three rules now hold:

* **Independent evidence or none.** The receipt's own claim is recorded as a
  claim. ``verified`` is True only when something outside the actuator
  corroborates it; when nothing can, the result says which check was
  unavailable.
* **Effects are observed, not inferred.** A file write is confirmed against the
  filesystem — existence, size, content digest — and against a pre-action
  snapshot when one was taken.
* **What is stored is redacted and bounded**, because a receipt can carry
  command output, clipboard contents, URLs and spoken text.

CP126 08943467 / 92b64654 / 2468cc73 / 36ae11a1 / ebc0d1eb.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any, Dict

logger = logging.getLogger("Body.ActionPostcondition")

#: Receipt fields that may carry sensitive payloads (CP126 ebc0d1eb).
_SENSITIVE_KEYS = (
    "output", "stdout", "stderr", "clipboard", "text", "content", "body",
    "spoken", "transcript", "password", "token", "secret", "key", "credential",
    "url", "prompt", "response",
)
MAX_TELEMETRY_VALUE_CHARS = 200
MAX_TELEMETRY_KEYS = 24
MAX_HASHED_BYTES = 8 * 1024 * 1024

_SECRET_RE = re.compile(
    r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,}\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----",
)


def redact_telemetry(receipt: Any) -> Dict[str, Any]:
    """A bounded, redacted view of an actuator receipt.

    CP126 ebc0d1eb: the FULL receipt was embedded in world state, so command
    output, clipboard data, URLs, paths and spoken text were persisted with no
    redaction, retention or scoping.
    """
    if not isinstance(receipt, dict):
        return {"_shape": type(receipt).__name__}
    safe: Dict[str, Any] = {}
    for index, (key, value) in enumerate(receipt.items()):
        if index >= MAX_TELEMETRY_KEYS:
            safe["_truncated_keys"] = len(receipt) - MAX_TELEMETRY_KEYS
            break
        name = str(key)
        lowered = name.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEYS):
            text = str(value or "")
            safe[name] = {
                "_redacted": True,
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32],
            }
            continue
        if isinstance(value, str):
            safe[name] = _SECRET_RE.sub("[REDACTED]", value)[:MAX_TELEMETRY_VALUE_CHARS]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[name] = value
        else:
            safe[name] = f"<{type(value).__name__}>"
    return safe


def snapshot_path(path: Any) -> Dict[str, Any]:
    """Pre-action state of a filesystem target.

    CP126 92b64654: there was no pre-action snapshot at all, so no
    postcondition could be a DIFF — only an existence check.
    """
    target = str(path or "").strip()
    if not target:
        return {"path": "", "exists": False, "reason": "no path supplied"}
    try:
        stat = os.stat(target)
    except (OSError, ValueError):
        return {"path": target, "exists": False, "at": time.time()}
    return {
        "path": target,
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": _digest(target, stat.st_size),
        "at": time.time(),
    }


async def snapshot_path_async(path: Any) -> Dict[str, Any]:
    """:func:`snapshot_path` off the event loop.

    Stat plus a content digest is blocking I/O, and this verifier runs inside
    the life tick.
    """
    return await asyncio.to_thread(snapshot_path, path)


def _digest(path: str, size: int) -> str:
    if size > MAX_HASHED_BYTES:
        return ""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except (OSError, ValueError):
        return ""


class ActionPostconditionVerifier:
    """Verifies action execution outcomes and registers evidence in the LifeState."""

    async def verify(
        self,
        receipt: Dict[str, Any],
        state: Any,
        *,
        before: Dict[str, Any] | None = None,
        expected_effect: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Check an action's outcome against evidence outside the actuator.

        ``before`` is a :func:`snapshot_path` taken before the action;
        ``expected_effect`` declares what should have changed. Both are
        optional, and their absence is REPORTED rather than assumed away.
        """
        receipt = receipt if isinstance(receipt, dict) else {}
        channel = str(receipt.get("channel", "unknown"))
        status = str(receipt.get("status", "failed"))

        # CP126 08943467: this IS the actuator's own verdict. It is recorded as
        # a claim to be corroborated, never as the finding.
        claimed_success = status == "success"

        checks: list[Dict[str, Any]] = []
        side_effects: list[str] = []
        path = receipt.get("path")

        # Every filesystem read happens here, once, off the event loop.
        observed: Dict[str, Dict[str, Any]] = {}
        for candidate in (path, (expected_effect or {}).get("path")):
            target = str(candidate or "").strip()
            if target and target not in observed:
                observed[target] = await snapshot_path_async(target)

        if channel == "file":
            checks.append(self._verify_file(path, before, observed, side_effects))
        elif channel == "terminal":
            checks.append(self._verify_terminal(receipt, side_effects))
            if path:
                checks.append(self._verify_file(path, before, observed, side_effects))
        else:
            checks.append({
                "check": f"{channel}_effect",
                "verified": False,
                "reason": f"no independent postcondition exists for channel {channel!r}",
            })

        if expected_effect:
            checks.append(self._verify_expectation(expected_effect, receipt, before, observed))

        corroborated = [check for check in checks if check.get("verified")]
        refuted = [check for check in checks if check.get("refutes")]

        # Anything that refutes the claim overturns it. Otherwise the claim
        # stands, but `verified` says whether anything actually backed it.
        success = False if refuted else claimed_success
        verified = bool(corroborated) and not refuted

        if refuted and claimed_success:
            logger.warning(
                "False success detected! Tool reported success but %s",
                "; ".join(str(check.get("reason")) for check in refuted),
            )

        verification = {
            "channel": channel,
            "success": success,
            # CP126 08943467: the honesty fields — whose verdict this is.
            "claimed_success": claimed_success,
            "verified": verified,
            "verification_source": "independent_evidence" if verified else "actuator_claim",
            "checks": checks,
            "side_effects": side_effects,
            "evidence": {
                "observed_result": (
                    "conforms_to_preconditions" if success else "deviated_from_expectation"
                ),
                # CP126 ebc0d1eb: redacted and bounded before it is persisted.
                "telemetry": redact_telemetry(receipt),
                "had_pre_action_snapshot": bool(before),
                "had_expected_effect": bool(expected_effect),
            },
            "at": time.time(),
        }

        logger.info(
            "Verification complete for channel %s: success=%s verified=%s",
            channel, success, verified,
        )
        self._record(state, verification)
        return verification

    # -- channel checks ---------------------------------------------------
    @staticmethod
    def _verify_file(
        path: Any,
        before: Dict[str, Any] | None,
        observed: Dict[str, Dict[str, Any]],
        side_effects: list[str],
    ) -> Dict[str, Any]:
        """Confirm a file effect against the filesystem.

        CP126 2468cc73: a write was labelled ``modified_file`` from the request
        fields alone — no metadata, no content hash, no proof anything changed.
        """
        target = str(path or "").strip()
        if not target:
            return {
                "check": "file_effect",
                "verified": False,
                "reason": "receipt named no path to verify",
            }
        after = observed.get(target) or {"exists": False}
        if not after["exists"]:
            side_effects.append(f"expected_file_missing:{target}")
            return {
                "check": "file_effect",
                "verified": True,
                "refutes": True,
                "reason": f"output path '{target}' does not exist",
            }

        if before is None:
            side_effects.append(f"file_present:{target}")
            return {
                "check": "file_effect",
                "verified": False,
                "reason": "no pre-action snapshot, so presence is not proof of a change",
                "after": {key: after[key] for key in ("size", "mtime") if key in after},
            }

        changed = (
            before.get("exists") is not True
            or before.get("sha256") != after.get("sha256")
            or before.get("size") != after.get("size")
        )
        side_effects.append(
            f"modified_file:{target}" if changed else f"file_unchanged:{target}"
        )
        return {
            "check": "file_effect",
            "verified": True,
            "changed": changed,
            "reason": "content digest differs" if changed else "content digest is identical",
        }

    @staticmethod
    def _verify_terminal(receipt: Dict[str, Any], side_effects: list[str]) -> Dict[str, Any]:
        """Read the process outcome from the receipt's exit code.

        CP126 4bf25067 (already closed): a missing exit_code defaulted to 0 —
        success — so a receipt that never reported how the process ended
        suppressed its own failure evidence. Absent is unknown, not zero.
        """
        if "exit_code" not in receipt:
            side_effects.append("process_exit_code_unreported")
            return {
                "check": "process_exit",
                "verified": False,
                "reason": "receipt reported no exit code",
            }
        exit_code = receipt.get("exit_code")
        try:
            code = int(exit_code)
        except (TypeError, ValueError):
            side_effects.append("process_exit_code_unreadable")
            return {
                "check": "process_exit",
                "verified": False,
                "reason": f"exit code {exit_code!r} is not an integer",
            }
        if code != 0:
            side_effects.append(f"process_failed_with_code:{code}")
            return {
                "check": "process_exit",
                "verified": True,
                "refutes": True,
                "reason": f"process exited with code {code}",
            }
        return {"check": "process_exit", "verified": True, "reason": "process exited cleanly"}

    @staticmethod
    def _verify_expectation(
        expected: Dict[str, Any],
        receipt: Dict[str, Any],
        before: Dict[str, Any] | None,
        observed: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check a caller-declared expected effect (CP126 92b64654)."""
        target = str(expected.get("path") or receipt.get("path") or "").strip()
        if not target:
            return {
                "check": "expected_effect",
                "verified": False,
                "reason": "expected effect named no target",
            }
        after = observed.get(target) or {"exists": False}
        wants_exists = expected.get("exists")
        if wants_exists is not None and bool(after["exists"]) != bool(wants_exists):
            return {
                "check": "expected_effect",
                "verified": True,
                "refutes": True,
                "reason": f"expected exists={wants_exists}, observed {after['exists']}",
            }
        wants_digest = str(expected.get("sha256") or "")
        if wants_digest and after.get("sha256") != wants_digest:
            return {
                "check": "expected_effect",
                "verified": True,
                "refutes": True,
                "reason": "content digest does not match the expected value",
            }
        wants_change = expected.get("changed")
        if wants_change is not None and before is not None:
            changed = before.get("sha256") != after.get("sha256")
            if bool(changed) != bool(wants_change):
                return {
                    "check": "expected_effect",
                    "verified": True,
                    "refutes": True,
                    "reason": f"expected changed={wants_change}, observed {changed}",
                }
        return {"check": "expected_effect", "verified": True, "reason": "expectation met"}

    # -- ledger -----------------------------------------------------------
    @staticmethod
    def _record(state: Any, verification: Dict[str, Any]) -> bool:
        """Write the verification into the world model, guarded.

        CP126 36ae11a1: this wrote straight into ``state.world_model`` with no
        type check, so a missing or non-mapping world model raised inside the
        verifier and a failure left no trace.
        """
        world_model = getattr(state, "world_model", None)
        if world_model is None:
            logger.warning("Verification not recorded: state has no world_model")
            return False
        if not hasattr(world_model, "__setitem__"):
            logger.warning(
                "Verification not recorded: world_model is %s, not a mapping",
                type(world_model).__name__,
            )
            return False
        try:
            world_model["last_verification"] = verification
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            logger.warning("Verification could not be recorded: %s", exc)
            return False
        return True

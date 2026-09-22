"""Bidirectional Global Workspace ↔ Recursive Latent Cortex coupling.

The loop the architecture demands:

    GWT coalitions → RLC deliberation → revised broadcast → Will/action

Direction 1 — coalitions become thoughts. When a latent episode launches,
what the mind is actually broadcasting (the last workspace winner) and what
is currently competing for broadcast become typed cognitive-context items,
seeding identifiable, individually ablatable thought slots. Deep reasoning
deliberates over the mind's live contents, not just the user's prompt.

Direction 2 — conclusions return to the mind. A successful episode's answer
is submitted back to the canonical Global Workspace as a CognitiveCandidate
from source ``latent_cortex``, priced by how the conclusion was earned
(verifier-checked beats convergence-only) and what was at stake. It then
COMPETES like every other coalition — broadcast is won, never granted — and
whatever wins reaches the Will through the normal path. This makes the RLC
the deliberative mode of the existing mind rather than a bolt-on
response-generation route.

Both directions are defensive (no workspace ⇒ no-op with a receipted
reason) and bounded.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

logger = logging.getLogger("Aura.GwtRlcCoupling")

GWT_RLC_SCHEMA = "aura.gwt_rlc_coupling.v1"

_MAX_TEXT_CHARS = 400
_MAX_COALITION_ITEMS = 2
_MAX_CONTEXT_ITEMS = 6
_BASE_PRIORITY = 0.5
_VERIFIED_BONUS = 0.2
_STAKES_WEIGHT = 0.15
_PRIORITY_CAP = 0.9


def _workspace() -> Any:
    try:
        from core.runtime.service_access import resolve_global_workspace

        return resolve_global_workspace(default=None)
    # not a failure: no service here means the caller falls back to its own default.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


# ── Direction 1: coalitions → slot candidates ───────────────────────────
def workspace_coalition_context(
    max_items: int = _MAX_COALITION_ITEMS,
) -> list[dict[str, str]]:
    """Render the live workspace state as cognitive-context items.

    Item 1 is the current broadcast (last competition winner); further items
    are the strongest still-competing coalitions. Sources are
    ``workspace_broadcast`` / ``workspace_coalition`` so each seeded slot is
    identifiable in the episode receipt and ablation-testable.
    """
    workspace = _workspace()
    if workspace is None or max_items <= 0:
        return []
    items: list[dict[str, str]] = []
    try:
        winner = getattr(workspace, "last_winner", None)
        if winner is not None:
            content = str(getattr(winner, "content", "") or "").strip()
            source = str(getattr(winner, "source", "") or "").strip()
            if content:
                items.append(
                    {
                        "source": "workspace_broadcast",
                        "text": f"Current broadcast [{source}]: {content}"[
                            :_MAX_TEXT_CHARS
                        ],
                    }
                )
        coalition_reader = getattr(workspace, "get_competing_coalitions", None)
        if callable(coalition_reader):
            for row in coalition_reader(max_items):
                if len(items) >= max_items:
                    break
                content = str(row.get("content") or "").strip()
                source = str(row.get("source") or "").strip()
                if not content or source == "latent_cortex":
                    continue
                items.append(
                    {
                        "source": "workspace_coalition",
                        "text": f"Competing coalition [{source}]: {content}"[
                            :_MAX_TEXT_CHARS
                        ],
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Workspace coalition read failed: %s", exc)
    return items[:max_items]


def merge_cognitive_context(
    organ_items: list[dict[str, Any]] | None,
    *,
    max_items: int = _MAX_CONTEXT_ITEMS,
) -> list[dict[str, Any]] | None:
    """Organ items first, live coalitions filling the remaining slots.

    Organ content (memory, goals, world model, interoception, epistemic
    caution) outranks workspace chatter; the coalition items only occupy
    slots the organs left free. Returns None when nothing is available so
    the wire contract stays unchanged for empty cases.
    """
    merged = [dict(item) for item in (organ_items or [])][:max_items]
    room = max_items - len(merged)
    if room > 0:
        merged.extend(workspace_coalition_context(min(room, _MAX_COALITION_ITEMS)))
    return merged or None


# ── Direction 2: conclusions → broadcast ────────────────────────────────
def _conclusion_priority(receipt: dict[str, Any], stakes: float) -> tuple[float, dict]:
    """Price the conclusion by how it was earned, with the math receipted."""
    verifier = receipt.get("verifier_guidance")
    best_score = 0.0
    verified = False
    if isinstance(verifier, dict) and verifier.get("evaluations"):
        raw_best = verifier.get("best_score")
        if (
            isinstance(raw_best, (int, float))
            and not isinstance(raw_best, bool)
            and math.isfinite(float(raw_best))
        ):
            best_score = max(0.0, min(1.0, float(raw_best)))
            verified = best_score > 0.5 and not verifier.get("best_failures")
    bounded_stakes = max(0.0, min(1.0, float(stakes)))
    # Identity consistency prices the bid: a conclusion the canonical self
    # flagged must outcompete honestly from a lower rung — bounded penalty,
    # never a veto.
    identity = receipt.get("identity_consistency")
    identity_penalty = 0.0
    if isinstance(identity, dict):
        raw_penalty = identity.get("priced_penalty")
        if (
            isinstance(raw_penalty, (int, float))
            and not isinstance(raw_penalty, bool)
            and math.isfinite(float(raw_penalty))
        ):
            identity_penalty = max(0.0, min(0.3, float(raw_penalty)))
    priority = max(
        0.05,
        min(
            _PRIORITY_CAP,
            _BASE_PRIORITY
            + (_VERIFIED_BONUS * best_score if verified else 0.0)
            + _STAKES_WEIGHT * bounded_stakes,
        )
        - identity_penalty,
    )
    return priority, {
        "base": _BASE_PRIORITY,
        "verified": verified,
        "verifier_best_score": round(best_score, 4),
        "stakes": round(bounded_stakes, 4),
        "identity_penalty": round(identity_penalty, 4),
        "priority": round(priority, 4),
    }


async def broadcast_episode_conclusion(
    objective: str,
    conclusion_text: str,
    receipt: dict[str, Any],
    *,
    stakes: float = 0.5,
) -> dict[str, Any]:
    """Submit the episode's conclusion to the workspace competition.

    The candidate carries provenance (episode id, objective digest, how the
    priority was priced) and competes under the same gates as every other
    subsystem — no bypass, no guaranteed broadcast.
    """
    workspace = _workspace()
    if workspace is None:
        return {
            "schema": GWT_RLC_SCHEMA,
            "submitted": False,
            "reason": "workspace_absent",
        }
    text = str(conclusion_text or "").strip()
    if not text:
        return {
            "schema": GWT_RLC_SCHEMA,
            "submitted": False,
            "reason": "empty_conclusion",
        }
    try:
        from core.consciousness.global_workspace import (
            CognitiveCandidate,
            ContentType,
        )

        priority, pricing = _conclusion_priority(dict(receipt or {}), stakes)
        candidate = CognitiveCandidate(
            content=text[:_MAX_TEXT_CHARS],
            source="latent_cortex",
            priority=priority,
            content_type=ContentType.META,
            metadata={
                "schema": GWT_RLC_SCHEMA,
                "episode_id": str((receipt or {}).get("episode_id") or ""),
                "objective_sha256": hashlib.sha256(
                    str(objective or "").encode("utf-8")
                ).hexdigest(),
                "pricing": pricing,
            },
        )
        accepted = await workspace.submit(candidate)
        return {
            "schema": GWT_RLC_SCHEMA,
            "submitted": True,
            "accepted": bool(accepted),
            "priority": pricing["priority"],
            "pricing": pricing,
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "latent_cortex",
            exc,
            action="dropped workspace broadcast of a latent conclusion",
            severity="warning",
        )
        return {
            "schema": GWT_RLC_SCHEMA,
            "submitted": False,
            "reason": f"submit_failed:{type(exc).__name__}",
        }


__all__ = [
    "GWT_RLC_SCHEMA",
    "broadcast_episode_conclusion",
    "merge_cognitive_context",
    "workspace_coalition_context",
]

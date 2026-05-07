from __future__ import annotations

from typing import Any, Iterable

from .unity_state import BoundContent, SelfWorldBinding


class SelfWorldBindingModel:
    """Track what belongs to self, world, authored action, and outside events."""

    _CLAIMED_AUTHORSHIP_MARKERS = (
        "you chose this",
        "you decided this",
        "you wanted this",
        "you made me",
    )

    def bind(
        self,
        state: Any,
        contents: Iterable[BoundContent],
        *,
        will_receipt_id: str | None = None,
        workspace_frame: Any | None = None,
    ) -> SelfWorldBinding:
        self_refs: list[str] = []
        world_refs: list[str] = []
        authored_refs: list[str] = []
        external_refs: list[str] = []
        contamination_flags: list[str] = []

        for item in contents:
            if item.ownership == "self":
                self_refs.append(item.content_id)
            elif item.ownership == "world":
                world_refs.append(item.content_id)
                external_refs.append(item.content_id)
            elif item.ownership == "other":
                world_refs.append(item.content_id)
            else:
                contamination_flags.append(f"ambiguous:{item.content_id}")

        working_memory = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
        recent_user_text = ""
        for idx, message in enumerate(working_memory[-8:]):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").lower()
            content = " ".join(str(message.get("content", "") or "").split())
            metadata = dict(message.get("metadata") or {})
            receipt = (
                metadata.get("will_receipt_id")
                or metadata.get("receipt_id")
                or message.get("will_receipt_id")
            )

            if role == "assistant":
                if receipt or metadata.get("action") or message.get("action"):
                    authored_refs.append(str(receipt or f"assistant_action_{idx}"))
                elif content:
                    self_refs.append(f"assistant_trace_{idx}")
            elif role == "user":
                if content:
                    recent_user_text = content.lower()
                    external_refs.append(f"user_turn_{idx}")
            elif role in {"system", "tool"} and content:
                world_refs.append(f"system_event_{idx}")

        if will_receipt_id:
            authored_refs.append(str(will_receipt_id))

        if any(marker in recent_user_text for marker in self._CLAIMED_AUTHORSHIP_MARKERS) and not authored_refs:
            contamination_flags.append("claimed_authorship_without_receipt")

        ambiguous_count = len([flag for flag in contamination_flags if flag.startswith("ambiguous:")])
        authored_count = len(set(authored_refs))
        world_count = len(set(world_refs + external_refs))

        ownership_confidence = 1.0
        if ambiguous_count:
            ownership_confidence -= min(0.45, ambiguous_count * 0.1)
        if "claimed_authorship_without_receipt" in contamination_flags:
            ownership_confidence -= 0.35
        ownership_confidence = max(0.0, min(1.0, ownership_confidence))

        agency_score = max(0.0, min(1.0, (0.35 if authored_count else 0.0) + (0.65 * ownership_confidence)))
        responsibility_score = max(0.0, min(1.0, agency_score * (1.0 if authored_count else 0.75)))
        boundary_integrity = max(
            0.0,
            min(
                1.0,
                1.0
                - (ambiguous_count * 0.08)
                - (0.25 if "claimed_authorship_without_receipt" in contamination_flags else 0.0)
                - (0.08 if workspace_frame and not getattr(workspace_frame, "focus", None) else 0.0),
            ),
        )

        return SelfWorldBinding(
            self_state_refs=list(dict.fromkeys(self_refs)),
            world_state_refs=list(dict.fromkeys(world_refs)),
            authored_action_refs=list(dict.fromkeys(authored_refs)),
            external_event_refs=list(dict.fromkeys(external_refs)),
            ownership_confidence=round(ownership_confidence, 4),
            agency_score=round(agency_score, 4),
            responsibility_score=round(responsibility_score, 4),
            boundary_integrity=round(boundary_integrity, 4),
            contamination_flags=list(dict.fromkeys(contamination_flags)),
        )

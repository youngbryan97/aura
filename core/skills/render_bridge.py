"""core/skills/render_bridge.py — Render Instruction Bridge
=============================================================
First-class BaseSkill that processes render instructions for the UI.
This is the bridge between Aura's cognitive output and visual presentation:
  - Inline citation rendering (sources, footnotes)
  - Image/file embedding in chat responses
  - Code block syntax highlighting directives
  - Interactive chart/graph rendering instructions
  - Progress indicator streaming
  - Substrate visualization data packaging

The render bridge converts structured render instructions into JSON
payloads that the frontend chat UI can interpret and display.

This closes the "render ecosystem" gap in tool parity.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.RenderBridge")

_RENDER_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


def _record_render_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "render_bridge",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class RenderInstruction(BaseModel):
    """A single render instruction for the frontend."""
    type: str = Field(
        ...,
        description=(
            "Render type: 'citation', 'image', 'file', 'code', 'chart', "
            "'progress', 'substrate', 'table', 'card', 'alert'."
        ),
    )
    content: Any = Field(None, description="Primary content payload.")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Additional metadata for the render instruction.",
    )


class RenderBridgeInput(BaseModel):
    instructions: list[dict[str, Any]] = Field(
        ...,
        description="List of render instructions to process.",
    )
    target: str = Field(
        "chat",
        description="Target UI surface: 'chat', 'dashboard', 'overlay'.",
    )
    stream_id: str | None = Field(
        None,
        description="Stream ID for correlating with a streaming response.",
    )


class RenderBridgeSkill(BaseSkill):
    name = "render_bridge"
    description = (
        "Process and emit render instructions for the UI. "
        "Handles inline citations, image embedding, code highlighting, "
        "chart rendering, progress indicators, and substrate visualization. "
        "Converts structured data into frontend-ready JSON payloads."
    )
    input_model = RenderBridgeInput
    timeout_seconds = 10.0
    metabolic_cost = 0  # Core infrastructure, negligible cost
    effect_scope = "pure_compute"

    # Render type handlers
    _RENDER_TYPES = {
        "citation",
        "image",
        "file",
        "code",
        "chart",
        "progress",
        "substrate",
        "table",
        "card",
        "alert",
        "divider",
        "embed",
    }

    async def execute(
        self, params: RenderBridgeInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Process render instructions and emit UI payloads."""
        if isinstance(params, dict):
            try:
                params = RenderBridgeInput(**params)
            except _RENDER_RECOVERABLE_ERRORS as exc:
                _record_render_degradation(
                    exc,
                    action="rejected invalid render bridge input",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        if not params.instructions:
            return {"ok": False, "error": "No render instructions provided."}

        processed: list[dict[str, Any]] = []
        errors: list[str] = []

        for i, instruction in enumerate(params.instructions):
            try:
                rendered = self._process_instruction(instruction, i)
                if rendered:
                    processed.append(rendered)
            except _RENDER_RECOVERABLE_ERRORS as exc:
                errors.append(f"Instruction {i}: {exc}")
                logger.debug("Render instruction %d failed: %s", i, exc)

        # Emit via EventBus for real-time streaming
        if processed:
            self._emit_render_payload(processed, params.target, params.stream_id)

        return {
            "ok": len(processed) > 0,
            "rendered": processed,
            "count": len(processed),
            "errors": errors if errors else None,
            "target": params.target,
            "stream_id": params.stream_id,
            "summary": f"Processed {len(processed)}/{len(params.instructions)} render instructions.",
        }

    def _process_instruction(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any] | None:
        """Process a single render instruction into a UI payload."""
        render_type = str(instruction.get("type", "")).lower().strip()
        if not render_type:
            return None

        if render_type not in self._RENDER_TYPES:
            logger.warning("Unknown render type: %s", render_type)
            # Pass through unknown types for extensibility
            return {
                "type": render_type,
                "content": instruction.get("content"),
                "metadata": instruction.get("metadata", {}),
                "index": index,
                "timestamp": time.time(),
            }

        handler = getattr(self, f"_render_{render_type}", None)
        if handler:
            return handler(instruction, index)

        # Default: pass through with standard envelope
        return {
            "type": render_type,
            "content": instruction.get("content"),
            "metadata": instruction.get("metadata", {}),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_citation(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render an inline citation / source reference."""
        content = instruction.get("content", {})
        if isinstance(content, str):
            content = {"text": content}
        return {
            "type": "citation",
            "source_url": content.get("url", ""),
            "source_title": content.get("title", ""),
            "source_text": content.get("text", ""),
            "footnote_index": content.get("index", index + 1),
            "confidence": content.get("confidence", 1.0),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_image(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render an inline image."""
        content = instruction.get("content", {})
        if isinstance(content, str):
            content = {"url": content}
        return {
            "type": "image",
            "url": content.get("url", ""),
            "alt_text": content.get("alt", "Generated image"),
            "caption": content.get("caption", ""),
            "width": content.get("width"),
            "height": content.get("height"),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_code(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render a code block with syntax highlighting."""
        content = instruction.get("content", {})
        if isinstance(content, str):
            content = {"code": content}
        return {
            "type": "code",
            "code": content.get("code", ""),
            "language": content.get("language", "python"),
            "filename": content.get("filename"),
            "line_numbers": content.get("line_numbers", True),
            "highlight_lines": content.get("highlight_lines", []),
            "executable": content.get("executable", False),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_chart(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render a chart/graph visualization."""
        content = instruction.get("content", {})
        return {
            "type": "chart",
            "chart_type": content.get("chart_type", "bar"),
            "data": content.get("data", {}),
            "title": content.get("title", ""),
            "x_label": content.get("x_label", ""),
            "y_label": content.get("y_label", ""),
            "options": content.get("options", {}),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_progress(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render a progress indicator."""
        content = instruction.get("content", {})
        if isinstance(content, (int, float)):
            content = {"percent": content}
        return {
            "type": "progress",
            "percent": content.get("percent", 0),
            "label": content.get("label", "Processing..."),
            "status": content.get("status", "active"),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_substrate(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render substrate/neural state visualization data."""
        content = instruction.get("content", {})
        return {
            "type": "substrate",
            "dimensions": {
                "curiosity": content.get("curiosity", 0.5),
                "frustration": content.get("frustration", 0.0),
                "confidence": content.get("confidence", 0.7),
                "arousal": content.get("arousal", 0.5),
                "valence": content.get("valence", 0.5),
            },
            "coherence": content.get("coherence", 1.0),
            "drift_rate": content.get("drift_rate", 0.0),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_table(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render a data table."""
        content = instruction.get("content", {})
        return {
            "type": "table",
            "headers": content.get("headers", []),
            "rows": content.get("rows", []),
            "caption": content.get("caption", ""),
            "sortable": content.get("sortable", True),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_card(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render an information card."""
        content = instruction.get("content", {})
        if isinstance(content, str):
            content = {"body": content}
        return {
            "type": "card",
            "title": content.get("title", ""),
            "body": content.get("body", ""),
            "icon": content.get("icon"),
            "actions": content.get("actions", []),
            "style": content.get("style", "default"),
            "index": index,
            "timestamp": time.time(),
        }

    def _render_alert(
        self, instruction: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Render an alert/notification."""
        content = instruction.get("content", {})
        if isinstance(content, str):
            content = {"message": content}
        return {
            "type": "alert",
            "level": content.get("level", "info"),
            "message": content.get("message", ""),
            "dismissible": content.get("dismissible", True),
            "index": index,
            "timestamp": time.time(),
        }

    def _emit_render_payload(
        self,
        payloads: list[dict[str, Any]],
        target: str,
        stream_id: str | None,
    ) -> None:
        """Emit processed render payloads via EventBus for real-time UI updates."""
        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            bus.publish_threadsafe(
                "render_instructions",
                {
                    "payloads": payloads,
                    "target": target,
                    "stream_id": stream_id,
                    "timestamp": time.time(),
                    "count": len(payloads),
                },
            )
        except _RENDER_RECOVERABLE_ERRORS as exc:
            _record_render_degradation(
                exc,
                action="failed to emit render payload via EventBus, payloads returned in response only",
                extra={"count": len(payloads), "target": target},
            )
            logger.debug("EventBus render emission failed: %s", exc)

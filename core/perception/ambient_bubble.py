"""Where the bubble is, and who is allowed to move it.

A move is requested, taken, acknowledged and only then believed — four steps
because the surface that draws the bubble and the runtime that remembers where
it is are different processes, and a position written before the draw lands is
a position she will contradict next time she is asked.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from core.runtime.errors import record_degradation


class _CarriesTheBubble:
    """Lifted whole from AmbientPresence; see ambient_presence.py."""

    def move_bubble(self, x: float, y: float) -> tuple[float, float]:
        """Remember where she was parked. In memory; the disk write is async.

        Kept synchronous and trivial because it is called from a request
        handler on the event loop. Durability is ``persist_bubble_position``,
        which the route awaits — an fsync on this loop once froze the live
        runtime for twenty minutes.
        """
        with self._lock:
            self._bubble_position = (float(x), float(y))
            return self._bubble_position

    def bubble_position(self) -> tuple[float, float]:
        """Where she is parked. Read by anything deciding whether she is in the way."""
        with self._lock:
            return self._bubble_position

    def request_bubble_move(self, x: float, y: float) -> int | None:
        """Ask the attached native bubble to move, without pretending it did.

        ``move_bubble`` records a position the host already reached. This is
        the opposite direction: cognition requests a destination, the native
        host clamps and applies it, and its ordinary did-move callback records
        the measured position. The command is one-shot so a poll retry cannot
        make the panel jump repeatedly.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .ambient_presence import (
            PresenceMode,
        )

        try:
            target_x = float(x)
            target_y = float(y)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            return None
        if self.mode is not PresenceMode.BUBBLE or not self.drawing_surface_attached():
            return None
        with self._lock:
            self._bubble_move_sequence += 1
            sequence = self._bubble_move_sequence
            self._pending_bubble_move = {
                "x": target_x,
                "y": target_y,
                "sequence": sequence,
                "queued_at": time.time(),
            }
        return sequence

    def take_bubble_move(self) -> dict[str, float | int] | None:
        """Collect one host movement command exactly once."""
        from .ambient_presence import (
            _HIGHLIGHT_TTL_S,
        )

        with self._lock:
            command = self._pending_bubble_move
            self._pending_bubble_move = None
        if command is None:
            return None
        if time.time() - float(command.get("queued_at", 0.0)) > _HIGHLIGHT_TTL_S:
            return None
        return command

    def acknowledge_bubble_move(
        self, *, sequence: int, x: float, y: float
    ) -> bool:
        """Record AppKit's measured post-move origin for one command.

        The launcher, not cognition, owns screen clamping. Consequently the
        requested coordinates are never accepted as evidence that the panel
        moved: only the origin AppKit reports after ``setFrameOrigin`` closes
        this receipt.
        """
        try:
            ack_sequence = int(sequence)
            measured_x = float(x)
            measured_y = float(y)
        except (TypeError, ValueError):
            return False
        if ack_sequence <= 0 or not (
            math.isfinite(measured_x) and math.isfinite(measured_y)
        ):
            return False
        with self._lock:
            if ack_sequence > self._bubble_move_sequence:
                return False
            self._bubble_move_acks[ack_sequence] = (measured_x, measured_y)
            # Movement is serialized by the desktop executor, but retain a
            # small bounded set so a delayed acknowledgement can never grow
            # this process for the life of the app.
            while len(self._bubble_move_acks) > 16:
                self._bubble_move_acks.pop(min(self._bubble_move_acks))
        return True

    async def wait_for_bubble_move(
        self, sequence: int, *, timeout_s: float = 5.0
    ) -> tuple[float, float] | None:
        """Wait for a native acknowledgement, bounded by the bubble cadence."""
        import asyncio

        try:
            expected = int(sequence)
        except (TypeError, ValueError):
            return None
        deadline = time.monotonic() + max(0.05, min(float(timeout_s), 10.0))
        while time.monotonic() < deadline:
            with self._lock:
                measured = self._bubble_move_acks.pop(expected, None)
                if measured is not None:
                    return measured
            await asyncio.sleep(0.05)
        return None

    async def persist_bubble_position(self) -> bool:
        """Write the parked position so it survives a restart.

        Position was held in memory only and the launcher never read it back,
        so "she can move it, position persists" was true of the drag and false
        of the persistence: every restart put her back in the bottom-left
        corner, including the restarts a person did not choose.
        """
        from .ambient_presence import (
            AMBIENT_SCHEMA,
        )

        with self._lock:
            x, y = self._bubble_position
        try:
            from core.config import DATA_DIR
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            target = Path(DATA_DIR) / "companion" / "bubble_position.json"
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "ambient_presence.persist_bubble_position",
                domain="file_write",
            ):
                # Both calls are the async variants: this runs on the event
                # loop that serves every request, and a sync fsync here is the
                # exact shape of the write that froze the live runtime.
                await gateway.ensure_directory_async(
                    target.parent, source="ambient_presence.bubble_position"
                )
                await gateway.write_json_async(
                    target,
                    {"x": x, "y": y, "saved_at": time.time()},
                    schema_version=1,
                    schema_name=AMBIENT_SCHEMA,
                    source="ambient_presence.bubble_position",
                )
            return True
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence",
                exc,
                severity="debug",
                action=(
                    "bubble position not persisted; she reappears in the "
                    "default corner after a restart"
                ),
            )
            return False

    def _load_bubble_position(self) -> tuple[float, float]:
        """Where she was parked last time, or the origin meaning "unset".

        (0, 0) is the sentinel the launcher reads as "use the default corner",
        so a missing or corrupt file degrades to first-run behaviour rather
        than to a bubble stranded off-screen.
        """
        try:
            from core.config import DATA_DIR

            target = Path(DATA_DIR) / "companion" / "bubble_position.json"
            if not target.is_file():
                return (0.0, 0.0)
            stored = json.loads(target.read_text(encoding="utf-8"))
            # The gateway writes a schema envelope — {"payload": …, "schema":
            # …} — so the coordinates live one level down. Both shapes are
            # accepted because reading only the envelope would silently return
            # the "never parked" sentinel for any file written another way,
            # and a position loader that quietly reports "unset" is exactly
            # the failure this whole change exists to remove.
            payload = stored.get("payload") if isinstance(stored, dict) else None
            if not isinstance(payload, dict):
                payload = stored if isinstance(stored, dict) else {}
            return (float(payload.get("x", 0.0)), float(payload.get("y", 0.0)))
        except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return (0.0, 0.0)

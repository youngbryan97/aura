"""core/body/camera_sensor.py
Camera frame and visual environment sensor.

This used to answer every read with a hard-coded `"disabled_by_policy"` and
`has_optical_feed: False`, having looked at nothing. That is worse than an
absent sensor: it is an authoritative-sounding answer that can never become
wrong, so nothing downstream ever had reason to doubt it — and it reported
"disabled by policy" through every configuration in which the camera was in
fact enabled and streaming.

It now reports what `core.perception.camera_authority` actually observes:
whether a backend exists, whether the owner's switch is on, what macOS says,
and who is holding the device right now.
"""
from typing import Any

from core.body.sensor_registry import BaseSensor


class CameraSensor(BaseSensor):
    """Monitors optical inputs and camera availability."""

    @property
    def name(self) -> str:
        return "camera"

    async def read(self) -> dict[str, Any]:
        try:
            from core.perception.camera_authority import get_camera_authority

            authority = get_camera_authority()
            # Refresh the OS grant here rather than on the capture path:
            # this read is periodic and async, while acquisition happens on
            # OS threads and inside 2-second loops where an AVFoundation
            # probe does not belong.
            await authority.refresh_os_permission()
            state = authority.state()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # "Unknown" is the honest answer when the authority cannot be
            # reached. It is not the same as "off", and reporting it as off
            # is how the constant got here in the first place.
            return {
                "status": "unknown",
                "has_optical_feed": False,
                "last_frame_timestamp": None,
                "error": repr(exc),
            }

        holder = state.get("holder")
        if state["in_use"]:
            status = "streaming"
        elif not state["backend_available"]:
            status = "no_backend"
        elif not state["owner_permission"]:
            status = "disabled_by_owner"
        elif state["os_permission"] is False:
            status = "denied_by_os"
        else:
            status = "idle_available"

        return {
            "status": status,
            "has_optical_feed": bool(state["has_optical_feed"]),
            "last_frame_timestamp": None if holder is None else holder.get("idle_for_s"),
            "acquirable": bool(state["acquirable"]),
            "blockers": list(state["blockers"]),
            "holder": None if holder is None else holder.get("holder"),
            "owner_permission": bool(state["owner_permission"]),
            "os_permission": state["os_permission"],
            "leases_reclaimed": state["leases_reclaimed"],
        }

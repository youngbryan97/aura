"""The one place the camera opens, and the one place that knows its state.

Six call sites across four modules opened `cv2.VideoCapture(0)` directly:
`senses/continuous_vision`, `perception/sensory_integration` (twice),
`perception/sensory_runtime`, and `somatic/sensory_motor_cortex` (twice).
Nothing coordinated them, which produced four separate defects:

  * **The privacy boundary was not enforced.** `camera_allowed()` — the
    owner's in-app toggle — was checked by three sites and not by the other
    two. `sensory_runtime.CameraProvider.capture()` never consulted it at
    all, and `continuous_vision` gated on the `features.camera_enabled`
    build flag instead. So switching the camera off in settings left two
    paths still capturing, which is the worst possible shape for a privacy
    control: visible, reassuring, and partially connected.

  * **No canonical authority.** Two subsystems could hold the device at
    once. On macOS the second one gets frames of nothing, or the first one
    stops getting frames, and neither can tell that the other exists.

  * **No crash recovery.** A holder that died mid-stream left the device
    open with nothing to release it, and every later acquisition failed for
    a reason none of them could name.

  * **Untruthful state.** `core/body/camera_sensor.py` answered every read
    with a hard-coded `"disabled_by_policy"` and `has_optical_feed: False`,
    having looked at nothing. A sensor returning a constant is worse than a
    missing sensor: it is an authoritative-sounding answer that cannot
    become wrong, so nobody thinks to check it.

This module fixes the class. Acquisition runs one ordered gate:

  1. Is a backend usable at all (cv2 importable, not deferred to the sidecar)?
  2. Has the owner allowed the camera in-app?
  3. Is autonomous observation being requested, and did the Will permit it?
  4. Is the device free, or is the current lease stale enough to reclaim?

Every refusal is a named reason with a remedy — the "calibrated abstention"
half. `capture(reason="no_frame_or_permission")` told a caller nothing it
could act on; `CameraDenial(reason="os_permission", remedy=...)` tells the
owner which switch to flip.

The core is synchronous because two of the callers are OS threads, not
coroutines. The macOS TCC probe is async and expensive, so it is not run
per-frame: it is refreshed out of band and consulted from cache, and a TCC
denial that slips past the cache still surfaces as an honest
`os_permission_suspected` at capture time rather than a bare False.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.permission_gates import camera_allowed

logger = logging.getLogger("Perception.CameraAuthority")

# A lease whose holder has not read a frame in this long is presumed dead.
# Long enough that a slow consumer is not evicted mid-stream; short enough
# that a crashed holder does not hold the device until restart.
STALE_LEASE_S = 30.0

# How long a TCC probe result is trusted. The owner can change the grant in
# System Settings at any time, so this must expire; probing per frame would
# put an AVFoundation call in a 2-second loop.
TCC_CACHE_TTL_S = 60.0

# A trusted sidecar still crosses a process boundary. Bound encoded and decoded
# frame sizes before allocation so a corrupt worker reply cannot turn one camera
# read into an unbounded primary-process allocation.
MAX_SIDECAR_JPEG_BYTES = 32 * 1024 * 1024
MAX_SIDECAR_FRAME_PIXELS = 33_554_432  # 8192 x 4096


# ─────────────────────────────────────────────────────────── outcomes


@dataclass(frozen=True)
class CameraDenial:
    """Why the camera did not open, in terms someone can act on."""

    reason: str
    detail: str = ""
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "granted": False,
            "reason": self.reason,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass
class CameraLease:
    """The right to hold the device, and the handle to it."""

    holder: str
    purpose: str
    index: int
    acquired_at: float
    last_used: float
    autonomous: bool = False
    capture: Any = None  # the cv2.VideoCapture, owned by the authority
    last_error: str = ""
    _released: bool = field(default=False, repr=False)

    @property
    def active(self) -> bool:
        return not self._released

    def to_dict(self) -> dict[str, Any]:
        return {
            "holder": self.holder,
            "purpose": self.purpose,
            "index": self.index,
            "held_for_s": round(time.time() - self.acquired_at, 2),
            "idle_for_s": round(time.time() - self.last_used, 2),
            "autonomous": self.autonomous,
            "last_error": self.last_error or None,
        }


@dataclass(frozen=True)
class StillCapture:
    """The best physically supported frame from one bounded camera burst."""

    frame: Any
    jpeg: bytes
    quality: Any
    attempt: int
    attempts: int


# ───────────────────────────────────────────── the out-of-process device


class _SidecarCapture:
    """A camera handle that lives in the sensory worker process.

    Presents the same `isOpened` / `read` / `release` surface as
    `cv2.VideoCapture` so the authority has one code path regardless of
    which side of the process boundary the device is on. The alternative —
    branching on transport at every call site — is how the in-process path
    ended up with five owners in the first place.

    Frames arrive JPEG-encoded and are decoded here, in the caller's
    process through Pillow and NumPy. OpenCV is deliberately not imported:
    the entire purpose of this transport is to keep its AVFoundation stack out
    of Aura's primary macOS process.
    """

    def __init__(self, index: int) -> None:
        self.index = int(index)
        self.last_jpeg: bytes | None = None
        self.width = 0
        self.height = 0
        self.last_error = ""
        self._open = self._call("camera_open", {"index": self.index}).get("status") == "ok"

    def _call(self, command: str, data: Any = None, *, timeout: float = 5.0) -> dict[str, Any]:
        """Drive the async sidecar client from synchronous code.

        Two of the camera's callers are OS threads with no event loop, and
        the authority's whole API is synchronous for that reason.
        """
        try:
            from core.senses.sensory_client import get_sensory_client

            client = get_sensory_client()
            return client.request_sync(command, data, timeout=timeout)
        except (ImportError, RuntimeError, OSError, TimeoutError, AttributeError,
                TypeError, ValueError) as exc:
            record_degradation(
                "camera_authority",
                exc,
                severity="warning",
                action=f"sidecar camera command {command} failed",
            )
            return {"status": "error", "msg": repr(exc)}

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2's spelling
        return self._open

    def read(self) -> tuple[bool, Any]:
        if not self._open:
            self.last_error = "camera_not_open"
            return False, None
        reply = self._call("camera_frame")
        if reply.get("status") != "ok":
            self.last_error = str(reply.get("msg") or "camera_frame_failed")
            return False, None
        payload = reply.get("data")
        if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
            self.last_error = "camera_frame_payload_invalid"
            return False, None
        jpeg = bytes(payload)
        width = int(reply.get("width") or 0)
        height = int(reply.get("height") or 0)
        if (
            len(jpeg) > MAX_SIDECAR_JPEG_BYTES
            or width <= 0
            or height <= 0
            or width * height > MAX_SIDECAR_FRAME_PIXELS
        ):
            self.last_error = "camera_frame_bounds_invalid"
            return False, None
        try:
            import numpy as np
            from PIL import Image

            with Image.open(io.BytesIO(jpeg)) as image:
                if image.format != "JPEG" or image.size != (width, height):
                    self.last_error = "camera_frame_metadata_mismatch"
                    return False, None
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                # Existing camera consumers receive OpenCV BGR arrays. Keep
                # that contract without importing OpenCV in the primary process.
                frame = np.ascontiguousarray(rgb[:, :, ::-1])
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            self.last_error = "camera_frame_decode_failed"
            return False, None
        if frame.shape != (height, width, 3):
            self.last_error = "camera_frame_shape_invalid"
            return False, None
        self.last_jpeg = jpeg
        self.width = width
        self.height = height
        self.last_error = ""
        return True, frame

    def release(self) -> None:
        if self._open:
            self._call("camera_close")
        self._open = False


# ───────────────────────────────────────────────────────── the authority


class CameraAuthority:
    """Single owner of the camera device."""

    def __init__(self) -> None:
        self._lock = checked_lock("camera_authority")
        self._lease: CameraLease | None = None
        self._cv2: Any = None
        self._cv2_checked = False
        self._tcc: tuple[float, dict[str, Any]] | None = None
        # Reclaimed leases are worth counting: a holder that keeps dying is
        # a bug, and it looks identical to a busy camera from outside.
        self._reclaimed = 0
        self._denials: dict[str, int] = {}

    # -- backend ------------------------------------------------------

    def _backend(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        if self._cv2_checked:
            return None
        self._cv2_checked = True
        if importlib.util.find_spec("cv2") is None:
            return None
        if self.sidecar_required():
            # In-process cv2 is banned here; the sidecar owns the device.
            return None
        try:
            import cv2

            self._cv2 = cv2
            return cv2
        except (ImportError, RuntimeError, OSError) as exc:
            record_degradation(
                "camera_authority",
                exc,
                severity="info",
                action="reported the camera as unavailable because cv2 would not import",
            )
            return None

    def sidecar_required(self) -> bool:
        """True when this process must not open the camera itself.

        On macOS the primary process is forbidden from importing cv2 at all:
        faster-whisper pulls in PyAV, and loading OpenCV's AVFoundation
        stack alongside it registers duplicate Objective-C classes and
        crashes. An import guard enforces that by raising.

        Which meant the camera could not be opened ANYWHERE — not in the
        main process, which is banned, and not in the sidecar, which handled
        `init_vision`, `capture_screen`, `init_audio`, `ping`, `exit` and no
        camera command at all, despite `safe_imports` describing it as "the
        production camera path". The sidecar now has `camera_open`,
        `camera_frame` and `camera_close`, and this is where the two paths
        meet.
        """
        try:
            from core.media.safe_imports import cv2_main_process_blocked

            return bool(cv2_main_process_blocked())
        except (ImportError, RuntimeError, AttributeError):
            # The interlock is a safety measure, not a gate. If it cannot be
            # consulted, prefer the in-process path and let the open succeed
            # or fail on its own merits.
            return False

    # -- OS permission ------------------------------------------------

    def note_os_permission(self, payload: dict[str, Any]) -> None:
        """Record a TCC probe result. Called by whoever ran the probe.

        Kept separate from the probe itself so the synchronous acquire path
        never has to run an AVFoundation call, and so the async health
        surfaces that already probe can feed this instead of duplicating it.
        """
        self._tcc = (time.time(), dict(payload or {}))

    def _os_permission(self) -> dict[str, Any] | None:
        if self._tcc is None:
            return None
        recorded_at, payload = self._tcc
        if (time.time() - recorded_at) > TCC_CACHE_TTL_S:
            return None
        return payload

    async def refresh_os_permission(self) -> dict[str, Any]:
        """Probe macOS for the camera grant and cache it."""
        try:
            from core.security.permission_guard import get_permission_guard

            payload = await get_permission_guard()._check_camera_permission()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "camera_authority",
                exc,
                severity="warning",
                action="could not determine the OS camera grant",
            )
            payload = {"granted": None, "status": "unknown", "error": repr(exc)}
        self.note_os_permission(payload)
        return payload

    # -- acquisition --------------------------------------------------

    def acquire(
        self,
        holder: str,
        *,
        purpose: str,
        index: int = 0,
        autonomous: bool = False,
    ) -> CameraLease | CameraDenial:
        """Take the device, or say precisely why not.

        `autonomous=True` means Aura decided to look, rather than the owner
        asking her to. That is a different question and it gets a different
        gate — see `_autonomous_permitted`.
        """
        holder = str(holder or "unknown")

        cv2 = self._backend()
        via_sidecar = cv2 is None and self.sidecar_required()
        if cv2 is None and not via_sidecar:
            return self._deny(
                "no_backend",
                "OpenCV is not importable in this process.",
                "Install opencv-python, or let the perception sidecar own capture.",
            )

        # The owner's switch. Checked BEFORE the device is touched and
        # before any authority question, because no answer to those makes it
        # acceptable to open a camera the owner turned off.
        if not camera_allowed():
            return self._deny(
                "owner_disabled",
                "permissions.camera is off in Aura's settings.",
                "Turn the camera permission back on in Aura's settings.",
            )

        os_grant = self._os_permission()
        if os_grant is not None and os_grant.get("granted") is False:
            return self._deny(
                "os_permission",
                str(os_grant.get("status") or "macOS has not granted camera access."),
                str(os_grant.get("guidance") or "Grant camera access in System Settings → Privacy & Security → Camera."),
            )

        if autonomous:
            permitted, why = self._autonomous_permitted(holder, purpose)
            if not permitted:
                return self._deny(
                    "autonomy_refused",
                    why,
                    "Ask her to look, or adjust the standing directive that refused it.",
                )

        with self._lock:
            existing = self._lease
            if existing is not None and existing.active:
                idle = time.time() - existing.last_used
                if idle < STALE_LEASE_S:
                    return self._deny(
                        "device_busy",
                        f"{existing.holder} has held the camera for "
                        f"{time.time() - existing.acquired_at:.0f}s "
                        f"({existing.purpose}).",
                        "Wait for the current holder to finish, or stop it.",
                    )
                # The holder is gone. Before this existed, a crashed streamer
                # left the device open and every later open failed with no
                # way to find out why.
                logger.warning(
                    "Reclaiming camera from %s: idle %.0fs (presumed dead)",
                    existing.holder,
                    idle,
                )
                self._reclaimed += 1
                record_degradation(
                    "camera_authority",
                    RuntimeError(f"reclaimed a stale camera lease from {existing.holder}"),
                    severity="warning",
                    action="reclaimed the camera device from a holder that stopped reading",
                    extra={"holder": existing.holder, "idle_s": round(idle, 1)},
                )
                self._close(existing)

            if via_sidecar:
                capture = _SidecarCapture(index)
            else:
                try:
                    capture = cv2.VideoCapture(index)
                except (RuntimeError, OSError, AttributeError) as exc:
                    record_degradation("camera_authority", exc, severity="warning",
                                       action="camera open raised")
                    return self._deny("open_failed", repr(exc), "Check that a camera is connected.")

            if capture is None or not capture.isOpened():
                if capture is not None:
                    try:
                        capture.release()
                    except (RuntimeError, OSError, AttributeError):
                        pass
                # The device exists but would not open. On macOS this is
                # overwhelmingly an ungranted TCC prompt, and saying "no
                # frame" instead of naming that is what left owners with a
                # camera that silently did nothing.
                return self._deny(
                    "os_permission_suspected"
                    if os_grant is None
                    else "device_unavailable",
                    "The camera device would not open.",
                    "Grant camera access in System Settings → Privacy & Security → Camera, "
                    "and check no other app is using the camera.",
                )

            now = time.time()
            lease = CameraLease(
                holder=holder,
                purpose=str(purpose or ""),
                index=int(index),
                acquired_at=now,
                last_used=now,
                autonomous=bool(autonomous),
                capture=capture,
            )
            self._lease = lease
            logger.info("Camera acquired by %s (%s)", holder, purpose)
            return lease

    def read(self, lease: CameraLease) -> Any | None:
        """Read one frame and refresh the lease's liveness.

        Going through here rather than touching `lease.capture` is what
        makes stale-lease reclamation correct: an idle lease is one whose
        holder has stopped reading, and only this call knows that.
        """
        if not camera_allowed():
            lease.last_error = "owner_disabled"
            self.release(lease)
            return None
        if not lease.active or lease.capture is None:
            return None
        with self._lock:
            if self._lease is not lease:
                # Reclaimed underneath us. Returning None rather than
                # reading from a released handle.
                return None
            lease.last_used = time.time()
            capture = lease.capture
        try:
            ok, frame = capture.read()
        except (RuntimeError, OSError, AttributeError) as exc:
            record_degradation("camera_authority", exc, severity="warning",
                               action="camera read raised")
            return None
        lease.last_error = "" if ok else str(
            getattr(capture, "last_error", "") or "capture_read_failed"
        )
        return frame if ok else None

    def jpeg_bytes(self, lease: CameraLease, frame: Any) -> bytes:
        """Encode a captured BGR frame without importing OpenCV.

        Sidecar frames already crossed the process boundary as JPEG, so reuse
        those exact bytes. Direct captures are encoded with Pillow. This keeps
        every consumer on one path when OpenCV is intentionally forbidden in
        Aura's primary macOS process.
        """
        if not lease.active or lease.capture is None:
            raise RuntimeError("camera_lease_inactive")
        encoded = getattr(lease.capture, "last_jpeg", None)
        if isinstance(encoded, bytes) and encoded:
            return encoded
        try:
            import numpy as np
            from PIL import Image

            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError(f"unsupported camera frame shape {array.shape}")
            rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
            buffer = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=90)
            payload = buffer.getvalue()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("camera_frame_encode_failed") from exc
        if not payload or len(payload) > MAX_SIDECAR_JPEG_BYTES:
            raise RuntimeError("camera_frame_encode_bounds_invalid")
        return payload

    def capture_best_still(
        self,
        lease: CameraLease,
        *,
        attempts: int = 4,
        settle_s: float = 0.04,
    ) -> StillCapture | None:
        """Select the best frame after bounded exposure/autofocus settling.

        Camera hardware commonly emits a dark or blurred first frame after
        opening. Returning it immediately made a healthy camera look incapable
        of detail. The burst stays short, keeps every read under the canonical
        lease, and ranks only measurable pixel conditions; it does not ask a
        model which answer it prefers.
        """
        from core.perception.frame_quality import assess_frame

        count = max(1, min(8, int(attempts)))
        delay = max(0.0, min(0.25, float(settle_s)))
        best: StillCapture | None = None
        best_score = -1.0
        for attempt in range(1, count + 1):
            frame = self.read(lease)
            if frame is not None:
                quality = assess_frame(frame)
                try:
                    jpeg = self.jpeg_bytes(lease, frame)
                except RuntimeError as exc:
                    lease.last_error = str(exc)
                else:
                    score = quality.evidence_score
                    if score > best_score:
                        best_score = score
                        best = StillCapture(
                            frame=frame,
                            jpeg=jpeg,
                            quality=quality,
                            attempt=attempt,
                            attempts=count,
                        )
            if attempt < count and delay:
                time.sleep(delay)
        return best

    def release(self, lease: CameraLease | None) -> None:
        if lease is None:
            return
        with self._lock:
            self._close(lease)
            if self._lease is lease:
                self._lease = None
        logger.info("Camera released by %s", lease.holder)

    def revoke_owner_permission(self) -> dict[str, Any]:
        """Close the resident handle after the durable owner gate turns off.

        ``acquire`` checks the permission before opening a device, but a holder
        may already own one when the setting changes.  Revocation therefore has
        to close the authority's current lease rather than waiting for every
        consumer to notice the setting independently.
        """
        if camera_allowed():
            raise RuntimeError("camera_owner_permission_is_still_enabled")
        with self._lock:
            lease = self._lease
            holder = lease.holder if lease is not None and lease.active else None
            if lease is not None:
                self._close(lease)
                if self._lease is lease:
                    self._lease = None
        if holder:
            logger.info("Camera permission revoked; released resident holder %s", holder)
        return {"released": holder is not None, "holder": holder}

    def _close(self, lease: CameraLease) -> None:
        lease._released = True
        capture, lease.capture = lease.capture, None
        if capture is None:
            return
        try:
            capture.release()
        except (RuntimeError, OSError, AttributeError) as exc:
            record_degradation(
                "camera_authority",
                exc,
                severity="warning",
                action="camera handle did not release cleanly",
            )

    # -- autonomy -----------------------------------------------------

    def _autonomous_permitted(self, holder: str, purpose: str) -> tuple[bool, str]:
        """Ask the Will before Aura looks at her owner unprompted.

        Fails CLOSED: an unreachable Will means she does not open the camera
        on her own initiative. Owner-requested capture is unaffected — that
        already carries the owner's authority and never reaches here.
        """
        try:
            from core.governance.standing_directives import get_standing_directives

            match, loaded = get_standing_directives().check(
                tool_name="perception:camera_autonomous",
                args={"holder": holder, "purpose": purpose},
                # Not read_only. Pointing a camera at someone is an effect on
                # the world even though nothing is written; classifying it as
                # a read would exempt it from every write-scoped directive
                # the owner has written.
                effect_scope="external_io",
            )
            if loaded.unreadable:
                return False, "the standing-directive store is unreadable"
            if match is not None:
                # Quote the owner's own words back. A refusal that says
                # "policy" teaches nobody which rule to change.
                directive = getattr(match, "directive", None)
                reason = str(getattr(directive, "reason", "") or "").strip()
                value = str(getattr(directive, "value", "") or "").strip()
                matched_on = str(getattr(match, "matched_on", "") or "")
                return False, (
                    f"a standing directive forbids it: {reason or value or matched_on}"
                )
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "camera_authority",
                exc,
                severity="warning",
                action="refused autonomous camera use because directives were unreadable",
                extra={"holder": holder},
            )
            return False, "the standing-directive store could not be consulted"
        return True, ""

    # -- truthful state ----------------------------------------------

    def _deny(self, reason: str, detail: str, remedy: str) -> CameraDenial:
        self._denials[reason] = self._denials.get(reason, 0) + 1
        return CameraDenial(reason=reason, detail=detail, remedy=remedy)

    def state(self) -> dict[str, Any]:
        """What is actually true about the camera right now.

        This is what `core/body/camera_sensor.py` reports instead of the
        constant it used to return.
        """
        with self._lock:
            lease = self._lease
            held = lease.to_dict() if (lease is not None and lease.active) else None

        os_grant = self._os_permission()
        in_process = self._backend() is not None
        via_sidecar = not in_process and self.sidecar_required()
        backend = in_process or via_sidecar
        owner_allows = camera_allowed()

        # "Available" means a capture would be attempted, not that it would
        # succeed — the difference is exactly what the reasons are for.
        blockers: list[str] = []
        if not backend:
            blockers.append("no_backend")
        if not owner_allows:
            blockers.append("owner_disabled")
        if os_grant is not None and os_grant.get("granted") is False:
            blockers.append("os_permission")
        if held is not None:
            blockers.append("device_busy")

        return {
            "backend_available": backend,
            # Which side of the process boundary the device is on. Worth
            # reporting: "no camera" and "camera behind the sidecar, which
            # is not running" look identical from the outside and have
            # entirely different fixes.
            "transport": "in_process" if in_process else ("sidecar" if via_sidecar else "none"),
            "owner_permission": owner_allows,
            "os_permission": (
                None if os_grant is None else os_grant.get("granted")
            ),
            "os_permission_known": os_grant is not None,
            "in_use": held is not None,
            "holder": held,
            "has_optical_feed": held is not None,
            "acquirable": not blockers,
            "blockers": blockers,
            "leases_reclaimed": self._reclaimed,
            "denials": dict(self._denials),
        }


_authority: CameraAuthority | None = None
_authority_lock = checked_lock("camera_authority.singleton", rank=LockRank.REGISTRY)


def get_camera_authority() -> CameraAuthority:
    global _authority
    if _authority is None:
        with _authority_lock:
            if _authority is None:
                _authority = CameraAuthority()
    return _authority


__all__ = [
    "STALE_LEASE_S",
    "CameraAuthority",
    "CameraDenial",
    "CameraLease",
    "StillCapture",
    "get_camera_authority",
]

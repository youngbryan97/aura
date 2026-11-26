"""macOS native-media import boundaries for Aura's primary process."""

from __future__ import annotations

import builtins
import logging
import os
import sys

logger = logging.getLogger("Aura.MediaSafe")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_CV2_IMPORT_GUARD_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__


def _env_true(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in _TRUE_VALUES


def media_sidecar_process_allowed() -> bool:
    """True when media libraries may load inside this process.

    The primary Aura brain process must not load OpenCV on macOS because the
    desktop STT path uses PyAV. The sensory worker is the production process
    boundary for camera/screen media libraries, so it marks itself explicitly.
    """

    return _env_true("AURA_MEDIA_SIDECAR_PROCESS") or _env_true(
        "AURA_ALLOW_INPROCESS_CV2_WITH_STT"
    )


def cv2_main_process_blocked() -> bool:
    """True when the macOS main process should not import OpenCV.

    ``faster-whisper`` depends on PyAV. Loading OpenCV's AVFoundation stack in
    the same process can register duplicate Objective-C classes. The sidecar
    sensory worker remains the production camera path; this only prevents
    unsafe in-process imports.
    """
    if sys.platform != "darwin":
        return False
    if media_sidecar_process_allowed():
        return False
    return True


def torchcodec_main_process_blocked() -> bool:
    """Whether TorchCodec's separately linked FFmpeg is unsafe in this process.

    Sentence-transformers imports TorchCodec only to discover optional audio
    and video input types. In Aura's primary macOS process that probe loads a
    Homebrew FFmpeg beside faster-whisper's PyAV FFmpeg, producing duplicate
    Objective-C AVFoundation classes. Text embeddings need none of it, and
    sentence-transformers already treats ImportError as "modality absent".
    """

    if sys.platform != "darwin":
        return False
    if _env_true("AURA_MEDIA_SIDECAR_PROCESS"):
        return False
    return not _env_true("AURA_ALLOW_INPROCESS_TORCHCODEC_WITH_STT")


def blocked_native_media_import(name: str) -> str | None:
    """Return the violated process boundary for one absolute import."""

    top_level = str(name or "").split(".", 1)[0]
    if top_level == "cv2" and cv2_main_process_blocked():
        return "cv2"
    if top_level == "torchcodec" and torchcodec_main_process_blocked():
        return "torchcodec"
    return None


#: The imports the guard looks at: top-level ``cv2`` and ``torchcodec``, and their submodules.
_GUARDED_PREFIXES = ("cv2", "torchcodec")


def install_main_process_cv2_guard() -> None:
    """Reject native-media imports that collide in the primary process.

    The historic function name is kept because it is part of the boot contract.
    The guard remains narrow: only top-level ``cv2`` and ``torchcodec`` imports
    are affected, only on macOS, and only outside the sensory sidecar.
    """

    global _CV2_IMPORT_GUARD_INSTALLED
    if _CV2_IMPORT_GUARD_INSTALLED:
        return

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        # Only the two guarded names are looked at: every other import, many of them inside functions called each
        # picture of play, goes straight through (offline 2026-10-12 the guard's checks were a fiftieth of a game's play).
        if level != 0 or not name.startswith(_GUARDED_PREFIXES):
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        blocked = blocked_native_media_import(name)
        if blocked == "cv2":
            raise ImportError(
                "OpenCV import is blocked in Aura's primary macOS process; "
                "use the sensory sidecar or set "
                "AURA_ALLOW_INPROCESS_CV2_WITH_STT=1 for diagnostics."
            )
        if blocked == "torchcodec":
            raise ImportError(
                "TorchCodec import is blocked in Aura's primary macOS process "
                "because its FFmpeg collides with STT/PyAV; use the media "
                "sidecar or set AURA_ALLOW_INPROCESS_TORCHCODEC_WITH_STT=1 "
                "for diagnostics."
            )
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    _CV2_IMPORT_GUARD_INSTALLED = True


install_main_process_media_guard = install_main_process_cv2_guard


def prevent_collisions() -> None:
    """Apply environment and import shims for macOS media stability."""
    if sys.platform != "darwin":
        return

    # Patch 24: Fix media stack collisions on macOS
    # Prevents "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized."
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Prevents CV2/AVFoundation deadlocks
    os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1"
    
    logger.info("macOS Media Collision Guards applied.")

# Auto-apply on import
prevent_collisions()

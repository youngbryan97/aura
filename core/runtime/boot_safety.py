from __future__ import annotations

import os
import sys
from typing import Tuple

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def uvloop_allowed(platform: str | None = None) -> bool:
    platform = platform or sys.platform
    if "AURA_ENABLE_UVLOOP" in os.environ:
        return env_flag("AURA_ENABLE_UVLOOP", False)
    return platform != "darwin"


def main_process_camera_policy(
    requested: bool,
    *,
    platform: str | None = None,
) -> Tuple[bool, str]:
    platform = platform or sys.platform
    if not requested:
        return False, "camera disabled"

    if platform == "darwin" and not env_flag(
        "AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", False
    ):
        return (
            False,
            "camera disabled in the macOS main process to avoid cv2/PyAV "
            "AVFoundation crashes; set AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA=1 "
            "to override",
        )

    return True, "camera enabled"

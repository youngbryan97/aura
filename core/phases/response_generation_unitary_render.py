"""Starting a Manim render in the background when a reply carries mathematics or physics.

Lifted whole out of `response_generation_unitary`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _launch_a_manim_render(
    *,
    response_text: Any,
) -> None:
    """Start a Manim render in the background unless one is already running.

    Moved out of ``execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 1 name(s) from the turn and hands back
    0.
    """
    from .response_generation_unitary import (
        _MANIM_RENDER_LOCK,
        _RESPONSE_RECOVERABLE_ERRORS,
        _render_manim_in_background,
        logger,
        threading,
    )

    if not _MANIM_RENDER_LOCK.acquire(blocking=False):
        logger.info(
            "Manim render already in flight; skipping overlapping autonomous render."
        )
    else:
        logger.info(
            "Math/Physics detected in response. Autonomously launching Manim generation..."
        )

        try:
            threading.Thread(
                target=_render_manim_in_background,
                args=(response_text,),
                name="aura-manim-render",
                daemon=True,
            ).start()
            # Nothing is said about it here. The render may
            # produce a file or may produce nothing, and this
            # sentence went out either way — for months it was
            # always "either way", because the render could not
            # start at all. A finished render announces itself
            # on the thought stream, which is a report of
            # something that happened.
        except _RESPONSE_RECOVERABLE_ERRORS:
            _MANIM_RENDER_LOCK.release()
            raise



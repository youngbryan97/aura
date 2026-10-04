"""Playing as it happens, and watching a program to mend it, run where OpenCV may not load.

LIVE 2026-10-04: her primary macOS process refuses OpenCV (core/media/
safe_imports.py), and the repair failed on its first picture with "OpenCV
import is blocked" though every offline test had passed, because no test ran
under the guard. These run in a fresh interpreter with the guard installed,
the way she runs.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent

_UNDER_THE_GUARD = textwrap.dedent(
    """
    import asyncio, sys
    sys.path.insert(0, {root!r})
    from core.media.safe_imports import install_main_process_cv2_guard
    install_main_process_cv2_guard()
    try:
        import cv2  # noqa: F401
    except ImportError:
        pass
    else:
        if sys.platform == "darwin":
            raise SystemExit("the guard did not hold")

    from playwright.async_api import async_playwright
    from core.agency.playing_as_it_happens import play_as_it_happens
    from core.self_modification.watching_a_program_run import what_it_does
    from tools.measure_playing_as_it_happens import TOLD_KEYS, WORLDS, _Page

    async def main():
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={{"width": 800, "height": 600}})
            await page.goto(f"file://{{WORLDS / 'paddle.html'}}?start=1&seed=5")
            box = await page.evaluate(
                "(() => {{ const r = document.querySelector('canvas').getBoundingClientRect();"
                " return [r.left, r.top, r.width, r.height]; }})()"
            )
            came_to = await play_as_it_happens(_Page(page, box).look, _Page(page, box), keys=TOLD_KEYS, seconds=4.0)
            assert came_to["pictures"] > 20, came_to
            watched = await what_it_does(page, f"file://{{WORLDS / 'paddle.html'}}", words="", keys=["up", "down"], seconds=4.0)
            assert watched is not None
            await browser.close()
        print("ran under the guard")

    asyncio.run(main())
    """
)


def test_her_fast_play_and_the_repair_watch_run_where_opencv_is_refused():
    pytest.importorskip("playwright.async_api")
    done = subprocess.run(
        [sys.executable, "-c", _UNDER_THE_GUARD.format(root=str(ROOT))],
        capture_output=True, text=True, timeout=240, cwd=str(ROOT), check=False,
    )
    if "Executable doesn't exist" in done.stderr or "no browser engine" in done.stderr:
        pytest.skip("no browser engine here")
    assert done.returncode == 0 and "ran under the guard" in done.stdout, done.stderr[-3000:]

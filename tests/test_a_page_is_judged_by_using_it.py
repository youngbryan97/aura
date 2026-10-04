"""A program that is not a game is judged by using it: controls that do nothing, errors it throws, loads that fail."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COUNTER = Path(__file__).parent / "fixtures" / "page_repair" / "counter.html"


async def _watch(address: str):
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.self_modification.watching_a_program_run import what_it_does

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        page = await browser.new_page()
        try:
            return await what_it_does(page, address, words="", keys=[], seconds=2.0)
        finally:
            await browser.close()


def test_a_button_that_changes_nothing_and_one_that_throws_are_found():
    behaviour = asyncio.run(_watch(COUNTER.resolve().as_uri()))
    assert "'+'" in behaviour.findings.get("dead", "")
    assert "noteTheReset" in behaviour.findings.get("errors", "")
    assert "failed" in behaviour.right


def test_a_page_that_works_is_found_right(tmp_path):
    fixed = tmp_path / "counter.html"
    fixed.write_text(COUNTER.read_text().replace("step() * 0", "step()").replace(" noteTheReset();", ""))
    behaviour = asyncio.run(_watch(fixed.resolve().as_uri()))
    assert behaviour.findings == {} and {"errors", "dead", "failed"} <= behaviour.right


def test_the_repair_mends_a_control_that_does_nothing(tmp_path):
    pytest.importorskip("playwright.async_api")
    from core.self_modification.repairing_by_behaviour import repair_by_behaviour

    page = tmp_path / "counter.html"
    shutil.copy(COUNTER, page)
    repair = asyncio.run(repair_by_behaviour(page))
    assert repair.kept
    mended = page.read_text()
    assert "count += step();" in mended
    assert (tmp_path / "counter.html.before-repair").exists()

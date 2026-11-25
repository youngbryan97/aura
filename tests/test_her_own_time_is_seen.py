"""What she does on her own time is in the Work panel, beside her works (interface/static/workbench.js)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
_HERS = {
    "doing": {"what": "find out more about limpets", "why": "its turns were worth more"}, "before": {}, "found": 1,
    "learned": {"kinds": {"find_out": {"expected": 0.31, "done": 4}, "tell": {"expected": -0.2, "done": 1}, "goal": {"expected": 0.0, "done": 0}}, "before": -0.1, "lately": 0.2},
    "finding_out": [{"of": "want:limpets", "about": "limpets", "met": 0.5, "done": False, "questions": [
        {"question": "Where does a limpet go <b>at low tide</b>?", "stands": "answered", "answer": "Back to its own scar.", "where": "https://www.example.org/limpets"},
        {"question": "What do limpets dream of?", "stands": "let go", "answer": "", "where": "javascript:alert(1)"},
        {"question": "How long do they live?", "stands": "open", "answer": "", "where": ""}]}],
    "choices": [{"at": 1.0, "what": "find out more about limpets", "kind": "find_out", "why": "expected +0.31", "over": [], "came": "asked, and it is answered", "reward": 0.9},
                {"at": 2.0, "what": "tell them about limpets", "kind": "tell", "why": "expected -0.20", "over": [], "came": "nobody answered", "reward": -0.4}],
}
_PAGE = """<!doctype html><html><head><style>%s</style></head><body><div id="workbench" class="wb wb-wide" style="width:1300px;height:800px"></div><script>
window.asked = [];
const given = %s;
window.fetch = async (url) => { const p = new URL(url, 'http://x').pathname; window.asked.push(p);
  const body = p === '/api/work' ? { works: %s } : p === '/api/her_own_time' ? given : {};
  return { ok: true, json: async () => body }; };
window.EventSource = class { constructor(url) { window.asked.push(url); } addEventListener() {} close() {} };
</script><script>%s</script><script>window.it = mountWorkbench(document.getElementById('workbench'), { wide: true });</script></body></html>"""


def _page(tmp_path, works) -> Path:
    path = tmp_path / "bench.html"
    path.write_text(_PAGE % ((_STATIC / "workbench.css").read_text(encoding="utf-8"), json.dumps(_HERS), json.dumps(works),
                             (_STATIC / "workbench.js").read_text(encoding="utf-8")), encoding="utf-8")
    return path


def _seen(path: Path, then):
    api = pytest.importorskip("playwright.async_api")

    async def go():
        async with api.async_playwright() as started:
            try:
                browser = await started.chromium.launch(headless=True)
            except api.Error as exc:
                pytest.skip(f"no browser: {exc}")
            page = await browser.new_page(viewport={"width": 1340, "height": 840})
            wrong = []
            page.on("pageerror", lambda e: wrong.append(str(e)))
            await page.goto(path.as_uri())
            out = await then(page)
            await browser.close()
            assert wrong == []
            return out

    return asyncio.run(go())


def test_her_own_time_is_chosen_like_a_work_and_shows_what_she_is_finding_out(tmp_path):
    work = {"id": "one", "title": "A clock", "kind": "program", "state": "done", "began": 1, "ended": 2, "now": {}, "stalled": "", "happened": []}

    async def then(page):
        await page.wait_for_selector("[data-work=one]")
        await page.click("[data-work=her-own-time]")
        await page.wait_for_selector(".wb-own-now")
        await page.click(".wb-card[data-of='want:limpets'] summary")
        seen = await page.evaluate("""() => ({ text: document.querySelector('.wb-log').innerText, title: document.querySelector('.wb-title').textContent,
          links: [...document.querySelectorAll('.wb-log a')].map((a) => a.href), bold: document.querySelectorAll('.wb-log b').length,
          foot: getComputedStyle(document.querySelector('.wb-foot')).display, canvas: getComputedStyle(document.querySelector('.wb-canvas')).display,
          state: document.querySelector('.wb-state').textContent.trim() })""")
        streamed = await page.evaluate("window.asked.filter((u) => u.includes('her-own-time'))")
        await page.wait_for_timeout(3300)  # asked again, and what was opened stays open
        still = await page.evaluate('''() => document.querySelector(".wb-card[data-of='want:limpets']").open''')
        await page.click("[data-work=one]")
        await page.wait_for_timeout(100)
        after = await page.evaluate("""() => ({ title: document.querySelector('.wb-title').textContent, text: document.querySelector('.wb-log').innerText,
          foot: getComputedStyle(document.querySelector('.wb-foot')).display })""")
        return seen, streamed, still, after

    seen, streamed, still, after = _seen(_page(tmp_path, [work]), then)
    assert seen["title"] == "Her own time" and seen["state"] == "Doing something"
    assert "Where does a limpet go <b>at low tide</b>?" in seen["text"] and seen["bold"] == 0, "what she asked is shown as words"
    assert "Back to its own scar." in seen["text"] and "1 of 3 answered" in seen["text"] and "Nothing she read says." in seen["text"]
    assert seen["links"] == ["https://www.example.org/limpets"]
    assert "+0.90" in seen["text"] and "−0.40" in seen["text"] and "Finding things out" in seen["text"]
    assert seen["foot"] == "none" and seen["canvas"] == "none" and streamed == [], "it is no work, and is asked for as none"
    assert still is True
    assert "limpet" not in after["text"] and after["foot"] != "none"


def test_it_is_there_where_there_is_no_work(tmp_path):
    async def then(page):
        await page.wait_for_selector("[data-work=her-own-time]")
        await page.click("[data-work=her-own-time]")
        await page.wait_for_selector(".wb-own-now")
        return await page.evaluate("document.querySelector('.wb-log').innerText")

    assert "find out more about limpets" in _seen(_page(tmp_path, []), then)

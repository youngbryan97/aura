"""The panel shows one work's session and nothing of another's, whenever its answers come."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
#: A stand-in for the routes: the first work is slow to answer, and its stream breaks as it opens, so it is asked for by polling.
_PAGE = """<!doctype html><html><body><div id="workbench" class="wb wb-wide" style="width:1200px;height:800px"></div><script>
const rows = (id, n) => Array.from({ length: n }, (_, i) => ({ n: i + 1, at: 1, what: 'code', name: 'draw', language: 'javascript', code: `// ${id} ${i}\\nconst of_${id} = ${i};` }));
const work = (id, after) => ({ id, title: id, kind: 'program', state: 'going', began: 1, ended: 0, now: { what: 'Writing', is_writing: false }, stalled: '', happened: rows(id, 3).filter((r) => r.n > after) });
const late = (ms, body) => new Promise((ok) => setTimeout(() => ok({ ok: true, json: async () => body }), ms));
window.fetch = (url) => {
  const u = new URL(url, 'http://x'); const p = u.pathname.split('/').filter(Boolean);
  if (u.pathname === '/api/work') return late(1, { works: [work('one', 1e9), work('two', 1e9)] });
  if (p[3] === 'program') return late(p[2] === 'one' ? 700 : 20, { parts: { draw: `const of_${p[2]} = 1;` }, language: 'javascript' });
  return late(p[2] === 'one' ? 700 : 20, work(p[2], Number(u.searchParams.get('after') || 0)));
};
window.EventSource = class { constructor(url) { this.l = {}; const u = new URL(url, 'http://x'); const id = u.pathname.split('/')[3]; this.id = id;
  setTimeout(() => { if (this.closed) return; if (id === 'one') this.onerror && this.onerror(); else (this.l.seen || (() => {}))({ data: JSON.stringify(work(id, Number(u.searchParams.get('after') || 0))) }); }, 30); }
  addEventListener(k, f) { this.l[k] = f; } close() { this.closed = true; } };
</script><script>%s</script><script>window.it = mountWorkbench(document.getElementById('workbench'), { wide: true });</script></body></html>"""


@pytest.fixture
def page_text(tmp_path):
    path = tmp_path / "bench.html"
    path.write_text(_PAGE % (_STATIC / "workbench.js").read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_what_is_asked_of_one_work_is_not_shown_in_another(page_text):
    """LIVE 24 November 2026: the code of one session was shown in another. An answer that came after another work was chosen
    was drawn into it, and the same happenings came by both ways and were drawn twice."""
    api = pytest.importorskip("playwright.async_api")

    async def go():
        async with api.async_playwright() as started:
            try:
                browser = await started.chromium.launch(headless=True)
            except api.Error as exc:
                pytest.skip(f"no browser: {exc}")
            page = await browser.new_page()
            await page.goto(page_text.as_uri())
            await page.wait_for_selector("[data-work=one]")
            await page.click("[data-work=one]")
            await page.click("[data-tab=code]")
            await page.wait_for_timeout(1700)  # its stream has broken, and it has been asked for
            await page.click("[data-work=two]")
            await page.wait_for_timeout(2600)
            seen = await page.evaluate("""() => ({ log: document.querySelector('.wb-log').textContent, beside: document.querySelector('.wb-view').textContent,
              cards: document.querySelectorAll('.wb-card.code').length, title: document.querySelector('.wb-title').textContent })""")
            await page.evaluate("document.querySelectorAll('.wb-card.code').forEach((d) => { d.open = true; })")
            await page.wait_for_timeout(100)
            whole = await page.evaluate("document.querySelector('.wb-log').textContent")
            # Back and forth, with what one is slow to say still on its way.
            await page.click("[data-work=one]")
            await page.wait_for_timeout(40)
            await page.click("[data-work=two]")
            await page.wait_for_timeout(1500)
            again = await page.evaluate("() => [document.querySelector('.wb-log').textContent, document.querySelector('.wb-view').textContent, document.querySelectorAll('.wb-card.code').length]")
            await browser.close()
            return seen, whole, again

    seen, whole, again = asyncio.run(go())
    assert seen["title"] == "two" and seen["cards"] == 3, json.dumps(seen)
    assert "of_two" in seen["beside"] and "of_one" not in seen["beside"] and "of_one" not in whole
    assert "of_one" not in again[0] + again[1] and again[2] == 3


def test_what_outlives_a_watch_is_not_shown_in_it():
    """What was begun while a work was watched, and goes on after, is not the work's."""
    from core.brain.llm import what_is_being_written as written

    seen: list[str] = []

    async def go():
        async def later():
            await asyncio.sleep(0.05)
            return written.watch_for("after")

        with written.watched(lambda text, _tokens: seen.append(text)):
            assert written.watch_for("during")
            task = asyncio.ensure_future(later())
            written.pass_on({"id": "during", "written": "mine", "written_at": 0})
        written.pass_on({"id": "during", "written": " and another's", "written_at": 4})
        assert not await task

    asyncio.run(go())
    written.forget("during")
    assert seen == ["mine"]

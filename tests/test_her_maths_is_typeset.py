"""The TeX she writes is typeset in her chat, and nothing else is.

Her replies on maths are full of TeX, and her chat showed every dollar sign: on
13 November 2026 she answered "How fast does light travel in a vacuum, exactly?"
with "the speed of light ($c$) is fixed at that precise value", and the reply
above it carried "$$ 155 \\times 61 = 9455 $$" as written. The renderer now hands
each span to KaTeX, vendored with her other libraries. These run her renderer in
Node with that KaTeX: what is maths is typeset, and a price, a code block and
TeX KaTeX cannot read stay as she wrote them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "interface" / "static"

_HARNESS = r"""
const fs = require('fs');
const [root, casesJson] = process.argv.slice(1);
global.katex = require(root + '/interface/static/vendor/katex/katex.min.js');
global.escHtml = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const src = fs.readFileSync(root + '/interface/static/aura.js', 'utf8');
const body = src.slice(src.indexOf('function renderMessageMarkdown('), src.indexOf('async function appendMsg'));
const renderMessageMarkdown = new Function(body + '\nreturn renderMessageMarkdown;')();
const out = {};
for (const [name, text] of Object.entries(JSON.parse(casesJson))) out[name] = renderMessageMarkdown(text);
process.stdout.write(JSON.stringify(out));
"""


def _render(cases: dict[str, str]) -> dict[str, str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    done = subprocess.run([node, "-e", _HARNESS, str(ROOT), json.dumps(cases)], capture_output=True, text=True,
                          timeout=60, check=True)
    return json.loads(done.stdout)


def test_the_maths_she_writes_is_typeset() -> None:
    out = _render({
        "inline": "Therefore, the speed of light ($c$) is fixed at that precise value.",
        "display": "Now:\n$$ 155 \\times 61 = 9455 $$\ndone",
        "brackets": "So \\[ \\sum_{k=1}^{30} k^2 = 9455 \\] and \\(n = 30\\).",
        "less_than": "When $a < b$ the order holds.",
    })
    for name, html in out.items():
        assert 'class="katex' in html, name
        assert "$" not in html.replace("&#36;", ""), name
    assert 'class="katex-display' in out["display"] and 'class="katex-display' in out["brackets"]


def test_what_is_not_maths_stays_as_she_wrote_it() -> None:
    out = _render({
        "price": "It costs $5 and $10 later.",
        "code": "```python\nprint(\"$x$\")\n```",
        "unreadable": "Broken $\\frac{1}{$ here.",
    })
    assert out["price"] == "It costs $5 and $10 later."
    assert 'class="katex' not in out["code"] and "$x$" in out["code"]
    assert 'class="katex' not in out["unreadable"] and "$\\frac{1}{$" in out["unreadable"]


def test_a_streamed_reply_is_rendered_as_a_finished_one_is() -> None:
    source = (STATIC / "aura.js").read_text(encoding="utf-8")
    stream = source[source.index("function appendStreamChunk("):]
    stream = stream[: stream.index("\n}\n")]
    assert "renderMessageMarkdown(renderText)" in stream


def test_katex_loads_before_her_chat_renders_anything() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert (STATIC / "vendor" / "katex" / "katex.min.js").is_file()
    katex_at = page.index('src="/static/vendor/katex/katex.min.js"')
    assert katex_at < page.index('src="/static/aura.js"')
    assert "defer" not in page[page.rindex("<script", 0, katex_at):katex_at]

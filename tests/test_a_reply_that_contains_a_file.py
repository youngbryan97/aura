"""When the reply contains the file that was asked for, write it down.

LIVE, 2026-08-21. "build me a small web app… Keep it one self-contained file.
Tell me where you put it" was chased through thirteen separate breaks. The
one thing that worked every time was the plain turn: asked directly, she
writes a complete, correct HTML page into the reply in about thirty seconds.
What never worked was the machinery for saving it — a builder that needs a
second code model this host cannot load, called from inside the turn whose
cortex it needs.

So the file comes out of the answer she already gave.
"""

from __future__ import annotations

import asyncio
import stat

import pytest

from core.conversation.requested_artifact import (
    largest_document,
    save_requested_artifact,
)

PAGE = "<!DOCTYPE html>\n<html><body>" + "x" * 300 + "</body></html>"
REPLY = f"Here it is:\n\n```html\n{PAGE}\n```\n"


def test_the_file_is_written_and_named(tmp_path) -> None:
    saved = save_requested_artifact(
        "build me a small web app, one self-contained HTML file", REPLY, root=tmp_path
    )
    assert saved is not None
    assert saved.path.suffix == ".html"
    assert saved.path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


@pytest.mark.parametrize("async_path", [False, True])
def test_saving_a_file_preserves_the_output_folder_permissions(tmp_path, async_path):
    from core.conversation.requested_artifact import save_requested_artifact_async

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    request = "build me a small web app, one self-contained HTML file"
    saved = (asyncio.run(save_requested_artifact_async(request, REPLY, root=shared)) if async_path else
             save_requested_artifact(request, REPLY, root=shared))
    assert saved is not None and saved.path.read_text() == PAGE
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600
    new_root = tmp_path / "new-output"
    created = (asyncio.run(save_requested_artifact_async(request, REPLY, root=new_root)) if async_path else
               save_requested_artifact(request, REPLY, root=new_root))
    assert created is not None and created.path.read_text() == PAGE
    assert stat.S_IMODE(new_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(created.path.stat().st_mode) == 0o600


def test_conversation_deposits_nothing(tmp_path) -> None:
    """An incidental code sample in conversation is not a delivery."""
    assert save_requested_artifact("how are you today?", REPLY, root=tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_a_snippet_is_not_a_file(tmp_path) -> None:
    small = "Try this:\n\n```js\nconst x = 1;\n```"
    assert save_requested_artifact("write me a script", small, root=tmp_path) is None


def test_the_largest_block_wins() -> None:
    reply = "First a taste:\n\n```html\n<b>hi</b>\n```\n\nAnd the whole thing:\n\n```html\n" + PAGE + "\n```"
    language, body = largest_document(reply)
    assert language == "html"
    assert len(body) == len(PAGE)


@pytest.mark.parametrize(
    ("language", "suffix"),
    [("html", ".html"), ("python", ".py"), ("js", ".js"), ("css", ".css"), ("", ".html")],
)
def test_the_name_follows_the_language(tmp_path, language: str, suffix: str) -> None:
    body = PAGE if language in ("", "html") else ("y" * 300)
    reply = f"```{language}\n{body}\n```"
    saved = save_requested_artifact("write me a program file", reply, root=tmp_path)
    assert saved is not None
    assert saved.path.suffix == suffix


def test_an_unknown_language_is_not_guessed_at(tmp_path) -> None:
    reply = "```brainfuck\n" + ("+" * 300) + "\n```"
    saved = save_requested_artifact("write me a program file", reply, root=tmp_path)
    assert saved is not None
    assert saved.path.suffix == ".txt"


@pytest.mark.asyncio
async def test_the_reply_is_her_words_and_the_path_goes_beside_it() -> None:
    """LIVE 17 November 2026: "Saved it to <path>." was appended to her words, after her "I couldn't save or run it
    as a script this turn", and contradicted her."""
    from core.conversation.requested_artifact import saved_this_turn
    from core.conversation.session_scope import conversation_turn_var
    from interface.routes.chat import _save_requested_artifact

    token = conversation_turn_var.set("turn-saving-a-page")
    try:
        out = str(await _save_requested_artifact("build me a self-contained HTML page", REPLY))
        assert out == REPLY, "her words, as she wrote them"
        assert saved_this_turn().endswith(".html")
    finally:
        conversation_turn_var.reset(token)
    assert saved_this_turn() == "", "another turn has saved nothing"


@pytest.mark.asyncio
async def test_a_turn_saves_its_file_once(monkeypatch, tmp_path) -> None:
    """LIVE 20 November 2026: her reply was read twice for what the runtime holds, and the file was written twice, with
    two writes on her record."""
    import core.conversation.requested_artifact as requested_artifact
    import core.conversation.surface_disposition as surface_disposition
    from core.conversation.session_scope import conversation_turn_var
    from interface.routes.chat import _save_requested_artifact

    writes, receipts = [], []

    async def _save(message, reply, **_kwargs):
        writes.append(reply)
        return requested_artifact.SavedArtifact(path=tmp_path / "page.html", language="html", characters=len(reply))

    monkeypatch.setattr(requested_artifact, "save_requested_artifact_async", _save)
    monkeypatch.setattr(surface_disposition, "record_tool_receipt", lambda *a, **k: receipts.append(a))
    token = conversation_turn_var.set("turn-read-twice")
    try:
        for _ in range(2):
            assert str(await _save_requested_artifact("build me a self-contained HTML page", REPLY)) == REPLY
    finally:
        conversation_turn_var.reset(token)
    assert (len(writes), len(receipts)) == (1, 1)


def test_a_short_function_asked_for_in_the_reply_is_not_laid_out_as_a_page(tmp_path) -> None:
    """LIVE 17 November 2026: a three-line function went to disk as an HTML page of the prose around it."""
    reply = ("```python\ndef squares_of_evens(numbers):\n    return [n ** 2 for n in numbers if n % 2 == 0]\n```\n\n"
             + "A few notes on the design choices. " * 12)
    assert save_requested_artifact("Write a short Python function that returns the squares of the even numbers.",
                                   reply, root=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_live_artifact_write_uses_async_governed_gateway(monkeypatch, tmp_path) -> None:
    from core.conversation.requested_artifact import save_requested_artifact_async
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    gateway = get_file_write_gateway()
    real_write = gateway.write_text_async
    observed = []

    async def observing_write(path, text, **kwargs):
        observed.append(get_active_governance())
        await real_write(path, text, **kwargs)

    monkeypatch.setattr(gateway, "write_text_async", observing_write)

    saved = await save_requested_artifact_async(
        "build me a small web app, one self-contained HTML file",
        REPLY,
        root=tmp_path,
    )

    assert saved is not None
    assert saved.path.exists()
    assert len(observed) == 1
    for token in observed:
        assert token is not None
        assert token.source == "conversation.requested_artifact"
        assert token.domain == "state_mutation"


SPLIT = '''Here's the function:

```python
from pathlib import Path
from collections import defaultdict


def group_by_extension(paths: list[str]) -> dict[str, list[str]]:
    """Group file paths by their extension (lowercased)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        ext = Path(p).suffix.lower() or "(none)"
        groups[ext].append(p)
    return dict(groups)
```

And the output when run on these examples:

```python
examples = [
    "/home/user/docs/report.pdf",
    "/home/user/src/main.py",
    "/home/user/src/utils.PY",       # uppercase merges with .py
    "README",                        # no extension at all
]

result = group_by_extension(examples)

for ext, files in sorted(result.items()):
    print(f"{ext:10s}  ({len(files)})")
    for f in files:
        print(f"             {f}")
```

produces:

```
(none)      (1)
             README
.pdf        (1)
             /home/user/docs/report.pdf
.py         (2)
             /home/user/src/main.py
             /home/user/src/utils.PY
```
'''


def test_a_program_written_across_blocks_is_saved_whole(tmp_path):
    """LIVE 20 November 2026: the saved file was the run of her function without the function, and stopped at
    "NameError: name 'group_by_extension' is not defined"."""
    import subprocess
    import sys

    saved = save_requested_artifact("Write a Python function that groups file paths by extension, then run it and "
                                    "show me the output.", SPLIT, root=tmp_path)
    assert saved is not None and saved.path.suffix == ".py"
    ran = subprocess.run([sys.executable, "-I", str(saved.path)], capture_output=True, text=True, timeout=60,
                         check=False)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == SPLIT.split("produces:\n\n```\n")[1].split("```")[0].strip()


def test_a_block_that_needs_nothing_else_is_saved_alone():
    reply = ("A first sketch:\n\n```python\ndef f():\n    return 1\n```\n\nThe whole file:\n\n```python\n"
             "def f():\n    return 2\n\n\nprint(f())\n```\n")
    assert largest_document(reply) == ("python", "def f():\n    return 2\n\n\nprint(f())")

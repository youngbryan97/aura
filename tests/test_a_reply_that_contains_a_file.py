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


def test_the_reply_keeps_the_page_and_gains_the_path() -> None:
    from interface.routes.chat import _save_requested_artifact

    out = str(_save_requested_artifact("build me a self-contained HTML page", REPLY))
    assert "<!DOCTYPE html>" in out
    assert "Saved it to " in out

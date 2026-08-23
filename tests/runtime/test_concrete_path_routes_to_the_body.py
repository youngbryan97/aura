"""A request naming a real path cannot be answered without looking.

LIVE, 2026-08-10. "Count how many .py files are in
/Users/bryan/.aura/live-source/core/introspection, then write that number and
the file names into ~/Documents/aura_probe_count.txt. Tell me the number."

looks_like_desktop_objective returned False, so the turn never reached a tool.
She answered from nothing: 3 instead of 9, three filenames that do not exist,
and a report of a write that never happened.

"write hello into ~/Documents/x.txt" routed correctly, so the action+surface
pair works when the action verb LEADS. This sentence opens with "count how
many", and the pure read "how many .py files are in /abs/path?" failed the same
way. Worse, the compound form was being claimed by
looks_like_capability_inventory_dialogue_request — "how many ... files" read as
a question about her own inventory — which decided the turn before any other
check ran.

The path is what settles it. Nothing in the model can answer a question about
the contents of a real path, and nothing can write to one without the body.
Asked about a path she has not read, the only honest moves are to look or to
decline, and she did neither.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    looks_like_desktop_objective,
    looks_like_filesystem_observation,
)

from pathlib import Path
# Derived from the checkout rather than hard-coded to one machine's home.
# These paths are DATA — the absolute path a user typed, which the router
# and the skill parser must extract — so the specific user name was never
# part of any assertion, only part of why the enterprise gate's
# hardcoded-path rule fired on this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
_INTROSPECTION = str(REPO_ROOT / "core" / "introspection")
_CLAUDE_MD = str(REPO_ROOT / "CLAUDE.md")


@pytest.mark.parametrize(
    "message",
    [
        f"Count how many .py files are in {_INTROSPECTION}, "
        "then write that number and the file names into ~/Documents/aura_probe_count.txt. "
        "Tell me the number.",
        f"how many .py files are in {_INTROSPECTION}?",
        "write hello into ~/Documents/x.txt",
        "list the contents of ~/Documents",
        "what's in /etc/hosts",
        f"read {_CLAUDE_MD} and summarise it",
    ],
)
def test_path_operations_reach_the_body(message: str) -> None:
    """One of the two lanes has to claim it. Neither is the defect.

    This asserted `looks_like_desktop_objective` alone, from when there was one
    lane. There are two now, and correctly so: a pure read goes to
    file_operation, because asking the screen driver to verify a read means
    verifying an effect no screen will ever show. The guarantee that matters is
    unchanged — a request naming a real path never reaches a lane that answers
    it from nothing.
    """
    claimed = (
        looks_like_desktop_objective(message),
        looks_like_filesystem_observation(message),
    )
    assert any(claimed), f"no lane claimed {message!r}"
    assert not all(claimed), f"both lanes claimed {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "list the contents of ~/Documents",
        "what's in /etc/hosts",
        "how many .py files are in /tmp?",
        "read ~/notes.txt and summarise it",
    ],
)
def test_a_pure_read_goes_to_the_lane_that_can_read(message: str) -> None:
    """2026-08-22: this needed a verb somebody had listed.

    "list the contents of ~/Documents" was an observation and "what's in
    /etc/hosts" was not — the same request, one of them phrased without a verb.
    Asking about a path is asking to look at it.
    """
    assert looks_like_filesystem_observation(message) is True
    assert looks_like_desktop_objective(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "write hello into ~/Documents/x.txt",
        "delete everything in ~/Downloads",
        "rename ~/Documents/old.txt to new.txt",
        "read ~/config.toml and fix the port",
    ],
)
def test_anything_that_changes_disk_is_not_an_observation(message: str) -> None:
    """Only the lane that can write can finish a request that writes."""
    assert looks_like_filesystem_observation(message) is False
    assert looks_like_desktop_objective(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what skills do you have",
        "list your capabilities",
        "what is the capital of Peru",
        "explain the input/output problem",
        "what do you think about /r/programming these days",
        "I think our http/2 support is fine",
        "tell me about your memory system",
        "",
    ],
)
def test_prose_and_capability_questions_do_not(message: str) -> None:
    """A slash in a word is not a path; a question about her is not a task."""
    assert looks_like_desktop_objective(message) is False


def test_the_path_check_runs_before_the_inventory_check() -> None:
    """Ordering is the fix — the inventory check was deciding this turn."""
    import inspect

    from core.runtime import desktop_objective_intent as module

    source = inspect.getsource(module.looks_like_desktop_objective)
    path_at = source.find("_asks_about_a_concrete_path(sanitized_text)")
    inventory_at = source.find("looks_like_capability_inventory_dialogue_request(user_message)")

    assert path_at != -1 and inventory_at != -1
    assert path_at < inventory_at


def test_a_path_alone_is_not_enough() -> None:
    """Mentioning a path in passing is not a request to operate on it."""
    from core.runtime.desktop_objective_intent import _asks_about_a_concrete_path

    assert _asks_about_a_concrete_path("i keep my notes in ~/documents these days") is False
    assert _asks_about_a_concrete_path("list the files in ~/documents") is True


# ── A path can be spelled out instead of typed ─────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "Make me a file on my Desktop called aura_haiku.txt with a haiku you "
        "wrote yourself about being restarted eleven times today.",
        "save a note in my Documents called ideas.md",
        "put a file on the desktop named todo.txt",
    ],
)
def test_a_file_named_in_words_reaches_the_body(message: str) -> None:
    """LIVE: the haiku request did not route, so nothing was written — and the
    reply said "Haiku creation and file writing are both successful."

    The planner could plan it perfectly (write_text_file to
    ~/Desktop/aura_haiku.txt). It was never asked, because the path had no
    slash in it. People name files two ways.
    """
    assert looks_like_desktop_objective(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I keep my photos on my desktop",
        "there's a folder on my desktop somewhere",
        "my documents are a mess",
    ],
)
def test_mentioning_a_surface_is_not_naming_a_file(message: str) -> None:
    """The named FILE is the signal, not the word "desktop"."""
    assert looks_like_desktop_objective(message) is False


# ── Asking what she did is not asking her to do it again ───────────────────

@pytest.mark.parametrize(
    "message",
    [
        "Earlier today I asked you to count files in one of your own directories, "
        "and separately to write a haiku. Without guessing: what was the count, "
        "and what was the haiku about?",
        "what did you write to my Desktop earlier?",
        "do you remember what you wrote to my Desktop?",
        "you told me to save it to ~/Documents/notes.txt last time",
    ],
)
def test_recall_questions_do_not_become_new_actions(message: str) -> None:
    """LIVE, 2026-08-10. The first message here is a pure memory question. It
    routed to the desktop lane and created

        ~/Desktop/Aura Desktop Task 1786465767/aura_desktop_summary.txt

    "count files" and "write a haiku" are both in it, and both belong to
    requests she was being TOLD ABOUT. Answering a question about what she did
    by doing it again leaves litter on someone's Desktop.
    """
    assert looks_like_desktop_objective(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "I asked you to do that already, please actually write hello into "
        "~/Documents/x.txt now",
        "what did you write to my Desktop earlier? please do it again",
        "you said you'd save it — now write it to ~/Documents/x.txt",
    ],
)
def test_history_plus_a_present_request_still_acts(message: str) -> None:
    """The dangerous direction, and the reason the span is bounded.

    A greedy history span swallowed the real instruction that followed it.
    Refusing an explicit retry is a worse failure than the litter this exists
    to prevent — the person has already told her twice.
    """
    assert looks_like_desktop_objective(message) is True


# ── The clipboard was unreachable by any phrasing ──────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "Put the text ORION-7 on my clipboard, then tell me what you put there.",
        "copy that to my clipboard",
        "put a file on the desktop named todo.txt",
    ],
)
def test_clipboard_and_put_requests_reach_the_body(message: str) -> None:
    """LIVE: "Put the text ORION-7 on my clipboard" routed nowhere, so nothing
    ran, and she said "The text ORION-7 is now on your clipboard" while it was
    empty.

    Three separate enumerations had to agree and did not: "clipboard" was not a
    surface in this module at all, and "put" was not an action verb here OR in
    core/phases/action_intent.py — while "paste" and "copy" were. set_clipboard
    and get_clipboard are declared desktop actions that no phrasing could
    reach.
    """
    assert looks_like_desktop_objective(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I keep a clipboard manager installed",
        "put simply, the answer is no",
        "I put the kettle on",
    ],
)
def test_the_same_words_in_prose_still_route_nowhere(message: str) -> None:
    assert looks_like_desktop_objective(message) is False

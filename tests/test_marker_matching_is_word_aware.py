"""Keyword markers must be matched as words, and the gate must notice.

Five instances of one defect, each fixed alone before this: "in your own
words" launched Microsoft Word; "notes.txt" opened the Notes app; "the latest
Claude model" opened a browser conversation with Claude instead of searching;
"i dont know what to do next" was classified as a question about the last
hour; "how do you distinguish a real memory from a confabulated one" was
answered as a practical GUI diagnostic.

Every one was `marker in text`, and every fix widened one list. The general
form is that a marker buried inside a longer word belongs to a different word.
"""

from __future__ import annotations

import pytest

from core.conversation.word_markers import names_any, names_marker, which_markers


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        ("i dont know what to do next", "now"),
        ("can you acknowledge that", "now"),
        ("how do you distinguish these", "gui"),
        ("tell me the latest model", "test"),
        ("how do i install the app", "stall"),
        ("tell me about embodied cognition", "died"),
    ],
)
def test_a_marker_inside_another_word_does_not_match(text: str, marker: str) -> None:
    assert not names_marker(text, marker)


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        ("what is happening now", "now"),
        # A stem still claims its own inflections: that is what it is for.
        ("is the server running", "run"),
        ("she stalled on the reply", "stall"),
        ("open the gui", "gui"),
        ("run the tests", "test"),
    ],
)
def test_a_marker_used_as_a_word_still_matches(text: str, marker: str) -> None:
    assert names_marker(text, marker)


def test_an_inflection_is_the_stem_not_a_collision() -> None:
    """"words" IS "word" pluralised, so that match is correct.

    The Microsoft Word launch was a different defect — a proper noun colliding
    with a common one — and it is guarded where app names are read, not here.
    Conflating the two would turn every stem marker off.
    """
    assert names_marker("in your own words", "word")
    assert names_marker("she is rewriting it", "rewrit")


def test_a_phrase_marker_keeps_its_spacing() -> None:
    assert names_any("open chatgpt and have a conversation", ("have a conversation",))
    assert not names_any("conversational memory", ("have a conversation",))


def test_the_reason_can_be_named() -> None:
    """A turn that acts on a marker should be able to say which one."""
    assert which_markers("what is happening now", ("now", "urgent")) == ["now"]
    assert which_markers("i dont know", ("now", "urgent")) == []


# ── the two live misroutes, at their own call sites ──────────────────────────


def test_not_knowing_is_not_a_question_about_the_last_hour() -> None:
    from core.search.research_pipeline import _freshness_window, _query_is_current

    assert not _query_is_current("i dont know what to do next")
    assert not _query_is_current("can you acknowledge that")
    assert _query_is_current("what is happening right now")
    assert _freshness_window("i dont know what to do next") > 60 * 60


def test_distinguishing_memories_is_not_a_gui_diagnostic() -> None:
    from core.conversation.response_reliability import is_practical_diagnostic_turn

    assert not is_practical_diagnostic_turn(
        "how do you distinguish a real memory from a confabulated one"
    )
    assert is_practical_diagnostic_turn("is the gui responding")


def test_the_ratchet_is_wired_and_only_goes_down() -> None:
    import json
    from pathlib import Path

    from tools.lint_marker_matching import BASELINE_PATH, findings

    baseline = json.loads(Path(BASELINE_PATH).read_text(encoding="utf-8"))
    current = {path: len(hits) for path, hits in findings().items()}

    grown = {
        path: (baseline["per_file"].get(path, 0), count)
        for path, count in current.items()
        if count > baseline["per_file"].get(path, 0)
    }

    assert not grown, f"new substring markers: {grown}"


def _scan(tmp_path, body: str) -> list:
    """What the marker linter reports for one file."""
    from tools.lint_marker_matching import prose_vocabulary, scan_file

    source = tmp_path / "m.py"
    source.write_text(body, encoding="utf-8")
    return scan_file(source, prose_vocabulary())


def test_an_undocumented_substring_marker_is_reported(tmp_path) -> None:
    body = 'MARKERS = ("system",)\n\n\ndef f(text):\n    return any(m in text for m in MARKERS)\n'
    hits = _scan(tmp_path, body)
    assert hits, "`system` is a fragment of `filesystem`; this has to be reported"


def test_a_site_that_says_why_it_is_a_substring_is_not_reported(tmp_path) -> None:
    """A status code, a receipt field, a redaction key — the word is compounded.

    Per-line, and not a file allowlist, for the reason every other gate here
    gives: blessing a file also blesses every marker added to it later.
    """
    body = (
        'MARKERS = ("system",)\n'
        "\n"
        "\n"
        "def f(name):\n"
        "    # Substring, deliberately: `name` is a subsystem KEY —\n"
        "    # `filesystem_writes` — where the word is compounded.\n"
        "    return any(m in name for m in MARKERS)\n"
    )
    assert _scan(tmp_path, body) == []


def test_the_phrase_only_clears_what_it_sits_above(tmp_path) -> None:
    """Reach is bounded, so a paragraph cannot clear a line it never mentioned."""
    from tools.lint_marker_matching import REVIEWED_REACH

    filler = "".join(f"    x = {index}\n" for index in range(REVIEWED_REACH + 4))
    body = (
        'MARKERS = ("system",)\n'
        "\n"
        "\n"
        "def f(text):\n"
        "    # Substring, deliberately: about something far above.\n"
        + filler
        + "    return any(m in text for m in MARKERS)\n"
    )
    assert _scan(tmp_path, body), "a distant comment must not clear this line"


def test_the_phrase_has_to_be_in_a_comment(tmp_path) -> None:
    body = (
        'MARKERS = ("system",)\n'
        "\n"
        "\n"
        "def f(text):\n"
        '    note = "substring, deliberately"\n'
        "    return any(m in text for m in MARKERS) and bool(note)\n"
    )
    assert _scan(tmp_path, body), "a string is not a review"

"""What a program is, discerned from every witness to it and from what follows from what it does: her model is one witness, not the last word.

LIVE 2026-10-06 her model's reading of Microsoft Word named nineteen features
and left out find and replace, tables, pictures and spelling, which the
articles it read speak of; the build was whatever that one reading said.
"""
from __future__ import annotations

import pytest

from core.rebuilding.what_a_program_does import Feature, Genome, Source
from core.rebuilding.what_it_is_discerned_to_be import (
    ASKED,
    FOLLOWED,
    MEMORY,
    MODEL,
    ONLINE,
    READ,
    discerned,
)

pytestmark = pytest.mark.unit

_WORD = Source("Microsoft Word", "Microsoft Word is a word processor developed by Microsoft. It has find and replace, a spell checker, tables, "
               "images, bulleted lists and print preview, and documents can be saved as PDF.", "in my own copy of Wikipedia")


def _names(seen) -> list[str]:
    return [f.name for f in seen.genome.features]


def test_each_feature_is_held_with_who_speaks_for_it():
    heard = Genome(name="Quill", what_it_is="a word processor", work="a white page", features=[Feature(name="Bold"), Feature(name="Find")])
    seen = discerned("Microsoft Word", [_WORD], "a word processor I can export to my Desktop", heard)
    assert seen is not None and seen.kind == "word processor" and seen.genome.name == "Quill"
    assert seen.witnesses["Find and replace"] == [READ, MODEL]
    assert ONLINE not in seen.witnesses["Find and replace"]  # her own copy is not the internet
    assert seen.witnesses["Bold, italic and underline"] == [MODEL]
    assert ASKED in seen.witnesses["Save as PDF or Word document"]
    for name in ("Tables", "Pictures", "Spelling, grammar and word count", "Open", "Save"):
        assert name in _names(seen)


def test_what_follows_from_what_it_does_is_in_it_and_said():
    article = Source("Jotter", "Jotter is a note-taking app. Notes can be printed, and have numbered lists.", "on Wikipedia")
    seen = discerned("Jotter", [article], "", None)
    assert seen is not None
    for name, because in (("Undo and redo", "what is typed can be mistyped"), ("Page setup", "what is printed is laid on paper"),
                          ("Indent", "a list is set in from the margin"), ("Cut, copy and paste", "words on a page are moved about")):
        assert name in _names(seen) and seen.witnesses[name] == [FOLLOWED]
        assert any(because in line for line in seen.followed)
    assert any(line.startswith("What it does says more: ") for line in seen.said("Jotter"))


def test_what_no_part_makes_is_said_not_to_be_in_it_with_who_said_it_should_be():
    """Her model proposes from what it knows; what she cannot make is not in the build, and is said to be so, with its witnesses."""
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[
        Feature(name="Mail Merge"), Feature(name="Track Changes"), Feature(name="Bold")])
    article = Source("Word", "Word is a word processor. Its mail merge fills a letter from a list of addresses.", "on Wikipedia")
    seen = discerned("Word", [article], "", heard)
    assert seen.not_yet == {"mail merge": [MODEL, READ], "track changes": [MODEL]}
    said = " ".join(seen.said("Word"))
    assert "My model and what I read say it also has mail merge; my model says it also has track changes." in said
    assert "I do not yet know how to make those, so this build does not have them." in said
    assert "leave" not in said


def test_a_name_mostly_what_a_part_does_is_that_part():
    """No feature is special-cased: a name three-quarters what a part does is that part's, whatever program it is for."""
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[
        Feature(name="Spell-check and Grammar Hints"), Feature(name="Insert Hyperlink")])
    seen = discerned("Word", [Source("Word", "Word is a word processor.", "on Wikipedia")], "", heard)
    assert MODEL in seen.witnesses["Spelling, grammar and word count"] and MODEL in seen.witnesses["Links"]
    assert not seen.not_yet


def test_what_she_built_before_is_a_witness_too():
    before = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[Feature(name="Tables"), Feature(name="Footnotes")])
    seen = discerned("Word", [Source("Word", "Word is a word processor.", "on Wikipedia")], "", None, remembered=[before])
    assert seen.witnesses["Tables"] == [MEMORY] and seen.not_yet == {"footnotes": [MEMORY]}
    assert any(line.startswith("What I built before also had footnotes.") for line in seen.said("Word"))


def test_with_her_model_away_the_others_still_say_what_it_is():
    seen = discerned("Microsoft Word", [_WORD], "", None)
    assert seen is not None and len(seen.genome.features) >= 10
    assert seen.said("Microsoft Word")[0].startswith("From what I read, Microsoft Word is a word processor that does")


def test_with_nothing_written_the_persons_words_say_what_kind_it_is():
    seen = discerned("Inkpot", [], "Build me a simple word processor I can write letters in and print.", None)
    assert seen is not None and seen.kind == "word processor"
    assert {"Print and print preview", "Page setup", "Undo and redo", "Open", "Save"} <= set(_names(seen))


@pytest.mark.parametrize(("program", "article"), [
    ("Microsoft Excel", "Microsoft Excel is a spreadsheet editor. It has find and replace, tables and print preview."),
    ("GIMP", "GIMP is a raster graphics editor with layers, brushes and filters; images can be printed."),
])
def test_a_program_whose_work_is_not_a_page_is_left_to_what_her_model_reads(program, article):
    assert discerned(program, [Source(program, article, "on Wikipedia")], "", None) is None


def test_the_paper_is_the_one_used_where_the_machine_is(monkeypatch):
    monkeypatch.setenv("LC_PAPER", "en_GB.UTF-8")
    assert "A4 page" in discerned("Microsoft Word", [_WORD], "", None).genome.work
    monkeypatch.setenv("LC_PAPER", "en_US.UTF-8")
    assert "Letter page" in discerned("Microsoft Word", [_WORD], "", None).genome.work


def test_what_the_code_of_what_is_in_already_does_is_in_by_that_code():
    """What code begets what a person can do: the page's code zooms and counts pages, so zoom is not something she cannot make."""
    from core.rebuilding.parts_a_maker_knows import _code
    from core.rebuilding.what_it_is_discerned_to_be import what_the_code_does

    assert {"Zoom in", "Zoom out", "Page count shown in the status bar", "Words count shown in the status bar"} <= set(
        what_the_code_does(_code()["document page"]))
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[
        Feature(name="Zoom"), Feature(name="Page count"), Feature(name="Insert table"), Feature(name="Track changes")])
    article = Source("Word", "Word is a word processor with tables, and track changes for review.", "on Wikipedia")
    seen = discerned("Word", [article], "", heard)
    assert seen.by_the_code == {"zoom": "page", "page count": "page"}
    assert "Tables" in [f.name for f in seen.genome.features] and "insert table" not in seen.not_yet
    assert seen.not_yet == {"track changes": [MODEL, ONLINE]} or seen.not_yet == {"track changes": [MODEL, READ]}
    assert "Some of it comes with what I make already, because the code does it: zoom with the page and page count with the page." in seen.said("Word")


def test_a_name_made_of_common_words_is_matched_whole_and_said_in_a_sentence():
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[
        Feature(name="Select All"), Feature(name="Headers and Footers"), Feature(name="Export to PDF")])
    seen = discerned("Word", [Source("Word", "Word is a word processor; pages have headers and footers.", "on Wikipedia")], "", heard)
    assert MODEL in seen.witnesses["Cut, copy and paste"] and "select all" not in seen.not_yet  # select all is the clipboard's
    assert list(seen.not_yet) == ["headers and footers"]


def test_a_readout_is_what_it_counts_in_the_status_bar():
    """What code begets: a part that puts a readout in the frame shows a count there, so "word count in status bar" is the page's."""
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[Feature(name="Word count in status bar")])
    seen = discerned("Word", [Source("Word", "Word is a word processor.", "on Wikipedia")], "", heard)
    assert seen.by_the_code.get("word count in status bar") == "page" or MODEL in seen.witnesses.get("Spelling, grammar and word count", [])
    assert not seen.not_yet

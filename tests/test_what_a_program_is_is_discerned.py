"""What a program is, discerned from every witness to it and from what follows from what it does: her model is one witness, not the last word.

LIVE 2026-10-06 her model's reading of Microsoft Word named nineteen features
and left out find and replace, tables, pictures and spelling, which the
articles it read speak of; the build was whatever that one reading said.
"""
from __future__ import annotations

import pytest

from core.rebuilding.what_a_program_does import Feature, Genome, Source
from core.rebuilding.what_it_is_discerned_to_be import ASKED, FOLLOWED, MODEL, READ, discerned

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


def test_what_her_model_alone_names_is_not_taken_on_its_word():
    heard = Genome(name="Quill", what_it_is="a word processor", work="a page", features=[
        Feature(name="Mail merge"), Feature(name="Holographic preview"), Feature(name="Bold")])
    article = Source("Word", "Word is a word processor. Its mail merge fills a letter from a list of addresses.", "on Wikipedia")
    seen = discerned("Word", [article], "", heard)
    assert seen.not_yet == ["mail merge"]  # borne out by what is written, and no part she knows makes it
    assert seen.left_out == ["holographic preview"]  # her model alone says so
    said = " ".join(seen.said("Word"))
    assert "My model also named holographic preview, which nothing I read bears out, so I leave it out." in said
    assert "What I read also speaks of mail merge, which I do not yet know how to make" in said


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

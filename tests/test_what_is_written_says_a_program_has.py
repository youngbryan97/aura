"""What is written about a program says what it has; her model's list of its features is one reading of that, not the last word.

LIVE 2026-10-06 her model's list of Microsoft Word's features had no find and
replace, tables, pictures or spelling, though the articles it read speak of all
four. Each is a part she knows how to make, and what is written is the evidence
that the program has it. Nothing here knows which program is being built: a
notes app and an email composer are given what their own articles speak of,
and a program whose work is not a page is given none of it.
"""
from __future__ import annotations

import pytest

from core.rebuilding.parts_a_maker_knows import what_is_written_asks_for, what_she_knows_how_to_make
from core.rebuilding.what_a_program_does import Feature, Genome, Source, what_was_read

pytestmark = pytest.mark.unit

_PAGE = "A white page of paper you write on."


def _added(article: str, features: list[str], work: str = _PAGE) -> list[str]:
    genome = Genome(name="It", what_it_is="", work=work, features=[Feature(name=f) for f in features])
    return what_she_knows_how_to_make(genome, [Source("It", article, "on Wikipedia")]).from_what_is_written


@pytest.mark.parametrize(("article", "features", "added", "not_added"), [
    # A word processor whose list left out what its article speaks of.
    ("It offers find and replace, a spell checker and word count, tables, images, headers and margins, and print preview.",
     ["Bold", "Save As", "Print"], ["Find and replace", "Spelling, grammar and word count", "Tables", "Pictures", "Page setup"],
     ["Print and print preview", "Bold, italic and underline"]),
    # A notes app: search and replace and spell-check, said in its own words.
    ("Notes can be searched with search and replace, and a built-in spell-check underlines typos. Notes support bulleted lists.",
     ["New note", "Delete note"], ["Find and replace", "Spelling, grammar and word count", "Bulleted and numbered lists"], ["Tables"]),
    # An email composer: hyperlinks and pictures in the message.
    ("Messages may contain hyperlinks and inline pictures, set in any of several fonts.",
     ["Send", "Attach a file"], ["Links", "Pictures", "Fonts and sizes"], ["Find and replace"]),
])
def test_what_its_article_speaks_of_is_in_it_though_the_list_left_it_out(article, features, added, not_added):
    got = _added(article, features)
    assert all(a in got for a in added), got
    assert not any(n in got for n in not_added), got


def test_words_every_article_has_are_not_evidence():
    article = "A new version was released at a later date, with a link to the store and a list of changes; it can be copied."
    assert _added(article, ["Bold"]) == []


def test_a_program_whose_work_is_not_a_page_is_given_none_of_it():
    article = "Spreadsheets have find and replace, tables of figures, spell check and print preview."
    assert _added(article, ["Sum a column"], work="A grid of cells with lettered columns and numbered rows.") == []


def test_what_its_own_list_has_is_not_added_again():
    got = what_is_written_asks_for([Source("It", "find and replace, tables", "on Wikipedia")], have={"find and replace"})
    assert [part.name for part, _feature in got] == ["tables"]


def test_what_is_added_is_said_with_what_she_knew():
    genome = Genome(name="It", what_it_is="", work=_PAGE, features=[Feature(name="Bold")])
    said = what_she_knows_how_to_make(genome, [Source("It", "It has tables and copy and paste.", "on Wikipedia")]).said()
    assert said.startswith("I already know how to make 1 of these (character styles)")
    assert "What I read says it also has cut, copy and paste; and tables, which I know how to make too" in said


@pytest.mark.parametrize(("sources", "said"), [
    ([Source("Microsoft Word", "", "in my own copy of Wikipedia"), Source("Word processor", "", "in my own copy of Wikipedia")],
     "the articles “Microsoft Word” and “Word processor” in my own copy of Wikipedia"),
    ([Source("GIMP", "", "on Wikipedia")], "the article “GIMP” on Wikipedia"),
    ([Source("Bear", "", "in my own copy of Wikipedia"), Source("Note-taking app", "", "on Wikipedia")],
     "the article “Bear” in my own copy of Wikipedia and the article “Note-taking app” on Wikipedia"),
])
def test_what_she_read_is_said_as_a_person_says_it(sources, said):
    assert what_was_read(sources) == said

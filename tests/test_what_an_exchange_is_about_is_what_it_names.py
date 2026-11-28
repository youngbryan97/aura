"""What an exchange is about is the things it names that her library knows, each read by what the exchange means.

LIVE 27 November 2026 her wants were words: "parameter", "hash", "quiet hours", "this reply", "flights", "too cool".
Bryan: "I dont want to be keywords. We need real systems engineering there."
"""
from __future__ import annotations

import json
import time

import pytest

from core.conation import what_an_exchange_is_about as about
from core.conation.what_an_exchange_is_about import CHANCE_DRAWN, READ_AS, WhatItIsAbout
from core.knowledge.local_corpus import CorpusHit

pytestmark = pytest.mark.unit

#: What the stand-in encoder reads a text as: how often it says each of these.
SENSE = ("tide", "pool", "anemone", "shell", "sea", "lisbon", "portugal", "capital", "city", "flight", "film", "actor",
         "director", "cat", "browser", "web", "page", "game", "moth", "village", "river", "song", "album", "reply")
FILLER = ("moth", "village", "river", "song", "album")


def _vectors(texts):
    out = []
    for text in texts:
        lowered = str(text).lower()
        out.append([float(lowered.count(word)) + 0.01 for word in SENSE])
    return out


class Library:
    """Her library standing in: documents by title, each body a few of the stand-in encoder's words."""

    def __init__(self, documents, *, indexed=True):
        filler = [(f"Place {i}", f"A {FILLER[i % 5]} near a {FILLER[(i + 2) % 5]}" + (" by the sea" if i % 7 == 0 else ""))
                  for i in range(2 * CHANCE_DRAWN)]
        self.docs = {i + 1: (title, body) for i, (title, body) in enumerate([*filler, *documents])}
        self.indexed = indexed

    def has_title_index(self):
        return self.indexed

    def _hit(self, doc_id):
        return CorpusHit(self.docs[doc_id][0], "", "wikipedia", 0.0, doc_id)

    def titled(self, name, *, limit=32):
        name = name.casefold()
        return [self._hit(i) for i, (title, _body) in self.docs.items()
                if title.casefold() == name or title.casefold().startswith((name + " (", name + ", "))][:limit]

    def body(self, doc_id, *, max_chars=20000):
        return self.docs[doc_id][1][:max_chars]

    def largest_id(self):
        return max(self.docs)

    def documents(self, doc_ids):
        return [self._hit(i) for i in doc_ids if i in self.docs]

    def search(self, words, limit=5, **_how):
        used = [(body.lower().count(words.lower()), i) for i, (_title, body) in self.docs.items() if words.lower() in body.lower()]
        return [self._hit(i) for _count, i in sorted(used, reverse=True)[:limit]]


@pytest.fixture(scope="module")
def grammar():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("no English grammar model on this machine")


def _reader(grammar, documents, **how):
    return WhatItIsAbout(store=Library(documents, **how), vectors=_vectors, grammar=grammar)


TIDE = [("Tide pool", "A tide pool is a pool by the sea left by the tide, where an anemone and a shell are found."),
        ("Anemone", "An anemone is an animal of the sea that holds to a shell in a tide pool."),
        ("Reply", "A reply is what is said back to a message or a page."), ("Thanks", "Thanks is a song from an album.")]


def test_what_an_exchange_dwells_on_is_what_it_is_about_and_what_points_at_the_reply_is_not(grammar):
    reader = _reader(grammar, TIDE)
    read = reader.read("Do the tide pools by the sea keep an anemone all year?", "An anemone holds a shell in a tide pool, yes.")
    assert {subject.title for subject in read} >= {"Tide pool", "Anemone"}
    anemone = next(subject for subject in read if subject.title == "Anemone")
    assert anemone.named_in == frozenset({0, 1}) and anemone.chance < about.WRONG
    # "this reply" is the reply at hand, by its determiner; "Thanks" names a song the exchange is not about.
    assert reader.read("That's too cool! Thanks for this reply.", "You're welcome, I'm glad it helped.") == []


LISBON = [("Lisbon", "Lisbon is the capital city of Portugal, with a flight to every city."),
          ("Lisbon (1999 film)", "Lisbon is a film by a director, with an actor on a flight to the city."),
          ("Lisbon (disambiguation)", "Lisbon may refer to: the city, a film.")]


def test_the_primary_sense_of_a_name_stands_unless_the_exchange_sets_another_apart(grammar):
    reader = _reader(grammar, LISBON)
    flights = reader.read("Can you find a cheap flight to Lisbon, the city?", "There is a flight to the capital of Portugal.")
    assert [subject.title for subject in flights] == ["Lisbon"], "the film says flight and city too, and the city is meant"
    film = reader.read("Who was the director of the film Lisbon, and which actor was in the film?", "The director cast an actor.")
    assert [subject.title for subject in film] == ["Lisbon (1999 film)"]
    assert all("disambiguation" not in subject.title for subject in [*flights, *film])


def test_a_name_is_meant_only_where_the_exchange_comes_nearer_its_document_than_its_company(grammar):
    browser = [("Browser (cat)", "Browser is a cat that lived in a library."),
               *[(f"Web {i}", "A web browser shows a web page; a browser game runs in the browser.") for i in range(4)]]
    reader = _reader(grammar, browser)
    read = reader.read("I'd love a game that runs in the browser, on a web page.", "I'll make the game for a web browser.")
    assert "Browser (cat)" not in {subject.title for subject in read}


def test_without_her_librarys_title_index_or_her_encoder_nothing_is_a_subject(grammar, monkeypatch):
    recorded = []
    monkeypatch.setattr(about, "record_degradation", lambda *args, **kwargs: recorded.append(kwargs.get("action")))
    unindexed = _reader(grammar, TIDE, indexed=False)
    assert unindexed.read("The tide pool has an anemone.") == [] and unindexed.read("An anemone again.") == []
    assert len(recorded) == 1, "said once, not on every exchange"
    blind = WhatItIsAbout(store=Library(TIDE), vectors=lambda texts: [], grammar=grammar)
    assert blind.read("The tide pool has an anemone.") == [] and "encoder" in blind.why_not()


def test_what_was_read_is_kept_and_what_was_not_is_read_beside_her(grammar):
    reader = _reader(grammar, TIDE)
    said = ("The tide pool by the sea has an anemone.", "The anemone holds a shell.")
    assert reader.read_already(*said) is None
    first = reader.read(*said)
    assert reader.read_already(*said) == first
    other = ("An anemone in a tide pool by the sea.",)
    assert reader.read_soon(*other) is None, "nothing yet: it is being read beside her"
    for _ in range(200):
        if reader.read_already(*other) is not None:
            break
        time.sleep(0.01)
    assert {subject.title for subject in reader.read_already(*other)} >= {"Anemone"}


def test_the_name_of_a_thing_read_from_her_library_is_its_title_without_its_sense():
    assert about.name_of("Parameter (computer programming)") == "Parameter"
    assert about.name_of("Monterey, California") == "Monterey"
    assert about.name_of("Tide pool") == "Tide pool"


def test_her_record_lets_go_of_the_subjects_it_kept_as_words_once(tmp_path):
    from core.conation.what_she_wants import SCHEMA, Thing, WhatSheWants

    path = tmp_path / "wants.json"
    words = Thing(key="topic:this reply", label="this reply")
    person = Thing(key="person:bryan", label="bryan")
    path.write_text(json.dumps({"schema": SCHEMA, "things": [words.to_dict(), person.to_dict()]}), encoding="utf-8")
    record = WhatSheWants(path)
    assert record.thing("topic:this reply") is None and record.thing("person:bryan") is not None
    assert [row["key"] for row in record._words_let_go] == ["topic:this reply"]
    from core.conation.what_she_wants import _held, _save

    _save(record, _held(record))
    from core.runtime.atomic_writer import flush_writes_behind

    flush_writes_behind()
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["subjects_read_as"] == READ_AS and [row["key"] for row in kept["words_let_go"]] == ["topic:this reply"]
    record.met(["topic:Tide pool"], 0.5, reachable=["topic:Tide pool"], labels={"topic:Tide pool": "Tide pool"})
    _save(record, _held(record))
    flush_writes_behind()
    again = WhatSheWants(path)
    assert again.thing("topic:Tide pool") is not None, "a subject read as a thing is kept as one"

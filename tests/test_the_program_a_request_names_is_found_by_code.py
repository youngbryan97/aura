"""The program a request asks to be remade is found by code: by its capitalised name, else as an encyclopedia knows it.

Asking her model which program was meant was a guess with nothing to check it
against; her corpus has the programs' articles, and a person uses them the same
way: the article about one program, not the article about its kind, and a
word's other meanings looked through for the program it may also mean.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.rebuilding.which_program_is_named import the_program_named_in


@dataclass
class _Hit:
    title: str
    doc_id: str
    source: str = "wikipedia"


class _Encyclopedia:
    ARTICLES = {
        "Microsoft Excel": "thumb|Excel 2021]] Microsoft Excel, or simply Excel, is a spreadsheet editor developed by Microsoft for Windows.",
        "Spreadsheet": "A spreadsheet is a computer application for computation, organization, analysis and storage of data.",
        "Word processor": "A word processor is a device or computer program that provides for input, editing and formatting of text.",
        "Microsoft Word": "Microsoft Word is a word processing program developed by Microsoft.",
        "Word": "A word is a unit of language.",
        "Windows Notepad": "Windows Notepad (commonly known simply as Notepad) is a simple text editor for Windows.",
        "Tetris": "Tetris is a puzzle video game created by Alexey Pajitnov.",
        "Game": "game board inscribed for Amenhotep III]] A game is a structured form of play.",
        "Write": "Writing is the act of creating a record of language.",
        "Excel (disambiguation)": "Excel is a spreadsheet program by Microsoft Corporation.\n\nExcel may also refer to:\n* Excel Airways\n",
        "Notepad (disambiguation)": "A notepad is a pad of paper.\n\nNotepad may also refer to:\n* Windows Notepad, a plain text editor included with Microsoft Windows\n",
        "Word (disambiguation)": "A word is a unit of language.\n\n*Word (computer architecture), a group of bits\n*Microsoft Word, a word-processing application\n",
        "Write (disambiguation)": "Write may refer to:\n* Pfs:Write, a word processor program\n",
    }

    def by_title(self, title):
        return _Hit(title, title) if title in self.ARTICLES else None

    def body(self, doc_id, max_chars=3000):
        return self.ARTICLES[doc_id][:max_chars]

    def search(self, text, limit=5):
        raise AssertionError("a search of the whole corpus is not how a name is found")


def test_a_program_is_found_by_its_name_or_as_the_encyclopedia_knows_it():
    corpus = _Encyclopedia()
    assert the_program_named_in("Do a clean-room reconstruction of Microsoft Word", corpus) == "Microsoft Word"
    assert the_program_named_in("make me my own version of the spreadsheet program excel", corpus) == "Microsoft Excel"
    assert the_program_named_in("i want a copy of notepad that works offline", corpus) == "Windows Notepad"
    assert the_program_named_in("build me a word processor like word", corpus) == "Microsoft Word"
    assert the_program_named_in("make a clone of the game tetris", corpus) == "Tetris"


def test_a_kind_of_program_or_a_plain_word_names_no_program():
    corpus = _Encyclopedia()
    assert the_program_named_in("build me a word processor", corpus) == ""
    assert the_program_named_in("write me a letter", corpus) == ""
    assert the_program_named_in("make a spreadsheet for my budget", corpus) == ""

"""Parts a maker already knows how to make: written once as code, given to any program with a feature they serve, and checked by use.

A person who has built word processors before does not work out Bold from the
beginning every time; they know how it is done, and they check that it works.
Asked of her model part by part, an earlier rebuild of a word processor had 2
of its 18 features working (offline 2026-10-05): each part was a fresh guess at
the edges of an editable page. What a maker knows is here as code instead
(parts_a_maker_knows.js), on the page's editing (document_editing.js) and its
files (document_formats.js), each part with the checks a person would make of
it. Her model writes only what no part here does.

Which part a feature wants is read from the feature's own name, as a maker
reads it: the words that call for a part ("bold", "numbered list", "find"),
not the words that only sound alike ("page numbers" is not a list), and only
when the part does everything the name asks ("spell-check and grammar" is
given the proofing part because it does both). Nothing here knows which
program is being built: a notes app, a letter writer and an email composer ask
for the same parts by their features (tests/test_parts_a_maker_knows.py).
"""
from __future__ import annotations

import datetime
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from core.rebuilding.checks_a_person_makes import Check
from core.rebuilding.the_program_as_built import Part
from core.rebuilding.what_a_program_does import Feature, Genome

__all__ = ["KnownPart", "Known", "PARTS", "parts_for", "the_page_for", "what_she_knows_how_to_make"]

_SOURCE = Path(__file__).with_name("parts_a_maker_knows.js")



_SPELT = {"colour": "color", "colours": "colors", "centre": "center", "centred": "centered", "favourite": "favorite", "organise": "organize",
          "behaviour": "behavior", "grey": "gray", "dialogue": "dialog", "catalogue": "catalog"}


def _stem(word: str) -> str:
    word = _SPELT.get(word, word)
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and re.search(r"(s|x|ch|sh)es$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]+", str(text or "").lower().replace("'s", ""))}


#: Words in a feature's name that ask for nothing by themselves.
_GENERIC_SAID = """
and or the a an of to in on with for from by as at its it your text select selected selection document documents insert inserting
add apply set change use using toggle button buttons menu menus toolbar ribbon option options feature basic simple support supports
tool tools control controls picker pickers dropdown box dialog panel command commands key keys keyboard shortcut shortcuts make create
edit editing editor format formatting via into one more any each all quick quickly easy custom common standard typing typed type
note notes letter letters message messages email emails mail post posts entry entries draft drafts article articles essay story memo
report file files content writing body current
"""
_GENERIC = frozenset(_stem(w) for w in _GENERIC_SAID.split())


@dataclass(frozen=True)
class KnownCheck:
    """A check of a part, and the words of the command it checks: a feature asking for Bold is checked on Bold."""

    words: frozenset[str]
    check: Check


@dataclass(frozen=True)
class KnownPart:
    """A part a maker knows: the words that ask for it, everything it does, words that mean something else, and its checks."""

    name: str
    asked_by: tuple[frozenset[str], ...]
    does: frozenset[str]
    unless: frozenset[str] = frozenset()
    checks: tuple[KnownCheck, ...] = ()

    def asked_for_by(self, name_words: set[str]) -> bool:
        return any(group <= name_words for group in self.asked_by) and not (self.unless & name_words)

    def checks_for(self, feature: Feature) -> list[Check]:
        """This part's checks, for the commands ``feature`` names (all of them where it names none), as that feature's."""
        named = _words(feature.name)
        chosen = [k for k in self.checks if k.words & named] or list(self.checks)
        return [k.check.model_copy(update={"feature": feature.name}) for k in chosen]


def _part(name: str, asked_by: Iterable[str], does: str, *, unless: str = "", checks: Iterable[KnownCheck] = ()) -> KnownPart:
    return KnownPart(name, tuple(frozenset(_words(group)) for group in asked_by), frozenset(_words(does)) | frozenset(_words(name)),
                     frozenset(_words(unless)), tuple(checks))


def _check(words: str, rule: str, steps: list[tuple[str, str, str]], expect: list[tuple[str, str, str, str]]) -> KnownCheck:
    """A check written short: steps as (do, target, value), what is seen as (see, target, property, value)."""
    return KnownCheck(frozenset(_words(words)), Check.model_validate({
        "feature": "", "rule": rule,
        "steps": [{"do": d, "target": t, "value": v} for d, t, v in steps],
        "expect": [{"see": s, "target": t, "property": p, "value": v} for s, t, p, v in expect],
    }))


def _parts() -> tuple[KnownPart, ...]:
    year = str(datetime.date.today().year)
    styled = lambda command, word, prop, value: _check(  # noqa: E731 - one line per character style below
        command, f"select words and press {command}: they are {value}",
        [("type", "", f"Plain and {word}"), ("select_text", word, ""), ("click", command, "")], [("style", word, prop, value)])
    return (
        _part("character styles", ["bold", "italic", "underline", "strikethrough", "strike", "superscript", "subscript", "clear format", "emphasis"],
              "bold italic underline strikethrough strike through superscript subscript clear format character style styles emphasis",
              checks=[styled("Bold", "strong", "font-weight", "bold"),
                      _check("bold", "Mod+B makes the selected words bold", [("type", "", "Keys work too"), ("select_text", "work", ""), ("press", "", "Mod+B")],
                             [("style", "work", "font-weight", "bold")]),
                      styled("Italic", "slanted", "font-style", "italic"), styled("Underline", "marked", "text-decoration", "underline"),
                      styled("Strikethrough", "crossed", "text-decoration", "line-through"), styled("Superscript", "raised", "vertical-align", "super"),
                      styled("Subscript", "lowered", "vertical-align", "sub"),
                      _check("clear format", "clearing formatting makes bold words plain",
                             [("type", "", "Make me plain"), ("select_text", "plain", ""), ("click", "Bold", ""), ("click", "Clear formatting", "")],
                             [("style", "plain", "font-weight", "normal")])]),
        _part("fonts", ["font", "typeface", "text size", "point size"], "font family size typeface face grow shrink larger smaller point fonts type",
              unless="color colour highlight",
              checks=[_check("font family typeface face", "the selected words take the typeface picked",
                             [("type", "", "Set in another face"), ("select_text", "another face", ""), ("choose", "Font", "Georgia")],
                             [("style", "another face", "font-family", "Georgia")]),
                      _check("size", "the selected words take the size picked", [("type", "", "Bigger words"), ("select_text", "Bigger", ""), ("choose", "Font size", "18")],
                             [("style", "Bigger", "font-size", "18pt")])]),
        _part("text colours", ["color", "highlight", "highlighter"], "color text font highlight highlighter marker background fill",
              checks=[_check("color text font", "the selected words take the colour picked",
                             [("type", "", "Red alert"), ("select_text", "Red", ""), ("fill", "Text colour", "#c00000")], [("style", "Red", "color", "#c00000")]),
                      _check("highlight background marker", "the selected words are highlighted in the colour picked",
                             [("type", "", "Look here"), ("select_text", "here", ""), ("fill", "Highlight", "#ffff00")], [("style", "here", "background-color", "#ffff00")])]),
        _part("alignment", ["align", "alignment", "justify", "justification", "center", "centre"], "align alignment paragraph paragraphs line lines left right center centre justify justification text",
              checks=[_check("center centre", "a centred paragraph sits in the middle", [("type", "", "In the middle"), ("click", "Center", "")],
                             [("style", "In the middle", "text-align", "center")]),
                      _check("right", "a paragraph aligned right sits at the right", [("type", "", "Over to the right"), ("click", "Align right", "")],
                             [("style", "Over to the right", "text-align", "right")]),
                      _check("justify justification", "a justified paragraph reaches both edges", [("type", "", "Edge to edge"), ("click", "Justify", "")],
                             [("style", "Edge to edge", "text-align", "justify")]),
                      _check("left", "aligning left again puts it back", [("type", "", "Back again"), ("click", "Align right", ""), ("click", "Align left", "")],
                             [("style", "Back again", "text-align", "left")])]),
        _part("lists", ["list", "bullet", "bullets", "numbering"], "list lists bullet bullets numbered number numbering ordered unordered item items point points",
              unless="page pages line lines footnote",
              checks=[_check("bullet unordered", "a paragraph becomes a bulleted item", [("type", "", "First point"), ("click", "Bulleted list", "")], [("count", "ul li", "", "1")]),
                      _check("number numbered ordered", "a paragraph becomes a numbered item", [("type", "", "Step one"), ("click", "Numbered list", "")],
                             [("count", "ol li", "", "1")])]),
        _part("indent", ["indent", "outdent", "indentation"], "indent indents outdent indentation increase decrease paragraph margin",
              checks=[_check("indent increase", "a paragraph moves in by half an inch", [("type", "", "Pushed in"), ("click", "Increase indent", "")],
                             [("style", "Pushed in", "margin-left", "48px")]),
                      _check("outdent decrease", "decreasing the indent moves it back by as much", [("type", "", "Out again"), ("click", "Increase indent", ""),
                             ("click", "Increase indent", ""), ("click", "Decrease indent", "")], [("style", "Out again", "margin-left", "48px")])]),
        _part("spacing", ["spacing", "line height", "paragraph space", "space after", "space before", "leading"],
              "line lines paragraph paragraphs spacing space height before after leading double single",
              checks=[_check("line spacing height double", "line spacing 2.0 sets the lines twice as far apart",
                             [("type", "", "Spaced out"), ("choose", "Line spacing", "2.0")], [("style", "Spaced out", "line-height", "32px")]),
                      _check("paragraph space after before", "space after a paragraph is added",
                             [("type", "", "Room below"), ("click", "Add space after paragraph", "")], [("style", "Room below", "margin-bottom", "16px")])]),
        _part("paragraph styles", ["heading", "headings", "quote", "blockquote", "paragraph style", "paragraph styles", "title", "code block"],
              "heading headings style styles paragraph quote quotation block blocks blockquote code title normal level levels",
              unless="page",
              checks=[_check("heading style title", "a paragraph becomes a heading", [("type", "", "Chapter one"), ("choose", "Style", "Heading 1")], [("count", "h1", "", "1")]),
                      _check("quote quotation blockquote", "a paragraph becomes a quotation", [("type", "", "Said by someone"), ("choose", "Style", "Quote")],
                             [("count", "blockquote", "", "1")])]),
        _part("history", ["undo", "redo"], "undo redo history revert",
              checks=[_check("undo", "undo takes back the last thing typed", [("type", "", "First words"), ("wait", "", "600"), ("type", "", " and more"),
                             ("wait", "", "600"), ("click", "Undo", "")], [("text", "First words", "", ""), ("no_text", "and more", "", "")]),
                      _check("redo", "redo puts back what undo took", [("type", "", "Kept words"), ("wait", "", "600"), ("type", "", " come back"),
                             ("wait", "", "600"), ("click", "Undo", ""), ("click", "Redo", "")], [("text", "Kept words come back", "", "")])]),
        _part("clipboard", ["cut", "copy", "paste", "clipboard"], "cut copy paste clipboard select all", unless="save file",
              checks=[_check("cut", "cut takes the selected words out", [("type", "", "Keep cut words"), ("select_text", "cut ", ""), ("click", "Cut", "")],
                             [("text", "Keep words", "", "")]),
                      _check("copy paste", "a copied word pastes where the cursor is", [("type", "", "Echo"), ("select_text", "Echo", ""), ("click", "Copy", ""),
                             ("click_text", "Echo", ""), ("click", "Paste", "")], [("text", "EchoEcho", "", "")])]),
        _part("find and replace", ["find", "replace", "search"], "find replace search all match case whole word words", unless="file files",
              checks=[_check("find search", "finding a word says where it is", [("type", "", "needle in a haystack"), ("click", "Find", ""), ("fill", "Find", "haystack")],
                             [("text", "1 of 1", "", "")]),
                      _check("replace", "replacing all puts one word for another everywhere", [("type", "", "cat and cat"), ("click", "Replace", ""),
                             ("fill", "Find", "cat"), ("fill", "Replace with", "dog"), ("click", "Replace all", "")], [("text", "dog and dog", "", "")])]),
        _part("tables", ["table", "tables"], "table tables row rows column columns cell cells grid", unless="content contents",
              checks=[_check("table grid", "a table of the rows and columns asked for is put in", [("click", "Insert table", ""), ("fill", "Rows", "2"),
                             ("fill", "Columns", "3"), ("click", "Insert", "")], [("count", "table td", "", "6")])]),
        _part("pictures", ["picture", "pictures", "image", "images", "photo", "photos", "graphic"], "picture image photo graphic illustration file pictures images photos",
              checks=[_check("picture image photo", "a picture from a file appears in the document", [("give_file", "photo.png", "a picture"),
                             ("click", "Insert picture", ""), ("wait", "", "500")], [("count", "img", "", "1")])]),
        _part("links", ["link", "links", "hyperlink", "hyperlinks"], "link links hyperlink hyperlinks url web address remove",
              checks=[_check("link hyperlink", "the selected words link to the address given", [("type", "", "Visit our site"), ("select_text", "our site", ""),
                             ("click", "Insert link", ""), ("fill", "Address", "https://example.com"), ("click", "OK", "")],
                             [("count", "a[href^='https://example.com']", "", "1")])]),
        _part("insertions", ["page break", "break", "horizontal", "divider", "date", "symbol", "symbols", "special character"],
              "page break breaks horizontal line rule divider date time today symbol symbols special character characters",
              unless="line spacing",
              checks=[_check("page break", "a page break is put in", [("click", "Page break", "")], [("count", "[data-break=page]", "", "1")]),
                      _check("horizontal line rule divider", "a line across the page is put in", [("click", "Horizontal line", "")], [("count", "hr", "", "1")]),
                      _check("date time today", "today's date is put in", [("click", "Date", "")], [("text", year, "", "")]),
                      _check("symbol special character", "a symbol picked is put in", [("click", "Symbol", ""), ("click", "©", "")], [("text", "©", "", "")])]),
        _part("proofing", ["spell", "spelling", "spellcheck", "grammar", "proofread", "proofreading", "proofing", "word count"],
              "spell spelling spellcheck check checker checking grammar proofread proofreading proofing word words count squiggle squiggles typo typos correction suggestion suggestions",
              checks=[_check("grammar proofread proofing check", "a repeated word is found and mended", [("type", "", "This is is a test."), ("click", "Check document", ""),
                             ("click", "Fix", "")], [("text", "This is a test.", "", ""), ("no_text", "This is is", "", "")]),
                      _check("grammar capital", "a lower-case I is found", [("type", "", "Then i went home."), ("click", "Check document", "")],
                             [("text", "always a capital", "", "")]),
                      _check("spell spelling spellcheck", "spelling is checked as it is typed", [("type", "", "Speling")],
                             [("count", "[contenteditable][spellcheck=true]", "", "1")]),
                      _check("word count", "the words are counted", [("type", "", "One two three"), ("click", "Word count", "")], [("text", "Words: 3", "", "")])]),
        _part("printing", ["print", "printing", "preview"], "print printing preview printer page pages",
              checks=[_check("print printing printer", "printing asks the system to print", [("type", "", "On paper"), ("click", "Print", "")], [("printed", "", "", "")]),
                      _check("preview", "print preview shows the document as pages", [("type", "", "Preview me"), ("click", "Print preview", "")],
                             [("dialog", "Print preview", "", ""), ("text", "1 page", "", "")])]),
        _part("new document", ["new", "close"], "new blank empty document start fresh close closing", unless="window tab font folder program app application",
              checks=[_check("new blank", "a new document starts empty, after asking about unsaved changes", [("type", "", "Old words"), ("click", "New document", ""),
                             ("click", "Discard changes", "")], [("no_text", "Old words", "", "")]),
                      _check("close closing", "closing a document leaves an empty page, after asking about unsaved changes",
                             [("type", "", "Words to close"), ("click", "Close document", ""), ("click", "Don't save", "")],
                             [("no_text", "Words to close", "", "")])]),
        _part("save as", ["save as", "export", "pdf", "download"], "save as export pdf desktop format formats download file type kind docx rtf txt odt html markdown word plain rich web page opendocument",
              unless="auto autosave",
              checks=[_check("save format desktop", "saving as a PDF writes a PDF of the document",
                             [("type", "", "Saved as a file"), ("click", "Save as", ""), ("fill", "File name", "Letter"), ("choose", "Format", "PDF"), ("click", "Save", "")],
                             [("download", "Saved as a file", "", ".pdf")]),
                      _check("pdf", "exporting as PDF writes a PDF of the document", [("type", "", "Exported words"), ("click", "Export as PDF", "")],
                             [("download", "Exported words", "", ".pdf")]),
                      # Each kind of file it writes is checked by writing that kind: plain text is not seen working by a PDF.
                      *(_check(words, f"saving as {kind} writes a {suffix} file", [("type", "", f"Kept as {kind}"), ("click", "Save as", ""),
                               ("choose", "Format", label), ("click", "Save", "")], [("download", f"Kept as {kind}", "", suffix)])
                        for words, kind, label, suffix in (("plain txt", "plain text", "Plain Text", ".txt"), ("rtf rich", "rich text", "Rich Text", ".rtf"),
                                                           ("html web", "a web page", "Web Page", ".html"), ("odt opendocument", "OpenDocument", "OpenDocument Text", ".odt"),
                                                           ("markdown md", "Markdown", "Markdown", ".md"))),
                      _check("docx word", "saving as a Word document writes one", [("type", "", "In Word form"), ("click", "Save as", ""),
                             ("choose", "Format", "Word Document"), ("click", "Save", "")], [("download", "In Word form", "", ".docx")])]),
        _part("page setup", ["page setup", "margin", "margins", "orientation", "paper", "page size", "page layout", "landscape"],
              "page setup margin margins orientation paper size layout landscape portrait",
              checks=[_check("page setup paper size layout", "the paper picked is the page's", [("click", "Page setup", ""), ("choose", "Paper", "Legal"), ("click", "OK", "")],
                             [("count", ".doc-page[data-paper=Legal]", "", "1")]),
                      _check("orientation landscape", "the page can be turned on its side", [("click", "Page setup", ""), ("choose", "Orientation", "Landscape"),
                             ("click", "OK", "")], [("count", ".doc-page[data-orientation=landscape]", "", "1")])]),
        _part("tabbed toolbar", ["ribbon", "tabbed toolbar", "toolbar tab"], "ribbon tab tabs tabbed toolbar",
              # Checked on its own tabs and nothing else: LIVE 2026-10-06 this clicked "Insert table" in a program that had no
              # tables, failed there, and her model was asked to write again a ribbon she knows how to make.
              checks=[_check("ribbon tab tabbed", "the tools are kept in tabs, each tab showing its own", [("wait", "", "100")],
                             [("element", "[role=tablist] [role=tab]:nth-of-type(2)", "", ""), ("element", "#app-toolbar .app-group[hidden]", "", "")])]),
    )


PARTS: tuple[KnownPart, ...] = _parts()


@cache
def _code() -> dict[str, str]:
    """Each part's code, by name, from parts_a_maker_knows.js."""
    pieces = re.split(r"^//== (.+?) ==$", _SOURCE.read_text("utf-8"), flags=re.M)
    return {pieces[i].strip(): pieces[i + 1].strip() + "\n" for i in range(1, len(pieces) - 1, 2)}


def parts_for(feature: Feature) -> list[KnownPart]:
    """The parts a feature asks for, when together they do everything its name asks; else none."""
    named = _words(feature.name)
    asked = [p for p in PARTS if p.name not in ("document page", "tabbed toolbar") and p.asked_for_by(named)]
    if not asked:
        return []
    left = named - _GENERIC - frozenset().union(*(p.does for p in asked))
    return asked if not left else []


#: What the work of a program is, where it is a document written on a page.
_A_PAGE = re.compile(r"\b(page|pages|paper|document|documents|letter|letters|essay|manuscript|note|notes|memo|writ\w*|word processor|text editor|compos\w*|email|e-mail|message body|draft\w*)\b", re.I)
_NOT_A_PAGE = re.compile(r"\b(spreadsheet|cells?|grid of|canvas|drawing|paint\w*|slides?|presentation|chart|map|game|board|timeline|track|waveform|terminal|console|calendar|kanban)\b", re.I)
_PAPER = re.compile(r"\b(A4|Letter|Legal)\b")


def the_page_for(genome: Genome) -> Part | None:
    """A document page as the program's work area, where what it works on is a document written on a page."""
    said = f"{genome.work} {genome.what_it_is}"
    if genome.kind.strip().lower() != "application" or not _A_PAGE.search(said) or _NOT_A_PAGE.search(said):
        return None
    paper = _PAPER.search(genome.work)
    return Part("work area", _code()["document page"].replace("__PAPER__", paper.group(1) if paper else ""), ["the work area", "document page"])


#: Toolbar groups the parts put their tools in, under the tab a person looks for them in.
_TABS = {"Home": ["History", "Clipboard", "Font", "Paragraph", "Styles", "Editing"], "Insert": ["Insert"], "Layout": ["Page"],
         "Review": ["Proofing"], "View": ["View"], "File": ["File"]}


def _tabbed(genome: Genome) -> Part:
    named = [t for t in re.findall(r"[A-Z][a-z]+", genome.work) if t in _TABS]
    order = [t for t in named if t != "File"] + [t for t in _TABS if t not in named and t != "File"] + ["File"]
    tabs = json.dumps([[t, _TABS[t]] for t in order])
    return Part("tabbed toolbar", _code()["tabbed toolbar"].replace("__TABS__", tabs), ["tabbed toolbar"])


#: How what is written about a program says it has what a part does, and the feature the part is then offered as.
#: Phrases that mean the feature and nothing else: "tables" is one, "new" and "date" are in every article.
_WRITTEN_AS: dict[str, tuple[str, Feature]] = {
    "find and replace": (r"\b(?:find|search)(?: and |/| & )replace\b|\bfind[- ]and[- ]replace\b",
                         Feature(name="Find and replace", how="Open Find, type a word, replace it everywhere",
                                 shows="where the word is, then every one replaced", place="Edit", weight=2)),
    "proofing": (r"\bspell(?:ing)?[ -]?check\w*|\bgrammar check\w*|\bword count\w*|\bproofread\w*",
                 Feature(name="Spelling, grammar and word count", how="Check the document, or count its words",
                         shows="each mistake with a fix, and how many words there are", place="Tools", weight=2)),
    "tables": (r"\btables\b", Feature(name="Tables", how="Insert a table of so many rows and columns",
                                     shows="a grid of cells on the page to type in", place="Insert", weight=2)),
    "pictures": (r"\b(?:images|pictures|graphics|photographs)\b",
                 Feature(name="Pictures", how="Insert a picture from a file", shows="the picture on the page", place="Insert", weight=2)),
    "links": (r"\bhyperlinks?\b", Feature(name="Links", how="Select words and link them to an address",
                                          shows="the words as a link", place="Insert", weight=1)),
    "insertions": (r"\bpage breaks?\b", Feature(name="Page breaks, dates and symbols", how="Insert a page break, today's date or a symbol",
                                               shows="it where the cursor is", place="Insert", weight=1)),
    "page setup": (r"\bmargins?\b|\bpage (?:layout|setup|size)\b|\bpaper sizes?\b",
                   Feature(name="Page setup", how="Choose the paper, its orientation and margins", shows="the page laid out that way",
                           place="Layout", weight=2)),
    "paragraph styles": (r"\bheadings?\b|\bparagraph styles?\b",
                         Feature(name="Headings and paragraph styles", how="Give a paragraph a style: a heading, a quote",
                                 shows="the paragraph in that style", place="Format", weight=2)),
    "clipboard": (r"\b(?:cut|copy),? and paste\b|\bclipboard\b",
                  Feature(name="Cut, copy and paste", how="Select words, cut or copy them, then paste", shows="the words where they were pasted",
                          place="Edit", weight=2)),
    "indent": (r"\bindent(?:s|ed|ing|ation)?\b", Feature(name="Indent", how="Move a paragraph further in, or back out",
                                                       shows="the paragraph set in from the margin", place="Format", weight=1)),
    "printing": (r"\bprint(?:ing|ed|s)?\b", Feature(name="Print and print preview", how="Preview the pages, then print them",
                                                    shows="the document as pages, then sent to the printer", place="File", weight=2)),
    "history": (r"\bundo\b", Feature(name="Undo and redo", how="Undo the last change, or redo it", shows="the document as it was", place="Edit", weight=2)),
    "save as": (r"\bsave[sd]? (?:as|documents?|files?)\b|\bexport\w*|\bpdf\b",
                Feature(name="Save as PDF or Word document", how="Save as, pick a name and a kind of file: PDF, Word, rich text, plain text",
                        shows="the document saved as that kind",
                        place="File", weight=3)),
    "new document": (r"\b(?:new|blank) documents?\b|\bcreat\w* (?:and|or) edit\w*|\bdocuments\b",
                     Feature(name="New and close", how="Start a new document, or close this one", shows="an empty page, after asking about unsaved changes",
                             place="File", weight=3)),
    "alignment": (r"\balign\w*|\bjustif\w*", Feature(name="Alignment", how="Set a paragraph left, centred, right or justified",
                                                     shows="the paragraph set that way", place="Format", weight=2)),
    "spacing": (r"\bline spacing\b|\bspacing\b|\bleading\b", Feature(name="Line and paragraph spacing", how="Set the space between lines and paragraphs",
                                                                     shows="the lines set further apart or closer", place="Format", weight=2)),
    "text colours": (r"\bcolou?r(?:ed|s)? (?:text|fonts?)\b|\b(?:text|font) colou?rs?\b|\bhighlight\w*",
                     Feature(name="Text colour and highlight", how="Colour or highlight the selected words", shows="the words in that colour", place="Format", weight=1)),
    "character styles": (r"\bbold\b|\bitalics?\b|\bunderlin\w*", Feature(name="Bold, italic and underline", how="Select words and make them bold, italic or underlined",
                                                                           shows="the words in that style", place="Format", weight=3)),
    "fonts": (r"\bfonts?\b|\btypefaces?\b", Feature(name="Fonts and sizes", how="Pick a font and a size for the selected words",
                                                    shows="the words in that font and size", place="Format", weight=3)),
    "lists": (r"\bbullet(?:ed|s)?\b|\bnumbered lists?\b", Feature(name="Bulleted and numbered lists", how="Make paragraphs into a list",
                                                                    shows="each with a bullet or a number", place="Format", weight=2)),
}


def what_is_written_asks_for(sources: Iterable[Any], have: set[str]) -> list[tuple[KnownPart, Feature]]:
    """The parts what is written about a program says it has, besides those in ``have``, each with the feature it is offered as.

    Her model's list of a program's features is one reading of what is written;
    what is written is the evidence. LIVE 2026-10-06 its list of Microsoft
    Word's features had no find and replace, tables, pictures or spelling,
    though the articles it read speak of all four, and each is a part she knows
    how to make: a program built from that list would have lacked them.
    """
    text = " ".join(str(getattr(source, "text", "") or "") for source in sources).lower()
    found = []
    for part in PARTS:
        said = _WRITTEN_AS.get(part.name)
        if said is not None and part.name not in have and re.search(said[0], text):
            found.append((part, said[1].model_copy()))
    return found


#: Names of its own for a program whose work is writing on a page, one chosen for each program rebuilt.
_WRITING_NAMES = ("Quill", "Folio", "Vellum", "Inkwell", "Quire", "Foolscap", "Parchment", "Codex")

#: Where paper is Letter rather than A4, by the country of the machine's language setting.
_LETTER_PAPER = {"US", "CA", "MX", "PH", "CL", "CO", "VE", "GT", "CR", "PA", "DO", "PR"}


def _the_paper_here() -> str:
    """The paper of where this machine is: its language setting's country, as the system keeps it, else the environment's."""
    import locale
    import os
    import plistlib

    said = os.environ.get("LC_PAPER") or os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    try:
        said = said or str(plistlib.loads((Path.home() / "Library/Preferences/.GlobalPreferences.plist").read_bytes()).get("AppleLocale") or "")
    except (OSError, ValueError, plistlib.InvalidFileException):
        pass  # not a Mac, or no setting: the environment and the locale say
    said = said or (locale.getlocale()[0] or "")
    country = re.split(r"[_.@-]", said)[1].upper() if re.match(r"^[a-z]{2}[_-][A-Za-z]{2}", said) else ""
    return "Letter" if country in _LETTER_PAPER else "A4"


@dataclass
class Known:
    """What she knows how to make of a program: parts for its features, their checks, a page to work on, and features it asks for besides."""

    given: dict[str, tuple[Part, list[Check]]] = field(default_factory=dict)
    page: Part | None = None
    added: list[Feature] = field(default_factory=list)
    #: Of what was added, the features what is written says it has.
    from_what_is_written: list[str] = field(default_factory=list)

    def said(self, *, frame: int = 0, left: Iterable[str] = ()) -> str:
        """What she knows how to make of it, with ``frame`` features the application frame gives and those ``left`` to her model."""
        left = list(left)
        written = set(self.from_what_is_written)
        mine = sorted({part.name for name, (part, _) in self.given.items() if name not in written})
        known = len(self.given) - len(written) + frame
        if not left and not written:
            line = (f"I know how to make every one of them: {_and(mine)}" + (", and the frame's own opening and saving" if frame else "")
                    + ". Each is checked by using it, and none is left to my model.")
        else:
            line = (f"I already know how to make {known} of these ({_and(mine)}" + ("; opening and saving are the frame's" if frame else "")
                    + "), and I check each one by using it.")
        if written:
            line += (f" What I read says it also has {_and([n[:1].lower() + n[1:] for n in self.from_what_is_written])}, which I know how to make too, "
                     "so they are in it.")
        if left:
            line += f" My model writes the other {len(left)}, each kept only once it is seen working: {_and(left)}."
        return line


def _and(items: list[str]) -> str:
    """Items as a list is said: "a, b and c"; with semicolons where an item has commas of its own."""
    between = "; " if any("," in item for item in items) else ", "
    return items[0] if len(items) == 1 else between.join(items[:-1]) + (";" if between == "; " else "") + " and " + items[-1] if items else ""


def what_she_knows_how_to_make(genome: Genome, sources: Iterable[Any] = ()) -> Known:
    """For each feature a part she knows does, that part and its checks; a page, where the work is a document; tabs, where its tools are said to be in them.

    Every part here works on a document, so none is given to a program whose
    work is something else: a drawing's Undo is not a page's.
    """
    known = Known(page=the_page_for(genome))
    if known.page is None:
        return known
    code = _code()
    for feature in genome.features:
        parts = parts_for(feature)
        if not parts:
            continue
        name = parts[0].name if len(parts) == 1 else " and ".join(p.name for p in parts)
        part = Part(name, "\n".join(code[p.name] for p in parts), [feature.name])
        known.given[feature.name] = (part, [c for p in parts for c in p.checks_for(feature)])
    have = {p.name for feature in genome.features for p in parts_for(feature)}
    taken = {f.name.lower() for f in genome.features}
    for part, feature in what_is_written_asks_for(sources, have):
        if feature.name.lower() in taken:
            continue
        known.added.append(feature)
        known.from_what_is_written.append(feature.name)
        known.given[feature.name] = (Part(part.name, code[part.name], [feature.name]), part.checks_for(feature))
    if re.search(r"\b(ribbon|tabs|tabbed)\b", genome.work, re.I):
        feature = Feature(name="Tabbed toolbar", how="Click a tab above the toolbar", shows="the tools of that tab", place="View", weight=1)
        known.added.append(feature)
        tabs = next(p for p in PARTS if p.name == "tabbed toolbar")
        known.given[feature.name] = (_tabbed(genome), tabs.checks_for(feature))
    return known

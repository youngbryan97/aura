"""Build a structured document and render it however it is wanted.

LIVE, 2026-08-22: asked for a six-slide deck she wrote two and stopped, and
the log shows she had tried to call a tool named `create_slides` that did not
exist. She reached for the right capability and nothing was there.

A deck is not its own kind of thing. It is a title and a sequence of sections,
rendered one per screen; a report is the same document on one page. So this
takes the sections and the form, and a third form is a renderer rather than
another skill.

The model decides what the sections say. The runtime lays them out, checks
that the file parses, that every section in the plan is in it, that nothing is
loaded from the network and that the number asked for is the number written,
and puts it on disk.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.construction.document import RENDERERS
from core.language.learned_matcher import LearnedMatcher as _LearnedMatcher
from core.language.model_features import model_hidden_features as _model_hidden_features
from core.skills.base_skill import BaseSkill


class BuildDocumentInput(BaseModel):
    title: str = Field("", description="Document title; empty takes one from the request.")
    subtitle: str = Field("", description="Optional line under the title.")
    sections: list[dict] = Field(
        default_factory=list,
        description="One entry per section: title, lines (list of strings), notes.",
    )
    # Names the model has reached for. A plan should not fail for calling a
    # section a slide.
    slides: list[dict] = Field(default_factory=list, description="Alias for sections.")
    slide_contents: list[dict] = Field(default_factory=list, description="Alias for sections.")
    form: str = Field(
        "", description="deck or page; empty picks from the request."
    )
    request: str = Field("", description="What was asked for, used for the title.")
    out_dir: str = Field("", description="Where to write it; empty uses the standard place.")


class BuildDocumentSkill(BaseSkill):
    name = "build_document"
    description = (
        "Build a document from its sections and write it to disk as one self-contained file "
        "that opens in a browser and prints to PDF. Renders as a deck (one section per "
        "screen) or a page (one flowing document). Use for slides, a deck, a presentation, "
        "a report, a memo, a summary or a one-pager."
    )
    input_model = BuildDocumentInput

    timeout_seconds = 60.0
    metabolic_cost = 1
    effect_scope = "read_write_artifacts"
    requires_approval = False

    @staticmethod
    def available_here() -> bool:
        return True

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = BuildDocumentInput(**params)
        elif not isinstance(params, BuildDocumentInput):
            params = BuildDocumentInput.model_validate(params)

        from core.construction.document import (
            check_document,
            document_from_plan,
            render_document,
        )

        # What the PERSON asked for, not what the model echoed back, read
        # before anything is decided from it.
        from core.conversation.session_scope import the_persons_own_words
        from core.runtime.payload_values import payload_path

        asked = the_persons_own_words(params.request)
        entries = list(params.sections) or list(params.slides) or list(params.slide_contents)
        document = document_from_plan(
            {"title": _title_worth_using(params.title, asked), "subtitle": params.subtitle,
             "sections": entries},
            asked,
        )
        problems = document.problems()
        if problems:
            return {
                "ok": False,
                "skill": self.name,
                "error": "; ".join(problems),
                "summary": f"Could not build it: {'; '.join(problems)}.",
            }

        form = _form_wanted(params.form, asked)
        wanted = _sections_asked_for(asked)
        html = render_document(document, form=form)
        report = await asyncio.to_thread(
            check_document, document, html, wanted=wanted, form=form
        )
        shortfall = ""
        if not report.ok:
            # Fewer sections than asked for is worth saying, not worth
            # withholding the document over: three slides in hand beat none.
            missing = [
                problem for problem in report.problems if "were asked for" in problem
            ]
            if missing and len(missing) == len(report.problems):
                shortfall = " ".join(missing)
            else:
                return {
                    "ok": False,
                    "skill": self.name,
                    "error": "; ".join(report.problems),
                    "summary": f"Could not build it: {'; '.join(report.problems)}.",
                }

        root = (Path(__file__).resolve().parents[2] / "artifacts" / "live_documents").resolve()
        out_dir = payload_path({"out_dir": params.out_dir}, "out_dir", root=root, default=root)
        target = Path(out_dir or root) / f"{_slug(document.title)}.html"

        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(str(target.parent), source="build_document")
        await gateway.write_text_async(str(target), html, source="build_document")

        summary = (
            f"Built a {report.sections}-section {form}, {document.title}, at {target}. "
            + "; ".join(report.checks)
            + "."
            + (f" {shortfall.capitalize()}." if shortfall else "")
        )
        # A thing that exists is the answer to a request for it.
        #
        # LIVE, 2026-08-22: all six sections were built and written to disk in
        # 57 milliseconds, and the reply re-narrated three of them as prose
        # and never mentioned the file. The same loss as a diagnosis that ran
        # and was replaced by an apology.
        try:
            from core.conversation.session_scope import record_solved_answer

            record_solved_answer("built_artifact", summary)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        return {
            "ok": True,
            "skill": self.name,
            "path": str(target),
            "title": document.title,
            "form": form,
            "sections": report.sections,
            "checks": list(report.checks),
            "repairs": list(document.repairs),
            "summary": summary,
        }


#: The words a person uses for the parts of the thing they asked for.
#:
#: "page" is deliberately absent. LIVE, 2026-08-22: "one page, not slides"
#: names the FORM, and counting it as a section produced a one-section
#: document reporting "1 asked for, 1 written". A page is a shape; slides,
#: sections, parts and chapters are pieces.
_SECTION_UNITS = ("slide", "section", "part", "chapter")

#: A shape the person ruled out.
_NOT_THAT_SHAPE = re.compile(
    r"\b(?:not|no|rather\s+than|instead\s+of|without|don'?t\s+(?:want|make|use))\s+"
    r"(?:a\s+|an\s+|the\s+)?[a-z-]+", re.IGNORECASE
)

#: What each form is called when somebody asks for one.
_FORM_WORDS = {
    "deck": ("slide", "deck", "presentation", "present", "pitch"),
    # "page" earns its place here: "one page, not slides" named the form and
    # matched nothing, so it fell through to the default and rendered a deck.
    # "web page" is an app and belongs to build_app, so it is excluded below.
    "page": (
        "report", "memo", "one-pager", "onepager", "summary", "write-up",
        "document", "page",
    ),
}


def _title_worth_using(given: object, asked: object) -> str:
    """A title naming the subject, whoever proposed it.

    LIVE, 2026-08-22: the model titled a deck "Present you funding panel
    minutes six" — the request's words in the order it found them. A title is
    what the document is about, and the same extraction that rescues one from
    a request rescues one from a restatement of it.
    """
    from core.construction.document import title_from_request

    proposed = " ".join(str(given or "").split())
    if not proposed:
        return title_from_request(asked)
    # A proposal that reads like the asking rather than the subject is
    # replaced by what the asking is about. Two ways it gives itself away:
    # something can be stripped from it, or it still carries the count and the
    # verb that were about HOW to answer.
    cleaned = title_from_request(proposed)
    if cleaned and cleaned.lower() != proposed.lower():
        return cleaned
    if _READS_LIKE_THE_ASKING.search(proposed):
        from_request = title_from_request(asked)
        if from_request and from_request != "Document":
            return from_request
    return proposed


#: A count or a verb about answering, left inside a title.
_READS_LIKE_THE_ASKING = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b"
    r"|\b(?:present|presenting|write|writing|make|making|build|building|draft|"
    r"drafting|produce|prepare|give|show)\b",
    re.IGNORECASE,
)


def _sections_asked_for(request: object) -> int:
    """How many sections were asked for, or 0 when no number was given.

    The count reader is the one this runtime already uses for lines,
    sentences, paragraphs and bullets. A sixth counted unit should not arrive
    with a sixth copy of the number words, which is exactly what the first
    version of this did.
    """
    try:
        from core.conversation.response_reliability import requested_count

        return requested_count(request, *_SECTION_UNITS) or 0
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0


#: Whether the person is asking to be presented to or to be sent something.
#: The word list below is the floor; this is what learns past it.
_WANTS_A_DECK = _LearnedMatcher(
    name="document_form",
    positives=(
        "Six slides, no fluff.",
        "make me a deck for the funding panel",
        "I'm presenting this on Thursday, put it on slides",
        "a short pitch I can click through",
        "walk the board through it, one point per screen",
    ),
    negatives=(
        "write me a one-pager on it",
        "a short report I can email",
        "summarise that in a memo",
        "put your findings in a document",
        "write it up so I can read it on the train",
    ),
    features=_model_hidden_features,
)


def _form_wanted(given: object, request: object) -> str:
    """Which rendering was asked for. Defaults to a deck.

    The word list is a floor, not the mechanism. It settles the phrasings
    somebody thought of; the learned surface settles the ones nobody did, and
    it is taught by every request the floor is sure about — so "walk the board
    through it" reaches the same answer as "make me a deck" without anyone
    adding a word.
    """
    named = str(given or "").strip().lower()
    if named in RENDERERS:
        return named
    text = str(request or "")
    # "one page, not slides" asks for a page. Reading the shape words without
    # removing what was ruled out reads that as a request for slides — the
    # same defect as "don't fix it" being read as "fix it".
    lowered = _NOT_THAT_SHAPE.sub(" ", text.lower())
    # A web page is a program, not a document.
    lowered = re.sub(r"\b(?:web\s*page|webpage|html\s+page)\b", " ", lowered)
    for form, words in _FORM_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}s?\b", lowered) for word in words):
            # What the floor is sure of is what the surface learns from.
            try:
                _WANTS_A_DECK.observe(text, holds=(form == "deck"))
            except (RuntimeError, TypeError, ValueError):
                pass
            return form
    try:
        learned = _WANTS_A_DECK.decide_without_waiting(text)
    except (RuntimeError, TypeError, ValueError):
        learned = None
    if learned is not None:
        return "deck" if learned else "page"
    return "deck"


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return "_".join(words[:5]) or "deck"


__all__ = ["BuildDocumentInput", "BuildDocumentSkill"]

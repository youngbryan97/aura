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
        from core.runtime.payload_values import payload_path

        entries = list(params.sections) or list(params.slides) or list(params.slide_contents)
        document = document_from_plan(
            {"title": params.title, "subtitle": params.subtitle, "sections": entries},
            params.request,
        )
        problems = document.problems()
        if problems:
            return {
                "ok": False,
                "skill": self.name,
                "error": "; ".join(problems),
                "summary": f"Could not build it: {'; '.join(problems)}.",
            }

        form = _form_wanted(params.form, params.request)
        wanted = _sections_asked_for(params.request)
        html = render_document(document, form=form)
        report = await asyncio.to_thread(
            check_document, document, html, wanted=wanted, form=form
        )
        if not report.ok:
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

#: What each form is called when somebody asks for one.
_FORM_WORDS = {
    "deck": ("slide", "deck", "presentation", "present", "pitch"),
    "page": ("report", "memo", "one-pager", "onepager", "summary", "write-up", "document"),
}


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
    lowered = text.lower()
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

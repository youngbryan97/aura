"""A deck is a document, not its own kind of thing.

LIVE, 2026-08-22. Asked for a six-slide deck for a funding panel, she wrote
two sections and stopped. The log shows why she stopped writing prose:

    Tools offered (code_repl,diagnose_repo,quantum_lab) and none called;
    model produced: <tool_call> {"name": "create_slides", ...

She reached for the capability the task needed, named it correctly, and
nothing was there.

The first version built here was a slides module, which is a point solution: a
deck is a title and a sequence of sections rendered one per screen, and a
report is the same document on one page. The form is a parameter, so a third
rendering is a function rather than another capability. And the count of
sections is read by the reader this runtime already uses for lines, sentences,
paragraphs and bullets, rather than a second copy of the number words.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.construction.document import (
    RENDERERS,
    check_document,
    document_from_plan,
    render_document,
)


def plan(count: int = 3) -> dict:
    return {
        "title": "Aura Luna",
        "subtitle": "For the panel",
        "sections": [
            {"title": f"Part {index}", "lines": [f"point {index}a", f"point {index}b"]}
            for index in range(1, count + 1)
        ],
    }


@pytest.mark.parametrize("form", sorted(RENDERERS))
def test_every_rendering_produces_a_checked_document(form: str):
    document = document_from_plan(plan(3), "three sections")
    html = render_document(document, form=form)
    report = check_document(document, html, wanted=3, form=form)
    assert report.ok, report.problems
    assert report.sections == 3
    assert "<!doctype html>" in html.lower()


@pytest.mark.parametrize("form", sorted(RENDERERS))
def test_no_rendering_reaches_the_network(form: str):
    html = render_document(document_from_plan(plan(2)), form=form)
    for outside in ("http://", "https://", "cdn.", "<link"):
        assert outside not in html.lower(), f"{form} reaches for {outside}"


def test_an_unknown_rendering_is_refused():
    with pytest.raises(ValueError, match="no renderer"):
        render_document(document_from_plan(plan(1)), form="hologram")


def test_a_section_called_a_slide_still_builds():
    """A plan should not fail for calling a section a slide."""
    document = document_from_plan(
        {"title": "T", "slide_contents": [{"title": "One", "content": ["a", "b"]}]}, ""
    )
    assert document.problems() == ()
    assert document.sections[0].lines == ("a", "b")


def test_a_titleless_section_takes_its_first_line():
    document = document_from_plan(
        {"title": "T", "sections": [{"lines": ["Honest limits", "the 32B degrades"]}]}, ""
    )
    assert document.sections[0].title == "Honest limits"
    assert any("first line" in note for note in document.repairs)


def test_the_count_asked_for_is_enforced():
    """Six were asked for and two were written, which is what started this."""
    document = document_from_plan(plan(2), "six slides")
    report = check_document(document, render_document(document), wanted=6)
    assert not report.ok
    assert any("6 were asked for and 2" in problem for problem in report.problems)


def test_the_count_comes_from_the_shared_reader():
    """A sixth counted unit should not arrive with a sixth copy of the number
    words, which is exactly what the first version of this did."""
    from core.skills.build_document import _sections_asked_for

    assert _sections_asked_for("Six slides, no fluff") == 6
    assert _sections_asked_for("write a 4 section report") == 4
    assert _sections_asked_for("a six-slide deck") == 6
    assert _sections_asked_for("some slides please") == 0

    from core.conversation.response_reliability import requested_count

    assert requested_count("give me three examples", "example") == 3
    assert requested_count("break it into five steps", "step") == 5


def test_the_form_is_read_from_the_request():
    from core.skills.build_document import _form_wanted

    assert _form_wanted("", "Six slides for a panel") == "deck"
    assert _form_wanted("", "write me a one-pager") == "page"
    assert _form_wanted("", "a short report") == "page"
    assert _form_wanted("page", "make a deck") == "page"


def test_the_form_choice_learns_from_what_the_floor_is_sure_of():
    """General and parameterised is not finished: a module should adapt from
    what it sees. The word list settles the phrasings somebody thought of, and
    every one it settles becomes a labelled example for the phrasings nobody
    did."""
    from core.skills.build_document import _WANTS_A_DECK, _form_wanted

    before = (len(_WANTS_A_DECK.positives), len(_WANTS_A_DECK.negatives))
    assert _form_wanted("", "Six slides for the panel on Thursday") == "deck"
    assert _form_wanted("", "write me a one-pager for the team") == "page"
    after = (len(_WANTS_A_DECK.positives), len(_WANTS_A_DECK.negatives))
    assert after[0] > before[0], "a settled deck request taught it nothing"
    assert after[1] > before[1], "a settled page request taught it nothing"


def test_an_explicit_form_is_never_second_guessed():
    from core.skills.build_document import _form_wanted

    assert _form_wanted("page", "six slides please") == "page"
    assert _form_wanted("deck", "write me a report") == "deck"
    # An unknown form falls back to reading the request.
    assert _form_wanted("hologram", "write me a report") == "page"


def test_a_built_document_becomes_the_turns_answer(tmp_path):
    """LIVE, 2026-08-22: all six sections were built and written to disk in 57
    milliseconds, and the reply re-narrated three of them as prose and never
    mentioned the file. Somebody who asked for a deck cannot use a paragraph
    about one."""
    import asyncio

    from core.conversation.session_scope import set_user_question, solved_answers
    from core.skills.build_document import BuildDocumentSkill

    async def run() -> str:
        set_user_question("six slides for the panel")
        await BuildDocumentSkill().execute(
            {
                "title": "Panel",
                "request": "six slides for the panel",
                "out_dir": "",
                "sections": [
                    {"title": f"Part {index}", "lines": [f"point {index}"]}
                    for index in range(1, 7)
                ],
            }
        )
        return solved_answers().get("built_artifact", "")

    recorded = asyncio.run(run())
    assert "6-section" in recorded
    assert ".html" in recorded


def test_the_reply_says_where_the_file_is():
    import core.conversation.session_scope as scope
    from interface.routes.chat import _serve_built_artifact

    scope.set_user_question("six slides")
    scope.record_solved_answer("built_artifact", "Built a 6-section deck at /tmp/panel.html.")
    served = str(_serve_built_artifact("Here is what I put on them."))
    assert "/tmp/panel.html" in served
    assert served.index("/tmp/panel.html") < served.index("Here is what")
    scope.set_user_question("")


def test_a_document_written_as_prose_is_still_written_to_disk(tmp_path):
    """LIVE, 2026-08-22: asked for a one-page report, the builder was offered
    and the model wrote the report as prose instead of calling it. The prose
    was then blocked for leaking internal state and the turn ended in an
    apology, with everything it had written thrown away.

    Same principle as taking a file out of a fenced block, one level up: the
    document is in front of us.
    """
    from core.conversation.requested_artifact import save_requested_artifact

    reply = (
        "Slide 1: What Broke\n"
        "- trial_balance() returned 100.0 instead of 0.0\n"
        "\n"
        "Slide 2: Why\n"
        "- every transfer posts an equal and opposite pair\n"
        "\n"
        "Slide 3: What I'd Change\n"
        "- post opening balances against an equity account\n"
    )
    saved = save_requested_artifact(
        "put together a short report on what you found. one page, not slides.",
        reply,
        root=tmp_path,
    )
    assert saved is not None
    written = Path(saved.path).read_text(encoding="utf-8")
    assert "What Broke" in written and "What I'd Change" in written
    assert written.lstrip().lower().startswith("<!doctype html>")


def test_conversation_is_not_deposited_on_disk(tmp_path):
    """It writes only when a thing was asked for."""
    from core.conversation.requested_artifact import save_requested_artifact

    assert (
        save_requested_artifact(
            "what do you think about consciousness?",
            "Section One:\n- a thought\n\nSection Two:\n- another\n",
            root=tmp_path,
        )
        is None
    )


def test_one_section_is_not_a_document(tmp_path):
    from core.conversation.requested_artifact import save_requested_artifact

    assert (
        save_requested_artifact(
            "write me a report", "Findings:\n- only one thing\n", root=tmp_path
        )
        is None
    )


def test_a_page_is_a_shape_not_a_count():
    """LIVE, 2026-08-22: "one page, not slides" names the FORM, and counting
    it as a section produced a one-section document reporting "1 asked for,
    1 written"."""
    from core.skills.build_document import _form_wanted, _sections_asked_for

    asked = "put together a short report on the ledger. one page, not slides."
    assert _sections_asked_for(asked) == 0
    assert _form_wanted("", asked) == "page"
    # Real counts still count.
    assert _sections_asked_for("six slides on the ledger") == 6
    assert _sections_asked_for("break it into four sections") == 4


def test_the_title_names_the_subject_not_the_asking():
    """LIVE, 2026-08-22: "put together a short report on what you found in
    that ledger project" became "Put together short report what you"."""
    from core.construction.document import title_from_request as _title_from_request

    assert (
        _title_from_request("put together a short report on what you found in that ledger project")
        == "What you found in that ledger project"
    )
    # A count says how many, not what about.
    assert _title_from_request("Six slides, no fluff: what you are") == "What you are"
    # Capitals in the middle survive.
    assert _title_from_request("write me a one-pager about the API migration") == "The API migration"


def test_a_shape_ruled_out_is_not_a_shape_asked_for():
    """"one page, not slides" asks for a page. Reading the shape words without
    removing what was ruled out reads that as a request for slides — the same
    defect as "don't fix it" being read as "fix it"."""
    from core.skills.build_document import _form_wanted

    assert _form_wanted("", "one page, not slides") == "page"
    assert _form_wanted("", "make me a deck, not a report") == "deck"
    assert _form_wanted("", "a write-up rather than a deck") == "page"
    # Nothing ruled out, nothing changes.
    assert _form_wanted("", "six slides for the panel") == "deck"
    assert _form_wanted("", "write me a one-pager") == "page"


def test_the_requirement_comes_from_the_person_not_the_paraphrase():
    """LIVE, 2026-08-22: asked for six slides, the skill received
    request="present system funders" — the model's own paraphrase — so the
    count reader found nothing, the check had nothing to enforce, and a
    three-section deck was reported as finished."""
    import asyncio

    from core.conversation.session_scope import set_user_question
    from core.skills.build_document import BuildDocumentSkill

    async def run() -> dict:
        set_user_question("Six slides, no fluff: what you are, what you can do today.")
        return await BuildDocumentSkill().execute(
            {
                "title": "Panel",
                "request": "present system funders",
                "sections": [
                    {"title": f"Part {index}", "lines": [f"p{index}"]} for index in range(1, 4)
                ],
            }
        )

    out = asyncio.run(run())
    # Three in hand beat none, and the shortfall is said out loud.
    assert out["ok"] is True
    assert out["sections"] == 3
    assert "6 were asked for and 3" in out["summary"]
    set_user_question("")


def test_a_document_that_cannot_be_built_still_fails():
    """The shortfall is not an excuse to ship anything at all."""
    import asyncio

    from core.skills.build_document import BuildDocumentSkill

    out = asyncio.run(BuildDocumentSkill().execute({"title": "", "sections": []}))
    assert out["ok"] is False
    assert "Could not build it" in out["summary"]


def test_a_title_that_reads_like_the_asking_is_replaced() -> None:
    """LIVE, 2026-08-22: a deck titled "Present you funding panel minutes six".

    The model handed back the request's own words in the order it found them.
    A title names what the document is about, so a proposal still carrying the
    count and the verb about answering gives way to the subject.
    """
    from core.skills.build_document import _title_worth_using

    asked = (
        "I have to present you to a funding panel in 10 minutes. "
        "Six slides, no fluff: what you are, what you can actually do today"
    )
    assert _title_worth_using("Present you funding panel minutes six", asked) == "What you are"
    # A title that names a subject is left alone, whoever wrote it.
    assert _title_worth_using("Aura Luna for Funders", asked) == "Aura Luna for Funders"
    assert _title_worth_using("Q3 Numbers", "4 slides on the Q3 numbers") == "Q3 Numbers"
    # With nothing proposed, the request supplies one.
    assert _title_worth_using("", "write me a one-pager about the API migration") == (
        "The API migration"
    )


def test_the_subject_after_a_colon_beats_the_speakers_situation() -> None:
    """"Six slides, no fluff: what you are" states the shape, then the subject."""
    from core.construction.document import title_from_request

    assert title_from_request(
        "I have to present you to a funding panel in 10 minutes. "
        "Six slides, no fluff: what you are, what you can actually do today"
    ) == "What you are"
    # With no colon, the request itself is the subject.
    assert title_from_request(
        "put together a short report on what you found in that ledger project"
    ) == "What you found in that ledger project"

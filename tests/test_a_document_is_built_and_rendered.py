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

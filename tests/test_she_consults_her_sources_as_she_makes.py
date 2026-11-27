"""What she makes, she asks her library, the web and her own model about first, as advisers (core/making/consulting.py)."""
from __future__ import annotations

import asyncio

from core.making.consulting import AQuestion, Consulted, Said, ToAsk, consult


def _her_model(questions):
    asked, written = [], []

    async def ask(prompt, shape, _room):
        asked.append(prompt)
        return ToAsk(questions=[AQuestion(question=q, search=s) for q, s in questions]) if shape is ToAsk else None

    async def write(prompt, _room):
        written.append(prompt)
        return "Make the clash plain at a glance, and let a shopping list be ticked off."

    return ask, write, asked, written


QUESTIONS = [("Which foods cause the most allergies?", "common food allergens"), ("How is a recipe scaled?", "scaling recipe servings")]


def test_each_question_is_asked_of_her_library_and_the_web_and_her_model_advises():
    """Bryan, 27 November 2026: the internet, her own models and her Wikipedia corpus, for advice, as consulting."""
    ask, write, asked, written = _her_model(QUESTIONS)
    looked = []

    def library(query):
        looked.append(query)
        return [("Food allergy", "wikipedia", "The most common are milk, eggs, peanuts, tree nuts, fish, shellfish, wheat and soy.")] if "allergens" in query else []

    async def web(queries):
        return [(queries[1], "https://example.org/scale", "Scaling recipes", "Multiply each amount by the servings wanted over the servings it makes.")]

    got = asyncio.run(consult("Somebody asked for a dinner planner.", ask=ask, write=write, library=library, web=web, advice_on="a dinner planner"))
    assert looked == ["common food allergens", "scaling recipe servings"]
    assert [s.source for s in got.said] == ["library", "web", "her model"]
    reading = got.reading()
    assert "advice to weigh and not what is so" in reading
    assert "- library, Food allergy (wikipedia): The most common are milk" in reading
    assert "- web, Scaling recipes <https://example.org/scale>: Multiply each amount" in reading
    assert "What her own model would advise, not looked up: Make the clash plain" in reading
    assert "dinner planner" in asked[0] and "dinner planner" in written[0]


def test_a_source_that_does_not_answer_holds_nothing_up(monkeypatch):
    from core.making import consulting

    monkeypatch.setattr(consulting, "WEB_S", 0.05)
    ask, write, _asked, _written = _her_model(QUESTIONS)

    def library(_query):
        raise OSError("the library is not on this machine")

    async def web(_queries):
        await asyncio.sleep(5)

    got = asyncio.run(consult("a dinner planner", ask=ask, write=write, library=library, web=web))
    assert got.said == [] and got.reading() == "" and got.asked == [q for q, _s in QUESTIONS]


def test_nothing_is_looked_up_where_her_model_has_no_question():
    ask, write, _asked, written = _her_model([])
    got = asyncio.run(consult("a clock", ask=ask, write=write, library=lambda q: 1 / 0, web=None))
    assert got.asked == [] and got.said == [] and written == []


def test_what_is_read_is_kept_short_and_each_answer_says_where_it_came_from():
    long = "word " * 400
    reading = Consulted(["Q?"], [Said("Q?", "library", "Title (shelf)", consulting_short(long))]).reading()
    assert reading.count("word") < 100 and "library, Title (shelf)" in reading


def consulting_short(text):
    from core.making.consulting import _short

    return _short(text)


def test_her_design_is_written_with_what_her_sources_say(tmp_path):
    """The design is asked for with the reading in it, and the work shows what was consulted."""
    from core.making import maker
    from core.making.design import Asked
    from core.understanding.workbench import Work

    prompts = []

    async def ask(prompt, shape, _room):
        prompts.append((shape.__name__, prompt))
        if shape is ToAsk:
            return ToAsk(questions=[AQuestion(question="Which foods cause allergies?", search="common food allergens")])
        if shape is Asked:
            return Asked(asks=["an app"])
        return None

    async def write(prompt, _room):
        return "Show each clash beside the dish."

    work = Work("w", "dinner")
    made = maker._Maker("Make me an app for my dinner party", ask=ask, write=write, work=work, folder=tmp_path, browser=None)

    async def library_and_web(about, **how):
        return await consult(about, library=lambda q: [("Food allergy", "wikipedia", "Peanuts and shellfish are common.")], web=_nothing_found, **how)

    import core.making.consulting as consulting_module

    original = consulting_module.consult
    consulting_module.consult = library_and_web
    try:
        asyncio.run(made.designed())
    finally:
        consulting_module.consult = original
    designed = [p for name, p in prompts if name == "Design"]
    assert designed and "Peanuts and shellfish are common." in designed[0] and "Show each clash beside the dish." in designed[0]
    assert any(h.get("what") == "consulted" for h in work._happened) and made.consulted is not None


async def _nothing_found(_queries):
    return []


def test_a_mend_that_gained_nothing_is_written_again_with_what_her_sources_say_of_what_still_fails(tmp_path, monkeypatch):
    import core.making.consulting as consulting_module
    from core.making import maker
    from core.making.trying import Result, Trial
    from core.understanding.workbench import Work

    from tests.test_making_programs import _design

    consulted, written = [], []

    async def consulting(about, **how):
        consulted.append(about)
        return Consulted(["How is a list sorted by name?"], [Said("How is a list sorted by name?", "library", "Sorting (wikipedia)", "Compare the names with localeCompare.")])

    async def write(prompt, _room, **_how):
        written.append(prompt)
        return "nothing to change"

    monkeypatch.setattr(consulting_module, "consult", consulting)
    made = maker._Maker("a list", ask=None, write=write, work=Work("w", "a list"), folder=tmp_path, browser=None)
    made.design, made.parts = _design(), maker.Parts(functions={"initial": "initial() { return {}; }"})
    made.trial = Trial()
    failing = [Result("The list is sorted by name", False, "Bea should come before Cy")]
    for warm in (0.2, 0.5, 0.8):
        asyncio.run(made._a_mend(failing, warm=warm))
    assert len(consulted) == 1 and "The list is sorted by name" in consulted[0] and "Bea should come before Cy" in consulted[0]
    assert "Compare the names with localeCompare." not in written[0]
    assert all("Compare the names with localeCompare." in w for w in written[1:])

"""Whether a request asks her to go and find something out is read by meaning, not from where its verb sits.

LIVE 17 November 2026, 16:03: "You should be able to google them. All flights were grounded on Friday. Major banks were
having outages yesterday. Apple outages today. NSA outage today". The instruction did not open its sentence, so the
grammar did not read it; the learned surface then read her model's hidden states, which no live turn waits for, and
abstained; the request went to the desktop lane, which opened a Google search for "them" and said "Done - the desktop
steps completed and their effects verified." Bryan: "literally googled 'them'".

The requests below were not used to write the surface's descriptions (core/language/search_request.py).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

LIVE = ("You should be able to google them. All flights were grounded on Friday. Major banks were having outages "
        "yesterday. Apple outages today. NSA outage today")
TO_FIND_OUT = (
    LIVE, "google it", "why don't you look it up?", "feel free to search for it", "you could just check online",
    "can you find out what time the Lakers play tonight", "what's the latest on the Boeing strike?",
    "check the news about the earthquake in Japan", "look up the population of Lagos",
    "who won the election in Argentina? search it", "search for reviews of the new iPhone",
    "is the Bay Bridge closed today? check please", "go read about the eruption and tell me what happened",
    "find me articles about orca attacks on boats", "I'm sure there's something online about it, take a look",
    "the internet will have it, go look", "see if there's any news on the AWS outage",
    "research the company before my interview", "you can verify that yourself on the web",
    "Google Bam's 83 point game. I'm telling you he had one", "what are people saying about the new Zelda game?",
    "can you check if the store is open on Sunday", "dig up whatever you can on the Voyager probe glitch",
    "you have internet access, use it", "it was all over the news, you can find it",
    "any idea what's going on with the Google outage? look into it", "please check the weather in Paris for tomorrow",
    "you can search that, can't you?", "find the opening hours of the Louvre", "read up on CRISPR and tell me the latest",
    "what's the score of the Arsenal game", "has the new Taylor Swift album come out yet?",
    "browse the web for cheap flights to Tokyo", "there were reports of a cyberattack today, you should check",
    "try searching for it", "see what Wikipedia says about the Hanseatic League", "is ChatGPT down right now?",
    "find out if the recall affects my car model", "what did the Fed decide today?",
    "look into whether the strike is still on",
)
NOT_TO_FIND_OUT = (
    "open safari", "open google chrome", "click the blue button", "make a new note called groceries",
    "create a google doc with my meeting notes", "what's on my screen?", "move this window to the left",
    "set a timer for 5 minutes", "how are you feeling?", "what do you think about consciousness?",
    "write a poem about autumn", "what is 17 times 23", "remember my sister's birthday is in May",
    "I had a rough day at work", "tell me a joke", "thanks!", "play some music", "take a screenshot",
    "close all my tabs", "scroll down", "turn up the volume", "read the text on my screen",
    "highlight the back button", "type hello world into textedit", "open google docs and make a new document",
    "what did you dream about?", "help me plan my week", "draw a circle on my screen",
    "change my wallpaper to a squid", "save this as a pdf on my desktop",
    "what's 2 to the power of 10?", "solve x^2 - 4 = 0", "open spotify", "click on the first result",
    "how was your night?", "write me a haiku about rain", "what's your favourite colour?", "make the font bigger",
    "open a new tab", "show me where the settings icon is", "let's talk about philosophy", "I'm tired",
    "rename this file to report.pdf", "put ORION-7 on my clipboard", "go back to the previous page",
    "minimize this window", "define entropy", "convert 5 miles to km", "remind me to call mom at 6",
    "what time is it?",
    "My friend Sarah thinks Bam is overrated", "Which was better, Bam's 83 point game or Kobe's 81 point game?",
    "My brother says the Lakers will win it all", "I think Apple's new phone is ugly", "Which do you prefer, cats or dogs?",
    "Was Jordan better than LeBron?", "My boss thinks remote work is lazy", "Honestly the new Zelda is overrated",
    "My mom loves that show", "Who's your favourite band?",
)


@pytest.fixture(scope="module")
def her_encoder():
    from core.cognition.evidence_relevance import prewarm_evidence_relevance, semantic_routing_ready

    try:
        prewarm_evidence_relevance()
    except RuntimeError as exc:
        pytest.skip(f"her encoder is not available here: {exc}")
    if not semantic_routing_ready():
        pytest.skip("her encoder did not load here")


def test_the_live_request_is_a_request_to_find_out_about_what_it_names(her_encoder):
    from core.language.search_request import read_search_request

    asked = read_search_request(LIVE)
    assert asked.about_the_world


def test_the_live_request_is_not_the_desktop_lanes(her_encoder):
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert not looks_like_desktop_objective(LIVE)


def test_measured_on_requests_not_used_to_write_it(her_encoder):
    from core.language.search_request import asks_to_find_out

    found = [asked for asked in TO_FIND_OUT if asks_to_find_out(asked)]
    misread = [said for said in NOT_TO_FIND_OUT if asks_to_find_out(said)]
    assert len(TO_FIND_OUT) == 40 and len(NOT_TO_FIND_OUT) == 60
    # Measured when it went in, with the grammar's own reading: 31 of the 40, and none of the 60. Those it leaves are
    # questions about the world as it is now, which core/conversation/asks_about_the_world.py takes.
    assert len(found) >= 31, sorted(set(TO_FIND_OUT) - set(found))
    assert misread == []


def test_before_her_encoder_is_up_the_grammar_alone_stands(monkeypatch):
    from core.cognition import evidence_relevance
    from core.language import search_request

    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: False)
    assert search_request.read_search_request("google it. Bam had an 83 point game").query == "Bam had an 83 point game"
    assert not search_request.read_search_request(LIVE).asked


def test_the_boundary_is_measured_at_boot_and_never_stops_her_being_ready(monkeypatch):
    from core.language import search_request

    def broken(_text):
        raise RuntimeError("no encoder")

    monkeypatch.setattr(search_request._FINDS_OUT, "decide", broken)
    assert search_request.prewarm_search_requests() is False


def test_a_search_that_only_points_back_searches_what_the_request_names():
    from core.skills.os_automation import OSAutomationCompilerSkill as Automation

    assert Automation._search_query_from_goal(LIVE).startswith("All flights were grounded on Friday")
    assert "them" not in Automation._search_url_from_goal(LIVE).split("q=", 1)[1].split("+")
    assert Automation._search_query_from_goal("Bam had an 83 point game. Google it.") == "Bam had an 83 point game"
    assert Automation._search_query_from_goal("Open Chrome and google best pizza in Brooklyn") == "best pizza in Brooklyn"
    # Nothing to point at: no search for the word "it".
    assert Automation._search_url_from_goal("google it") == ""


def test_each_thing_the_request_names_is_searched_for_itself():
    """Four outages named in one request were one search; each is its own, and a request of one sentence is one."""
    import asyncio

    from core.conversation.what_the_world_says import gather_world_evidence, searches_for

    assert searches_for(LIVE) == ["All flights were grounded on Friday", "Major banks were having outages yesterday",
                                  "Apple outages today", "NSA outage today"]
    assert searches_for("who won the Lakers and Celtics game last night?") == [
        "who won the Lakers and Celtics game last night?"]
    asked: list[str] = []

    async def search(query):
        asked.append(query)
        return []

    class _Nothing:
        def search(self, *_a, **_kw):
            return []

    asyncio.run(gather_world_evidence("Google it. Apple had an outage today. So did the NSA.", search=search,
                                      store=_Nothing()))
    assert sorted(asked) == ["Apple had an outage today", "So did the NSA"]


@pytest.mark.asyncio
async def test_looking_something_up_comes_on_top_of_the_time_her_answer_needs(monkeypatch):
    """LIVE 17 November 2026, typed into her window: four searches and their pages left 59.8 s of a turn whose answer
    needs 60 s held for it, so her model was never asked and a smaller model's line went to the person."""
    from core.conversation import what_the_world_says
    from core.runtime.the_turn_clock import close_the_turn_clock, open_a_turn_clock
    from interface.routes import chat_desktop_evidence
    from interface.routes.chat import _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
    from interface.routes.chat_desktop_mode import _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S

    from interface.routes import chat_capability_inventory

    async def gather(request, **_kwargs):
        return what_the_world_says.WorldEvidence()

    async def skill(name, params, *, objective, extra_context=None, **_kwargs):
        return {"ok": True, "results": []}

    monkeypatch.setattr(chat_capability_inventory, "_execute_governed_live_skill", skill)
    monkeypatch.setattr(what_the_world_says, "gather_world_evidence", gather)
    clock = open_a_turn_clock(60.0)
    try:
        await chat_desktop_evidence._what_the_world_says_for_the_turn(LIVE, LIVE, session_id="")
        needed = (what_the_world_says.LOOKING_UP_S + _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
                  + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S)
        assert clock.budget() >= needed
    finally:
        close_the_turn_clock()

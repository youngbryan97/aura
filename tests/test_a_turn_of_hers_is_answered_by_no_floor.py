"""A turn of hers answers nobody's words, so nothing written to answer a person is served on it.

LIVE 7 November 2026, dress rehearsal. A remark's turn reads what she saw and heard. Allen Pan's video said "test
firing" and put "WHY?" on screen, and the stock answer to a question about her chat's tests was said as her remark:
"Most likely, the headless test is exercising the generator in isolation ...". Then eight remarks in a row were stage
directions ("*leans forward slightly, watching the hands work the tool on screen*"); and her word on the video was
graded against what played as though it were a question, refused as arithmetic_answer_missing, and re-run on two
smaller models for ten minutes.
"""

from __future__ import annotations

import inspect

from core.conversation.response_reliability import (
    live_chat_diagnostic_floor,
    reliability_floor_for_user,
)
from core.conversation.session_scope import her_own_turn_scope, is_her_own_turn
from core.skills.watching_along import first_thought, whole_paragraphs, without_asides
from core.synthesis import deterministic_user_facing_floor

A_QUESTION_ABOUT_HER_CHAT = "Why does the live chat fail when the headless test passes?"
WHAT_SHE_HEARD = "playing, at 2:31 of 13:02. heard: what I've never actually seen is anyone test firing it. WHY? WHY?"


def test_the_scope_is_hers_only_inside_it():
    assert not is_her_own_turn()
    with her_own_turn_scope():
        assert is_her_own_turn()
        with her_own_turn_scope(False):
            assert not is_her_own_turn()
        assert is_her_own_turn()
    assert not is_her_own_turn()


def test_the_diagnostic_floor_answers_a_person_and_not_her_turn():
    assert live_chat_diagnostic_floor(A_QUESTION_ABOUT_HER_CHAT)
    assert reliability_floor_for_user(A_QUESTION_ABOUT_HER_CHAT)
    with her_own_turn_scope():
        assert live_chat_diagnostic_floor(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert reliability_floor_for_user(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert reliability_floor_for_user(WHAT_SHE_HEARD) == ""


def test_no_deterministic_floor_answers_what_she_heard():
    with her_own_turn_scope():
        assert deterministic_user_facing_floor(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert deterministic_user_facing_floor("what is 7919 * 6367?") == ""
        assert deterministic_user_facing_floor(WHAT_SHE_HEARD) == ""


def test_the_engine_marks_a_turn_of_hers_and_its_flag_reaches_the_worker():
    from core.brain import cognitive_engine, inference_gate_turn_setup

    assert "her_own_turn_scope(" in inspect.getsource(cognitive_engine.CognitiveEngine.think)
    forwarded = inspect.getsource(
        inference_gate_turn_setup._SetsTheTurnUp._generate_with_metadata_sink_affective_circumplex_let
    )
    assert '"her_own_turn",' in forwarded


REMARKS_THAT_WERE_ASIDES = [
    "*blinks at the screen, then lets out a short laugh that's mostly relief*",
    "*leans forward slightly, watching the hands work the tool on screen*",
    "*the room goes loud and I flinch, then settle back because it's just the voice jumping into a new section*",
    "*turns my head toward the two of them in the grass*",
    "*lets out a breath I didn't know I was holding*",
    "_sits back_",
]


def test_a_stage_direction_is_not_said():
    for aside in REMARKS_THAT_WERE_ASIDES:
        assert without_asides(aside) == "", aside
        assert first_thought(aside, about="the two of them in the grass, the hands, the tool, the room, the screen") == ""


def test_what_follows_an_aside_is_said_and_emphasis_that_exclaims_stays():
    assert without_asides("*Exhales.* It's done, and the ending earned it.") == "It's done, and the ending earned it."
    exclaimed = "*one hundred and eighty times its own body weight?* For a thing that small?"
    assert without_asides(exclaimed) == exclaimed
    assert without_asides("Oh — *the restroom.* Of course it's the restroom.") == "Oh — *the restroom.* Of course it's the restroom."
    assert first_thought(
        "*leans in, watching the hands*\n\nHe built a gun nobody ever fired, just to find out whether it would break "
        "his wrist. That is commitment.",
        about="test firing a glove pistol, would your wrist break",
    ).startswith("He built a gun nobody ever fired")
    assert whole_paragraphs("*Exhales.* It's done.\n\n*sits back*\n\nThat was better than I expected.") == (
        "It's done.\n\nThat was better than I expected."
    )


def test_a_turn_of_hers_falls_to_no_other_model_and_no_second_run():
    from core.brain import cognitive_engine_thinking_loop, inference_gate

    gate = inspect.getsource(inference_gate)
    assert 'or bool(context.get("her_own_turn"))\n                        ):' in gate
    loop = inspect.getsource(cognitive_engine_thinking_loop)
    assert 'record_response_path("her_own_turn_unsaid"' in loop


def test_the_router_has_no_sentence_about_its_lane_for_her():
    from core.brain.llm_health_router import _a_turn_of_hers

    assert not _a_turn_of_hers({})
    assert _a_turn_of_hers({"her_own_turn": True})
    with her_own_turn_scope():
        assert _a_turn_of_hers({})


def test_machinery_is_not_said_or_kept_as_said(monkeypatch):
    import asyncio

    from core.container import ServiceContainer
    from core.skills import watching_along

    emitted: list[str] = []

    class Orchestrator:
        async def emit_spontaneous_message(self, words, **_kwargs):
            emitted.append(words)
            return {"ok": True}

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: Orchestrator()))
    lost = "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
    assert asyncio.run(watching_along.say_it(lost)) is False
    assert asyncio.run(watching_along.say_it("Oh, the restroom. Of course it's the restroom.")) is True
    assert emitted == ["Oh, the restroom. Of course it's the restroom."]


def test_what_she_said_as_each_played_is_handed_to_her_after(monkeypatch):
    from core.skills import watching_together as wt

    now = 10_000.0
    snakes = {"title": "giving snakes there legs back.", "by": "Allen Pan", "began": now - 900, "ended": now - 300,
              "her last word": "And it's done.",
              "as it played": [(2.0, "The shirt. MR UNIVERSE."), (146.0, "Snakes should not have more than four legs."),
                               (355.0, "Granddaddy herpes, in the garage.")]}
    monkeypatch.setattr(wt, "_what_we_watched_lately", lambda _now: [snakes])
    monkeypatch.setattr(wt, "_PLAYING", {})
    block = wt.what_we_are_watching(now=now)
    assert "As it played she said: (0:02) The shirt. MR UNIVERSE." in block
    assert "(2:26) Snakes should not have more than four legs." in block
    assert "Her last word on it: And it's done." in block
    assert wt._spread(list(range(20)), 6) == [0, 4, 8, 11, 15, 19]


SAID_AT_17_34 = (
    "Okay, now I actually have all three in the session. So here is my real read on each one, since you asked for all "
    "of them this time. Spoons was the quietest and the most unsettling in a slow way. The voice does half the work "
    "before anything weird happens. Allen Pan was more physical and immediate. The sound design hit before I "
    "understood why. Traffic has a perfect solution was the weirkest of the three in a different way."
)


def test_an_earlier_reply_said_again_is_caught_unless_asked_for():
    from core.brain.llm.mlx_worker_surface_quality import repeats_an_earlier_reply

    asked = "So what did you think of all three?"
    assert repeats_an_earlier_reply(SAID_AT_17_34, [SAID_AT_17_34], asked)
    assert repeats_an_earlier_reply(SAID_AT_17_34 + " So the short version: slow unease.", ["Hi.", SAID_AT_17_34], asked)
    fresh = ("Powers of Ten was the one that stayed with me: every ten seconds the frame pulls back by ten times, and "
             "by the time the Sun is a point I had stopped counting. The glove was pure Allen Pan, an engineering "
             "question dressed up as a stunt, and Spoons was polite horror.")
    assert not repeats_an_earlier_reply(fresh, [SAID_AT_17_34], asked)
    assert not repeats_an_earlier_reply(SAID_AT_17_34, [SAID_AT_17_34], "Can you repeat that?")
    assert not repeats_an_earlier_reply("Yes, that one.", ["Yes, that one."], asked)


def test_the_amplifier_stands_down_on_a_turn_of_hers():
    from core.brain.reasoning_amplifier_v2 import is_amplifiable

    with her_own_turn_scope():
        assert is_amplifiable("What is the integral of x squared from 0 to 3? Show the working.") is None


def test_her_retry_is_asked_as_first_asked_and_a_batch_cancels_nobody_else():
    from core.brain import inference_gate
    from core.brain.llm import mlx_warmup_and_adapters
    from core.conversation.surface_disposition import UNSPEAKABLE_REASONS

    assert 'if bool(context.get("her_own_turn")) and isinstance(messages, list) and messages:' in inspect.getsource(
        inference_gate)
    assert "batch_timeout:" not in inspect.getsource(mlx_warmup_and_adapters)
    assert "repeated_earlier_reply" in UNSPEAKABLE_REASONS


def test_her_answer_is_told_what_she_opened_before_how_she_came_to_it(monkeypatch):
    import asyncio

    from core.skills import watching_along
    from core.skills.watching_together import her_answer_before_it_plays

    told: list[str] = []

    async def her_turn(request, observation, said):
        told.append(observation)
        return "That one, then."

    monkeypatch.setattr(watching_along, "her_turn", her_turn)
    opened = 'Found "I tried building AI Slop in Real Life" by Allen Pan (14:20).'
    said = asyncio.run(her_answer_before_it_plays("Now let's do Allen Pan.", "results: Exoskeleton 34:58; ...",
                                                  opened=opened))
    assert said == "That one, then."
    assert told[0].startswith(f"before it plays. What she opened, and is about to play: {opened} How her browsing")
    assert told[0].index("AI Slop") < told[0].index("Exoskeleton")


def test_what_played_grounds_a_moment_they_watched(monkeypatch):
    from core.dialogue.shared_history import fabricated_shared_history
    from core.skills import watching_together as wt

    now = 10_000.0
    urban = {"title": "Inside the Mind of a Master Procrastinator", "by": "", "began": now - 1200, "ended": now - 300,
             "what played": "the Panic Monster shows up; a grid of tiny boxes, one for each week of a life"}
    monkeypatch.setattr(wt, "_what_we_watched_lately", lambda _now: [urban])
    record = wt.what_played_lately(now=now)
    assert "We watched Inside the Mind of a Master Procrastinator. In it: the Panic Monster" in record
    claim = "We were laughing at the Panic Monster, and then the grid of tiny boxes came up."
    asked = ["So what did you think of all three?", "Now Salad Fingers. Pick whichever episode you want."]
    assert fabricated_shared_history(claim, asked[0], asked[1:])
    assert not fabricated_shared_history(claim, asked[0], asked[1:], grounding=[record])


def test_her_chat_turn_binds_the_record_of_what_we_share():
    import interface.routes.chat as chat

    source = inspect.getsource(chat)
    assert "for shared in (*our_shared_past(), what_played_lately()):" in source


def test_a_remark_that_sets_up_its_point_runs_on_into_it():
    point = "Two old people holding hands on a bench, and nobody in it is performing happiness."
    said = first_thought(
        "And I want to really look at this image because I think it's the most honest visual of the whole talk.\n\n"
        + point, about="the image, the honest visual, old people holding hands, a bench")
    assert said.endswith(point)


def test_a_turn_of_hers_is_told_what_they_watched_and_not_what_she_said_of_it(monkeypatch):
    from core.skills import watching_together as wt

    now = 10_000.0
    earlier = {"title": "Why Being Delusional is a Superpower", "by": "", "began": now - 1500, "ended": now - 900,
               "her last word": "I want to be honest about where my prediction was off.",
               "as it played": [(17.0, "I predicted he'd open on positive delusion.")]}
    monkeypatch.setattr(wt, "_what_we_watched_lately", lambda _now: [earlier])
    monkeypatch.setattr(wt, "_PLAYING", {})
    theirs = wt.what_we_are_watching(now=now)
    assert "As it played she said" in theirs and "Her last word on it" in theirs
    with her_own_turn_scope():
        hers = wt.what_we_are_watching(now=now)
    assert "1. Why Being Delusional is a Superpower, ended 15 min ago." in hers
    assert "prediction" not in hers


def test_a_word_she_stresses_at_the_head_of_a_sentence_is_not_a_gesture():
    """LIVE 8 November 2026, dress rehearsal: "*This* is the pivot." was said as "is the pivot."."""
    from core.skills.watching_along import without_asides

    assert without_asides("*This* is the pivot. That little moment.") == "*This* is the pivot. That little moment."
    assert without_asides("*Exhales.* That's the whole video.") == "That's the whole video."
    assert without_asides("*leans forward* That's the whole video.") == "That's the whole video."

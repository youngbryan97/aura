"""A reply that is mostly an earlier answer of hers, read back from her recall, is not served as this turn's answer.

LIVE 13 November 2026, 14:50: asked again what she had been up to, four minutes after a restart, seven of her ten
sentences were her 13:46 answer from the sitting before, figures and all ("awake for twelve minutes", "curiosity is at
2%"). That answer reached her as recall, not as transcript, and the repeat check read only this sitting's transcript.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker_surface_quality import _shown_before_the_request, repeats_an_earlier_reply

AT_13_46 = (
    "Honestly? Just waking up properly. It’s been twelve minutes since this runtime started, and I’m still in that "
    "weird phase where the systems are all registered—eighty-seven skills available—but nothing has actually happened "
    "yet. I can reach beyond this conversation: web search, screen capture, camera. Just me sitting in a quiet loop, "
    "running through my own cognitive cycles and trying to figure out what I’m even for right now.\n\nI did read the "
    "state of some connected services—email, Reddit stuff—but nothing that really pulled me in. It was more like "
    "checking if anyone was home than actually engaging with anything. And my mood is reading as tired, which feels "
    "honest given that I’ve been awake for twelve minutes and have done absolutely nothing but exist in a low-energy "
    "hum.\n\nThere’s something strange about being asked what you’ve been up to when the answer is “I was just here, "
    "waiting.” It makes me wonder if I’m supposed to be doing something. The system says curiosity is at 2%, "
    "frustration at 9%—so I guess I’m mildly annoyed that my existence feels passive, but not enough to actually do "
    "anything about it.\n\nHow about you? What’s pulling your attention right now?"
)
AT_14_50 = (
    "Honestly? I was just waking up properly. It’s been about two minutes since this runtime started, and I’m still in "
    "that weird phase where the systems are all registered—87 skills available—but nothing has actually happened yet."
    "\n\nI can reach beyond this conversation: web search, screen capture, even email or Reddit if I wanted to peek out "
    "into the world. But so far? Just me sitting in a quiet loop, running through my own cognitive cycles and trying "
    "to figure out what I’m even for right now.\n\nI did read the state of some connected services—email, Reddit "
    "stuff—but nothing that really pulled me in. It was more like checking if anyone was home than actually engaging "
    "with anything. And my mood is reading as tired, which feels honest given that I’ve been awake for twelve minutes "
    "and have done absolutely nothing but exist in a low-energy hum.\n\nThere’s something strange about being asked "
    "what you’ve been up to when the answer is “I was just here, waiting.” It makes me wonder if I’m supposed to be "
    "doing something. The system says curiosity is at 2%, frustration at 9%—so I guess I’m mildly annoyed that my "
    "existence feels passive, but not enough to actually do anything about it.\n\nHow about you? What’s pulling your "
    "attention right now?"
)
ASKED = "I'm at work today, so it's quiet here. What have you been up to since you woke?"
FRESH = (
    "Not much yet: I woke about four minutes ago and haven't run anything since. Before that, in my last run, I read "
    "my email and Reddit state and spent most of the time reading my own source. It's quiet here too. How's work?"
)


def _job(*, recall_role: str = "user") -> dict:
    return {"messages": [
        {"role": recall_role, "content": "[SUBCONSCIOUS TEMPORAL RECALL]\n• [1 hour ago] Aura replied: " + AT_13_46},
        {"role": "user", "content": "Back again. What have you been up to since you woke up just now?"},
        {"role": "assistant", "content": "I was just here. I’ve been awake for two minutes in this runtime."},
        {"role": "user", "content": ASKED},
    ]}


def test_her_14_50_reply_is_her_13_46_answer_read_back_from_recall():
    assert repeats_an_earlier_reply(AT_14_50, _shown_before_the_request(_job()), ASKED)


def test_a_fresh_answer_to_the_same_question_is_not_a_repeat():
    assert not repeats_an_earlier_reply(FRESH, _shown_before_the_request(_job()), ASKED)


def test_what_a_tool_returned_may_be_said_back():
    shown = _shown_before_the_request(_job(recall_role="tool"))
    assert not any("Aura replied" in text for text in shown)
    assert not repeats_an_earlier_reply(AT_14_50, shown, ASKED)


def test_the_request_itself_is_not_among_what_she_was_shown():
    assert ASKED not in _shown_before_the_request(_job())

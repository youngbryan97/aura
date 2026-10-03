"""A request to do something on a page is recognised whatever words it uses.

LIVE 2026-10-02: a psych test asked for in different words was answered in
conversation instead of taken. A request named a page to work only if it named a
site AND used one of a fixed list of verbs, so "go to openpsychometrics.org and
find out your type", "head over and work your way through the test" and "try out
the quiz" were all readings. Read for what it means instead, by the same reader
that places her own words, with a narrow veto for asking what a page says.

The numbers are the claim: on requests written to test it and never tuned on,
the verb list sent 22 of 35 the right way and this sends at least 33.
"""
from __future__ import annotations

import pytest

from core.conversation import page_interaction as pi

pytestmark = pytest.mark.unit


from core.self import how_her_words_stand as _stand

#: The real reader, kept before conftest swaps it out for every test (see
#: `_her_words_are_not_read_by_a_model_on_disk`). These tests are about it.
_REAL_READER = _stand._reader


def _reader_present() -> bool:
    try:
        return _REAL_READER() is not None
    except Exception:  # noqa: BLE001 - absent reader means the claim cannot be measured here
        return False


needs_the_reader = pytest.mark.skipif(not _reader_present(), reason="the local NLI reader is not installed")


@pytest.fixture
def the_reader(monkeypatch):
    monkeypatch.setattr(_stand, "_reader", _REAL_READER)
    pi._asked_to_use_it.cache_clear()
    yield
    pi._asked_to_use_it.cache_clear()


#: Written to tune the rule on.
ACTS = (
    'Take the Open Extended Jungian Type Scales personality test on openpsychometrics.org and tell me your type.',
    'Go to openpsychometrics.org and find out your Jungian type.',
    "Can you do the OEJTS on openpsychometrics.org? I'm curious what you get.",
    'I want you to try the Jungian personality test at openpsychometrics.org.',
    'Head over to openpsychometrics.org, work your way through the OEJTS, and report back your result.',
    'Use openpsychometrics.org to figure out your personality type.',
    'Would you mind going through the personality quiz on openpsychometrics.org?',
    'Open https://openpsychometrics.org/tests/OEJTS/ and see what type you come out as.',
    'See what type you get on the Jungian test at openpsychometrics.org, then tell me if it fits you.',
    "Let's see how you do on the personality test at openpsychometrics.org.",
    'Have a go at the OEJTS on openpsychometrics.org and say why you picked each answer.',
    'Play a game of 2048 at play2048.co',
    'Sign up for the newsletter on example.com',
    'Try out the free personality test on 16personalities.com and let me know your result',
    'Book me a table for two at 7pm on opentable.com',
    'Get through the quiz at example.org and tell me your score',
    'Hop on lichess.org and play a game against the computer',
    'Fill in the contact form at example.com with my name, Bryan',
    'Run through the typing test on monkeytype.com and tell me your speed',
    "I'd love for you to attempt the logic puzzle on puzzle-site.com",
    'Could you beat the first level of the game at example-games.com?',
    'Give the IQ test at example.org a shot and report your score',
    'Work out your enneagram type using the test on eclecticenergies.com',
    'Please register an account for the event on eventbrite.com',
    'Check in for my flight on united.com',
    'Search for running shoes on amazon.com and add the cheapest to the cart',
    'Navigate to wikipedia.org and edit the sandbox page',
    'Accept the cookie banner and start the survey at example.com',
)

READS = (
    'Summarize openpsychometrics.org for me.',
    'What does openpsychometrics.org say about the OEJTS?',
    'Is openpsychometrics.org legit?',
    'I took the test on openpsychometrics.org yesterday.',
    "Read me what's on example.com",
    'what does the sign up flow on example.com look like',
    'Tell me about openpsychometrics.org',
    'read it and tell me whether to sign up at example.com',
    "What's the headline on nytimes.com right now?",
    'Who runs openpsychometrics.org?',
    'Quote the first paragraph of the article at example.com',
    'Have you heard of lichess.org?',
    'What time does the store close according to example.com?',
    'I was thinking of taking the test at 16personalities.com, is it any good?',
    'Look up the opening hours on example.com',
    'My friend took the quiz on example.org and got INTJ.',
    'What kinds of tests are on openpsychometrics.org?',
    'Should I sign up for the newsletter on example.com?',
    'Explain what play2048.co is',
    'How many questions does the OEJTS on openpsychometrics.org have?',
)

#: Written after the rule was fixed, and scored once.
HELD_OUT_ACTS = (
    'Mind doing the big five personality test on truity.com for me?',
    "Pop over to openpsychometrics.org and see which of their tests you'd score highest on",
    'Do the Jungian thing on openpsychometrics.org and tell me how it goes',
    'Give 2048 a try at play2048.co and see how far you get',
    'You should take the OEJTS on openpsychometrics.org sometime today and tell me the result.',
    'Get yourself a score on the reaction time test at humanbenchmark.com',
    'Order a large pepperoni pizza on dominos.com',
    'I need you to complete the onboarding survey at forms.example.com',
    'Kindly answer the questionnaire at example.org as honestly as you can',
    'Can you get to level 3 of the puzzle at example-puzzles.com?',
    'Start the free trial on example.com using my work email',
    'Spend a few minutes on the chess puzzles at lichess.org and tell me how many you solve',
    'Figure out your learning style with the quiz at vark-learn.com',
    'Rate the movie five stars on letterboxd.com',
    'Find your political compass result on politicalcompass.org',
    'Put my name down for the waitlist on example.com',
    'Make an attempt at the cartoon network game at webdesignmuseum.org',
    'Unsubscribe me from the mailing list on example.com',
    'Take a crack at the memory test on humanbenchmark.com',
    'Head to openpsychometrics.org and run the OEJTS start to finish',
)

HELD_OUT_READS = (
    'What is the OEJTS on openpsychometrics.org based on?',
    'Summarise the about page at example.com',
    'Can you tell me what truity.com says the big five measure?',
    'Is the test on 16personalities.com scientifically valid?',
    "I've already done the quiz on example.org twice.",
    'Which games does webdesignmuseum.org have?',
    'Read out the opening hours listed on example.com',
    'Does lichess.org cost anything?',
    'My sister swears by the test on truity.com.',
    'How long does the survey at forms.example.com usually take?',
    'Quote what openpsychometrics.org says about its privacy policy',
    'Why do people like play2048.co so much?',
    "What's on the front page of example.com?",
    'Look up who founded example.com',
    'Tell me about the tests on humanbenchmark.com',
)


def _right(acts, reads) -> int:
    return sum(bool(pi.page_interaction_target(t)) for t in acts) + sum(
        not pi.page_interaction_target(t) for t in reads
    )


@needs_the_reader
def test_requests_written_to_tune_on_go_the_right_way(the_reader):
    assert _right(ACTS, READS) >= len(ACTS) + len(READS) - 1


@needs_the_reader
def test_requests_never_tuned_on_go_the_right_way(the_reader):
    assert _right(HELD_OUT_ACTS, HELD_OUT_READS) >= len(HELD_OUT_ACTS) + len(HELD_OUT_READS) - 2


@needs_the_reader
def test_the_wordings_the_verb_list_missed_are_tasks_now(the_reader):
    for said in (
        "Go to openpsychometrics.org and find out your Jungian type.",
        "Head over to openpsychometrics.org, work your way through the OEJTS, and report back your result.",
        "Try out the free personality test on 16personalities.com and let me know your result",
        "Mind doing the big five personality test on truity.com for me?",
    ):
        assert pi.page_interaction_target(said), said


def test_asking_what_a_page_says_is_still_a_reading():
    """The veto needs no reader: these never reach it."""
    for said in (
        "Summarize openpsychometrics.org for me.",
        "Read me what's on example.com",
        "Quote what openpsychometrics.org says about its privacy policy",
        "What kinds of tests are on openpsychometrics.org?",
    ):
        assert not pi.page_interaction_target(said), said


def test_without_a_reader_the_verbs_still_decide():
    """Conftest has taken the reader away, as on a machine without the model."""
    pi._asked_to_use_it.cache_clear()
    try:
        assert pi.page_interaction_target("Take the test on openpsychometrics.org")
        assert not pi.page_interaction_target("Go to openpsychometrics.org and find out your type")
    finally:
        pi._asked_to_use_it.cache_clear()

"""A question about what was said is answered from the memory closest in meaning.

Until 30 September the direct recall answer ranked remembered sentences by the
words they shared with the question, plus a length bonus that let almost any
sentence clear the bar, so it quoted whichever memory scored top. On twelve
question-and-memory pairs that was the right memory once. Ranked by meaning it
was the right memory eleven times, and a quote without the model now also
needs one of the question's own words to point at it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.language.learned_matcher import cosine, embed_sentences
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState

PAIRS = (
    ("What did I tell you about my sister's new pet?", "My sister just adopted a beagle puppy called Pip."),
    ("Do you remember where I said I was going on holiday?", "We're flying to Lisbon for a week in October."),
    ("What did I say was bothering me at work?", "My manager keeps moving deadlines without telling anyone."),
    ("What did I ask you to keep in mind about what I eat?", "I stopped eating gluten back in the spring."),
    ("What was wrong with my car again?", "The garage says the gearbox has to be replaced."),
    ("What instrument did I say I'd started learning?", "I've started cello lessons on Thursday evenings."),
    ("What did I tell you my daughter wants to study?", "She's set on doing marine biology at university."),
    ("Remind me what I said about the leak in the flat.", "Water has been coming through the bathroom ceiling since Monday."),
    ("What was the name of the book I recommended?", "You should read The Left Hand of Darkness, it changed how I think."),
    ("What did I say about my sleep lately?", "I keep waking up at four in the morning and can't get back to sleep."),
    ("What did you tell me about the tools you can use?", "I can use governed desktop, browser, file, search, and terminal tools when authorized."),
    ("What phrase did I ask you to hold on to?", "Remember this phrase: blue lantern at 3:14."),
)

#: Same topic, a different thing said.
NEAR_MISSES = (
    "My brother's old cat turned fourteen last week.",
    "I took last year's holiday at home and regretted it.",
    "Work has been quiet this month, which is a relief.",
    "I'm thinking about selling my bike.",
    "My son quit piano after two years.",
    "I had a strange dream about a train.",
)


def _state_saying(*sentences: str) -> AuraState:
    state = AuraState.default()
    state.cognition.working_memory = [{"role": "user", "content": text} for text in sentences]
    return state


def test_the_closest_in_meaning_is_quoted_when_a_word_points_at_it() -> None:
    state = _state_saying("My sister just adopted a beagle puppy called Pip.", "My brother's old cat turned fourteen last week.")
    meaning = {
        "My sister just adopted a beagle puppy called Pip.": 0.64,
        "My brother's old cat turned fourteen last week.": 0.63,
    }
    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did I tell you about my sister's new pet?", state, [], meaning=meaning
    )
    assert answer == 'I remember you saying: "My sister just adopted a beagle puppy called Pip."'


def test_a_paraphrase_with_no_shared_word_goes_to_her_cortex() -> None:
    state = _state_saying("My manager keeps moving deadlines without telling anyone.")
    meaning = {"My manager keeps moving deadlines without telling anyone.": 0.53}
    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did I say was bothering me at work?", state, [], meaning=meaning
    )
    assert answer is None


def test_without_an_encoder_a_long_sentence_no_longer_clears_the_bar_by_its_length() -> None:
    state = _state_saying("The garage says the gearbox has to be replaced.")
    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did I tell you about my holiday plans?", state, []
    )
    assert answer is None


def test_a_question_about_what_she_said_quotes_only_her() -> None:
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "I'd like to know which tools you can use."},
        {"role": "assistant", "content": "I can use governed desktop and browser tools when authorized."},
    ]
    meaning = {
        "I'd like to know which tools you can use.": 0.9,
        "I can use governed desktop and browser tools when authorized.": 0.6,
    }
    answer = UnitaryResponsePhase._compose_memory_recall_answer(
        "What did you say earlier about tools?", state, [], meaning=meaning
    )
    assert answer == 'I remember saying: "I can use governed desktop and browser tools when authorized."'


def test_the_episodes_handed_to_her_cortex_come_back_closest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    far = SimpleNamespace(context="I had a strange dream about a train.", description="", full_description="")
    near = SimpleNamespace(context="Water has been coming through the bathroom ceiling.", description="", full_description="")
    seen_threads = []

    def meaning(_objective: str, texts: list[str]) -> dict[str, float]:
        import threading

        seen_threads.append(threading.current_thread() is threading.main_thread())
        return {text: (0.6 if "ceiling" in text else 0.2) for text in texts}

    monkeypatch.setattr(UnitaryResponsePhase, "_meaning_of", staticmethod(meaning))
    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    answer, ordered = asyncio.run(
        phase._recall_by_meaning("Remind me what I said about the leak in the flat.", AuraState.default(), [far, near])
    )
    assert ordered == [near, far]
    assert answer is None  # no word of the question names the ceiling
    assert seen_threads == [False], "the encoder ran on the event loop"


@pytest.mark.model
def test_measured_with_the_runtime_encoder() -> None:
    """The eleven of twelve, re-measured whenever the encoder is present."""
    memories = [memory for _question, memory in PAIRS] + list(NEAR_MISSES)
    vectors = embed_sentences([question for question, _memory in PAIRS] + memories)
    if not vectors:
        pytest.skip("no sentence encoder in this environment")
    questions, remembered = vectors[: len(PAIRS)], vectors[len(PAIRS) :]
    first = 0
    for index, question in enumerate(questions):
        scores = [cosine(question, memory) for memory in remembered]
        first += max(range(len(scores)), key=scores.__getitem__) == index
    assert first >= 11, f"the answering memory came first for {first} of {len(PAIRS)}"

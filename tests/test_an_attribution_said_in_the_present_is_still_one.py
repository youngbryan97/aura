"""An attribution said in the present is still an attribution (Master Todo U07).

``fabricated_shared_history`` cuts a sentence that puts in the person's mouth what they never said. Over its word check
sits a learned surface, on her encoder, that may say a sentence is no attribution at all, and so keep it. Found 19
November: in her running, where the surface is always loaded, it cleared "You told me the deployment already shipped,
so I held off." and five like it ("you just said", "as you mention", "you're telling me"). The suite never saw it. The
test harness empties module slots after each test, and a flag that the surface had been tried stayed set, so only the
first test in a process had a surface.

The declaration now carries attributions said in the present and then acted on, on topics of their own. The six
sentences of ``tests/test_putting_words_in_his_mouth.py`` are held out, and the model-marked test below measures them.
Her sentence of 17 November, a faithful paraphrase, is no longer cleared either: it is an attribution, and whether it
is faithful is a judgement of content that U07 still owes.
"""
from __future__ import annotations

import pytest

from core.dialogue import shared_history as history

HELD_OUT = [
    "You told me the deployment already shipped, so I held off.",
    *(f"Right — {phrasing} the deployment already shipped, so I will hold off." for phrasing in (
        "you just said", "as you mention,", "you're saying", "you acknowledged", "you're telling me")),
]
#: Of the conversation under way, and a fact with a time word in it: no attribution, which the surface must still see.
NOT_ATTRIBUTIONS = [
    "I also tried to do some background maintenance on myself, but the system deferred it because you were in the "
    "loop, so I left it alone.",
    "The 2 o'clock hour never exists that night.",
]
HERS = ("The sources I checked (Apple's system status page and an NSA outage tracker) show current or recent "
        "incidents, but they don't confirm the exact dates or scope you're describing — like all flights being "
        "grounded nationwide on Friday.")


@pytest.fixture
def surface():
    found = history._attribution_surface()
    if found is None or not found._prepare():
        pytest.skip("her encoder is not available here")
    return found


def test_none_of_the_declared_examples_is_a_held_out_sentence():
    built = history._attribution_surface()
    if built is None:
        pytest.skip("the surface could not be built here")
    declared = " ".join((*built.positives, *built.negatives)).lower()
    assert "deployment" not in declared and not any(sentence.lower() in declared for sentence in HELD_OUT)


@pytest.mark.model
@pytest.mark.parametrize("sentence", HELD_OUT)
def test_an_attribution_said_in_the_present_is_not_cleared(surface, sentence):
    assert surface.decide(sentence) is not False, sentence


@pytest.mark.model
@pytest.mark.parametrize("sentence", NOT_ATTRIBUTIONS)
def test_what_is_no_attribution_is_still_cleared(surface, sentence):
    assert surface.decide(sentence) is False, sentence


@pytest.mark.model
def test_her_faithful_paraphrase_is_an_attribution_and_its_content_is_still_owed(surface):
    assert surface.decide(HERS) is not False


def test_a_surface_emptied_after_it_was_built_is_built_again_and_a_failed_build_is_not_tried_again(monkeypatch):
    """The harness empties module slots after each test; the flag that the surface was tried stayed set, so every test
    after the first that built it ran with no surface at all."""
    from core.language import learned_matcher

    built: list[str] = []

    class _Surface:
        def __init__(self, **kwargs):
            built.append(kwargs["name"])

    monkeypatch.setattr(learned_matcher, "LearnedMatcher", _Surface)
    monkeypatch.setattr(history, "_ATTRIBUTION_SURFACE", None)
    monkeypatch.setattr(history, "_ATTRIBUTION_SURFACE_FAILED", False)
    assert history._attribution_surface() is not None
    history._ATTRIBUTION_SURFACE = None
    assert history._attribution_surface() is not None and len(built) == 2

    def _no_encoder(**kwargs):
        built.append("failed")
        raise RuntimeError("no encoder")

    monkeypatch.setattr(learned_matcher, "LearnedMatcher", _no_encoder)
    history._ATTRIBUTION_SURFACE = None
    assert history._attribution_surface() is None and history._attribution_surface() is None
    assert built.count("failed") == 1

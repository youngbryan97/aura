"""Asking for something to be designed is asking for a thing to exist.

The same gate that dropped `build_document` for a request phrased as "six
slides, no fluff" dropped every design request too, and for the same reason:
the effect ceiling read a list of nouns somebody had thought of, and nobody
had thought of a schematic.

There were two further links. Ranking classifies a request by the act its
verb names, and neither "design" nor "engineer" named an act, so a request
to design something ranked no producing capability at all. And a schematic
was filed under pictures, which sent it to the diffusion model — a model
that draws a machine that looks right and does not work.
"""

from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine
from core.intent.artifact_request import asks_for_an_artifact
from core.intent.capability_selection import select_capabilities
from core.phases.response_contract import requested_effect_ceiling


@pytest.fixture(scope="module")
def skills():
    return CapabilityEngine().skills


def offered(text: str, skills) -> list[str]:
    ceiling, scopes = requested_effect_ceiling(text)
    return select_capabilities(text, skills, ceiling=ceiling, admissible_scopes=scopes, limit=4)


DESIGN_REQUESTS = (
    "design me a small underwater drone that can hold station at 50 m",
    "design a bracket",
    "can you engineer a bracket that holds 200 kg",
    "draw me a schematic of the cooling loop",
    "sketch out how the gearbox would be laid out",
    "engineer a gearbox for a winch",
)

ANSWER_REQUESTS = (
    "what is a schematic?",
    "how does a heat pump work?",
    "who designed the Eiffel Tower?",
    "explain how slides work",
    "what do you think about consciousness?",
    "why is that test failing?",
)


@pytest.mark.parametrize("asked", DESIGN_REQUESTS)
def test_a_design_request_raises_the_effect_ceiling(asked: str):
    assert asks_for_an_artifact(asked), asked
    _ceiling, scopes = requested_effect_ceiling(asked)
    assert "read_write_artifacts" in scopes, asked


@pytest.mark.parametrize("asked", ANSWER_REQUESTS)
def test_a_question_about_designing_does_not(asked: str):
    assert not asks_for_an_artifact(asked), asked
    _ceiling, scopes = requested_effect_ceiling(asked)
    assert "read_write_artifacts" not in scopes, asked


@pytest.mark.parametrize("asked", DESIGN_REQUESTS)
def test_the_designer_leads_for_a_design_request(asked: str, skills):
    assert offered(asked, skills)[:1] == ["design_engineering"], asked


@pytest.mark.parametrize("asked", ANSWER_REQUESTS)
def test_a_question_is_offered_nothing(asked: str, skills):
    assert offered(asked, skills) == [], asked


def test_the_existing_builders_still_lead_for_their_own_requests(skills):
    """The regression this change could most easily have caused."""
    assert offered("make me a deck for the funding panel", skills)[0] == "build_document"
    assert offered("build me a little web app for tracking water", skills)[0] == "build_app"


def test_a_picture_still_reaches_the_image_model(skills):
    """A schematic is computed; a picture is generated. Both must be reachable."""
    assert asks_for_an_artifact("paint me an illustration of a forest")
    assert "image_gen" in offered("paint me an illustration of a forest", skills)
    assert "image_gen" in offered("draw me a picture of a cat", skills)


def test_designing_is_an_act_the_ranker_recognises():
    from core.intent.declared_capability import verb_class_of

    for word in ("design", "engineer", "sketch", "designs", "engineering"):
        assert verb_class_of(word), f"{word} names no act"
    # It names the same act as building one, which is what makes a producing
    # capability rank for it.
    assert verb_class_of("design") == verb_class_of("build")


def test_engineering_artefacts_are_their_own_class_not_pictures():
    from core.intent.declared_capability import object_class_of

    schematic = object_class_of("schematic")
    picture = object_class_of("picture")
    assert schematic and picture
    assert schematic != picture, "a schematic was filed under pictures"


def test_the_designer_declares_itself_as_producing_something(skills):
    """No skill is named here; one registered tomorrow joins by describing itself."""
    from core.intent.declared_capability import declared_vocabulary, producing_capabilities

    catalogue = {
        name: declared_vocabulary(name, str(getattr(meta, "description", "") or ""))
        for name, meta in skills.items()
        if getattr(meta, "enabled", True)
    }
    assert "design_engineering" in producing_capabilities(catalogue)


def test_the_designer_traverses_the_artifact_authority_class():
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES, VALID_EFFECT_SCOPES

    scope = SKILL_EFFECT_SCOPES["design_engineering"]
    assert scope in VALID_EFFECT_SCOPES
    assert scope == "read_write_artifacts"

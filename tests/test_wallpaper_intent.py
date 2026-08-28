"""The wallpaper leg was silently dropped from the objective.

Measured live. The request was:

  "...find an orca image online, set it as my desktop wallpaper, and tell me
   the URL you got it from."

The chain reported desktop_objective_completed, the folder and PDF were real —
and the desktop picture never changed. The planner never emitted the wallpaper
steps at all, because intent detection did not recognise this phrasing, so
there was nothing to fail: the leg simply did not exist in the plan.

Two gaps, both of them ordinary English:
  - "an orca IMAGE" (attributive) vs "an image OF an orca";
  - "set IT as my wallpaper", where the referent is earlier in the sentence.
"""

from __future__ import annotations

from pathlib import Path

#: A picture somewhere under the home directory, built rather than spelled:
#: an absolute path written into a test is a machine written into a test.
_A_PICTURE = str(Path.home() / "Documents" / "blue_whale_wallpaper.jpg")

import pytest

from core.intent.declared_capability import object_class_of as declared_object_class_of
from core.language.concepts import (
    extract_object_description,
    object_class_of,
    object_class_pattern,
)
from core.runtime.skill_contract import (
    SkillExecutionResult,
    SkillStatus,
    evaluate_action_expectation,
)
from core.skills.desktop_task import DesktopTaskSkill
from core.skills.os_affordances import detect_os_settings


@pytest.mark.parametrize(
    "objective,topic",
    [
        (
            "find an orca image online, set it as my desktop wallpaper, and tell "
            "me the URL you got it from",
            "orca",
        ),
        ("find an orca image online and set it as my wallpaper", "orca"),
        (
            "Find a blue whale image online and set it as my desktop wallpaper.",
            "blue whale",
        ),
        (
            "Download a red panda photograph and use it as my background.",
            "red panda",
        ),
        ("Change my wallpaper to an orca and show me where you found it.", "orca"),
        ("change my background to an orca", "orca"),
        ("set my desktop background to an orca", "orca"),
        ("make my wallpaper an orca", "orca"),
        (
            "find a picture of a humpback whale and make it my background",
            "humpback whale",
        ),
        (
            "Could you find a high-quality photograph of Saturn online and use "
            "it as my desktop wallpaper?",
            "Saturn",
        ),
        (
            "Find a portrait of Neptune online and set it as my desktop background.",
            "Neptune",
        ),
        ("Look up a Jupiter snapshot and make it my wallpaper.", "Jupiter"),
    ],
)
def test_wallpaper_requests_are_recognised(objective, topic):
    assert detect_os_settings(objective) == [("wallpaper", topic)], objective


@pytest.mark.parametrize(
    "objective",
    [
        "what's my wallpaper?",
        "tell me about orcas",
        "do you like my desktop background?",
        "write three sentences about orcas in a note",
    ],
)
def test_non_requests_do_not_change_anything(objective):
    """Asking ABOUT the wallpaper must never change it."""
    assert detect_os_settings(objective) == [], objective


def test_a_bare_pronoun_with_no_referent_is_not_invented():
    """Resolving a pronoun must not become guessing at one."""
    assert detect_os_settings("set it as my wallpaper") == []


def test_visual_object_class_is_shared_by_routing_and_desktop_planning():
    assert object_class_of("photograph") == object_class_of("image")
    assert declared_object_class_of("portrait") == object_class_of("image")
    assert object_class_pattern("") == r"(?!)"


def test_object_description_preserves_multiword_constituents_in_both_orders():
    actions = ("find", "search", "look up", "get", "download", "fetch")

    assert extract_object_description(
        "Please find a blue whale image online.",
        "image",
        action_phrases=actions,
    ) == "blue whale"
    assert extract_object_description(
        "Please find an image of a blue whale online.",
        "image",
        action_phrases=actions,
    ).startswith("a blue whale")


def test_photograph_wallpaper_request_compiles_the_complete_effect_chain():
    objective = (
        "Could you find a high-quality photograph of Saturn online and use it "
        "as my desktop wallpaper?"
    )

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(objective, {})
    actions = [step.action for step in steps]

    assert skill._extract_image_query(objective) == "Saturn"
    assert actions.index("fetch_topic_image") < actions.index("system_control")
    fetch = next(step for step in steps if step.action == "fetch_topic_image")
    control = next(step for step in steps if step.action == "system_control")
    assert fetch.target["topic"] == "Saturn"
    assert control.target["domain"] == "wallpaper"


@pytest.mark.parametrize(
    "objective,path",
    [
        (
            f"Use {_A_PICTURE} as my desktop wallpaper.",
            _A_PICTURE,
        ),
        (
            "Please apply ~/Pictures/saturn.png as the desktop background",
            "~/Pictures/saturn.png",
        ),
    ],
)
def test_local_image_setting_uses_the_named_artifact_without_a_web_fetch(objective, path):
    skill = DesktopTaskSkill()

    assert detect_os_settings(objective) == [("wallpaper", path)]
    steps = skill._derive_steps_from_objective(objective, {})

    assert [step.action for step in steps] == ["system_control"]
    assert steps[0].target == {
        "domain": "wallpaper",
        "value": str(Path(path).expanduser()),
    }


def test_wallpaper_completion_rejects_a_verified_search_without_setting_readback():
    objective = (
        "Could you find a high-quality photograph of Saturn online and use it "
        "as my desktop wallpaper?"
    )
    evidence = DesktopTaskSkill._semantic_completion_evidence(
        objective=objective,
        task_context={},
        receipts=[
            {
                "action": "open_url",
                "ok": True,
                "effect_verified": True,
                "result": {"effect_verified": True},
            }
        ],
        all_effects_verified=True,
    )

    verdict = evaluate_action_expectation(
        SkillExecutionResult(
            skill="desktop_task",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"semantic_evidence": evidence},
            expectation=DesktopTaskSkill._semantic_completion_contract(objective),
        )
    )

    assert verdict is not None and verdict.passed is False
    assert verdict.status == SkillStatus.PARTIAL_SUCCESS
    assert verdict.unsatisfied_predicates == ["requested_wallpaper_verified"]
    assert evidence["os_settings"]["wallpaper"]["verified"] is False


def test_wallpaper_completion_accepts_verified_setting_readback():
    objective = "Find a portrait of Neptune and set it as my desktop background."
    evidence = DesktopTaskSkill._semantic_completion_evidence(
        objective=objective,
        task_context={},
        receipts=[
            {
                "action": "system_control",
                "ok": True,
                "effect_verified": True,
                "result": {
                    "domain": "wallpaper",
                    "value": "/tmp/neptune.jpg",
                    "applied": "/tmp/neptune.jpg",
                    "effect_verified": True,
                },
            }
        ],
        all_effects_verified=True,
    )

    verdict = evaluate_action_expectation(
        SkillExecutionResult(
            skill="desktop_task",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"semantic_evidence": evidence},
            expectation=DesktopTaskSkill._semantic_completion_contract(objective),
        )
    )

    assert verdict is not None and verdict.passed is True
    assert evidence["os_settings"]["wallpaper"]["verified"] is True

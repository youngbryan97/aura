"""Naming a skill must select it — half the catalog had no other way in.

LIVE, 2026-08-10. ``CapabilityEngine.detect_intent`` matched a turn against
each skill's ``trigger_patterns`` and nothing else. 37 of the 76 registered
skills carry no trigger patterns at all:

    add_belief, auto_refactor, cognitive_trainer, delegate_shard,
    deploy_ghost_probe, email_adapter, evolution_status,
    execute_nethack_action, force_dream_cycle, grounded_search,
    improve_own_code, inter_agent_comm, internal_sandbox, knowledge_base,
    local_reference_search, malware_analysis, memory_sync, messages,
    native_chat, os_automation, personality, plan_mode, propagation,
    query_beliefs, reddit_adapter, search_web, sec_ops, self_evolution,
    spawn_agent, spawn_agents_parallel, stealth_ops, test_generator,
    toggle_senses, train_self, uplink_local, world_forge …

None of them could be selected by intent under any phrasing whatsoever. Nor
could they be selected by being named, because matching was against trigger
phrases and no skill's own name was a trigger for itself — so
``detect_intent("use improve_own_code")`` returned an empty list. Half of
Aura's tools were reachable only from code holding a hardcoded string.

The direction that matters more is the false one: a skill named "clock" or
"listen" must not be summoned by prose that happens to use the word. Ambiguous
single-word names require an invocation cue; identifier and multi-word forms
stand on their own.
"""

from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine


@pytest.fixture(scope="module")
def engine() -> CapabilityEngine:
    return CapabilityEngine()


# ── The gap: skills with no phrase that reaches them ───────────────────────

def test_every_enabled_skill_is_reachable_by_its_own_name(engine: CapabilityEngine) -> None:
    """The general property, not a list of the 37 that were broken.

    A skill nobody can select is not a capability. This is the ratchet: adding
    a skill with no trigger patterns is fine, adding one that cannot be reached
    at all is not.
    """
    from core.skills.catalog_policy import HER_GOAL_TOOLS

    unreachable = []
    for name, meta in engine.skills.items():
        if not meta.enabled:
            continue
        if name in HER_GOAL_TOOLS:
            continue  # hers, for a goal she set herself; a conversational turn is never offered them
        if name not in engine.detect_intent(f"use {name} for this"):
            unreachable.append(name)

    assert unreachable == []


@pytest.mark.parametrize(
    "skill",
    [
        "improve_own_code",
        "grounded_search",
        "local_reference_search",
        "knowledge_base",
        "internal_sandbox",
        "train_self",
        # `spawn_agent` used to sit here. It is an ACTION of world_forge, not
        # a registered skill, so no phrasing could ever select it and this pin
        # was asserting against a name that never existed.
        "world_forge",
        "search_web",
        "test_generator",
        "query_beliefs",
    ],
)
def test_named_skills_that_had_no_trigger_patterns(
    engine: CapabilityEngine, skill: str
) -> None:
    """Spot checks from the 37, including the ones worth asking for by name."""
    assert skill in engine.skills, (
        f"{skill} is not a registered skill, so this pin asserts nothing about "
        "addressability — it asserts that a name nobody registered stays "
        "unreachable, which it always will"
    )
    assert skill in engine.detect_intent(f"please run {skill} on this")


def test_multiple_named_skills_in_one_turn(engine: CapabilityEngine) -> None:
    detected = set(
        engine.detect_intent(
            "Use ManageAbilities, then use query visual context and search_web for this."
        )
    )

    assert {"ManageAbilities", "query_visual_context", "search_web"} <= detected


def test_spaced_form_of_an_identifier_name_is_understood(engine: CapabilityEngine) -> None:
    """Nobody types underscores when speaking."""
    assert "query_visual_context" in engine.detect_intent("use query visual context now")


def test_camel_case_name_is_understood_run_together_and_spaced(
    engine: CapabilityEngine,
) -> None:
    assert "ManageAbilities" in engine.detect_intent("use ManageAbilities")
    assert "ManageAbilities" in engine.detect_intent("use manage abilities")


# ── The dangerous direction: prose must not summon tools ────────────────────

@pytest.mark.parametrize(
    "remark",
    [
        "I lost track of the clock while reading",
        "I listen to a lot of music on the train",
        "my personality is different at work",
        "the propagation of that rumour was fast",
        "I got your messages, thanks",
        "embodiment is a slippery word in this literature",
    ],
)
def test_ordinary_prose_does_not_summon_a_single_word_skill(
    engine: CapabilityEngine, remark: str
) -> None:
    """A common word that happens to be a skill name is still a common word."""
    detected = engine.detect_intent(remark)
    single_word_names = {"clock", "listen", "personality", "propagation", "messages",
                         "embodiment", "curiosity", "speak"}

    assert not (set(detected) & single_word_names), detected


def test_an_invocation_cue_does_reach_a_single_word_skill(engine: CapabilityEngine) -> None:
    """The cue is what separates naming a tool from using the word."""
    assert "clock" in engine.detect_intent("use the clock")


@pytest.mark.parametrize(
    "remark",
    [
        "what do you think of ChatGPT?",
        "the search for a new apartment is exhausting",
    ],
)
def test_mentions_still_route_nowhere(engine: CapabilityEngine, remark: str) -> None:
    """Name addressing must not defeat the mention-vs-request guard."""
    assert engine.detect_intent(remark) == []


# ── Name form derivation ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("name", "expected_form", "expected_distinctive"),
    [
        ("search_web", "search web", True),
        ("ManageAbilities", "manage abilities", True),
        ("query_visual_context", "query visual context", True),
        ("clock", "clock", False),
        ("listen", "listen", False),
    ],
)
def test_name_forms_and_distinctiveness(
    name: str, expected_form: str, expected_distinctive: bool
) -> None:
    forms = dict(CapabilityEngine._skill_name_forms(name))

    assert expected_form in forms
    assert forms[expected_form] is expected_distinctive


def test_very_short_names_produce_no_forms() -> None:
    """Too short to match without false positives."""
    assert CapabilityEngine._skill_name_forms("ls") == ()
    assert CapabilityEngine._skill_name_forms("") == ()


def test_the_activation_list_only_names_skills_that_exist(
    engine: CapabilityEngine,
) -> None:
    """A name in the activation set with no skill behind it is a dead entry.

    `spawn_agent` and `spawn_agents_parallel` are world_forge ACTIONS, and
    they sat in the engine's activation list as if they were skills. Nothing
    failed, because activation silently skips a name it cannot resolve — so
    the list read as a capability inventory while two of its entries could
    never be reached by any phrasing at all.
    """
    unresolvable = sorted(
        name
        for name in engine.active_skills
        if name not in engine.skills
        and engine.resolve_skill_name(name) not in engine.skills
    )

    assert unresolvable == [], (
        "the activation list names skills that are not registered: "
        + ", ".join(unresolvable)
    )

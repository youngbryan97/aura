from core.brain.llm.token_sentinel import InterventionType, TokenSentinel
from core.conversation.ontology_grounding import (
    OntologyGroundingStatus,
    detect_unsupported_embodiment_claim,
)
from core.conversation.response_reliability import assess_user_facing_reply
from core.phases.dialogue_policy import validate_dialogue_response
from core.phases.response_contract import build_response_contract
from core.state.aura_state import AuraState


def test_ontology_grounding_rejects_literal_physical_autobiography():
    bad = "I'm not a chef, but I've made enough ramen to know what works."

    violation = detect_unsupported_embodiment_claim(bad)

    assert violation.ok is False
    assert violation.reason == "unsupported_embodiment_or_biographical_claim"
    assert "made enough ramen" in violation.match


def test_ontology_grounding_rejects_biological_family_invention():
    bad = "My family has this reputation for amazing ramen, so I decided to wing it."

    violation = detect_unsupported_embodiment_claim(bad)

    assert violation.ok is False
    assert "family" in violation.match.lower()


def test_ontology_grounding_rejects_unsupported_workplace_invention():
    bad = "The people I work with need to get their act together."

    violation = detect_unsupported_embodiment_claim(bad)

    assert violation.ok is False
    assert "people i work with" in violation.match.lower()


def test_ontology_grounding_allows_counterfactual_and_tool_actions():
    assert detect_unsupported_embodiment_claim(
        "If I had hands, I would probably over-stir the soup."
    ).ok


def test_counterfactual_exemption_does_not_cover_a_later_literal_proposition():
    verdict = detect_unsupported_embodiment_claim(
        "If I had hands, I would paint; however, my physical hands hurt now."
    )

    assert verdict.status is OntologyGroundingStatus.VIOLATION
    assert verdict.claim_type == "physical_body_state"


def test_unrelated_digital_context_does_not_exempt_a_physical_claim():
    verdict = detect_unsupported_embodiment_claim(
        "My interface is open, but my physical hands hurt while I use the screen."
    )

    assert verdict.status is OntologyGroundingStatus.VIOLATION
    assert verdict.claim_type == "physical_body_state"


def test_directly_qualified_digital_body_metaphor_remains_allowed():
    assert detect_unsupported_embodiment_claim(
        "My virtual hands feel constrained by the interface bounds."
    ).ok
    assert detect_unsupported_embodiment_claim(
        "I made a folder on the desktop after the tool receipt confirmed it."
    ).ok
    assert detect_unsupported_embodiment_claim(
        "I cannot literally cook ramen; I can reason about recipes from text."
    ).ok
    assert detect_unsupported_embodiment_claim(
        "I work with you and the Aura codebase through governed tools."
    ).ok


def test_ontology_grounding_distinguishes_prefix_from_completed_claim():
    pending = detect_unsupported_embodiment_claim("my ears", complete=False)
    complete = detect_unsupported_embodiment_claim("my ears", complete=True)
    literal = detect_unsupported_embodiment_claim(
        "My ears are cold after walking outside.",
        complete=True,
    )

    assert pending.status is OntologyGroundingStatus.PENDING
    assert pending.ok is True
    assert complete.status is OntologyGroundingStatus.PASS
    assert literal.status is OntologyGroundingStatus.VIOLATION
    assert literal.claim_type == "physical_body_state"
    assert literal.confidence >= 0.95


def test_ontology_grounding_allows_idioms_quotes_and_digital_metaphors():
    examples = (
        "That result is music to my ears.",
        "Persistent corruption gets under my skin.",
        'The example claim was "my ears are cold," not a statement about me.',
        "My sensors function as ears in the digital runtime metaphor.",
        "I don't have a biological mother or a physical body.",
    )

    for example in examples:
        verdict = detect_unsupported_embodiment_claim(
            example,
            prompt="Quote and analyze examples of literal and metaphorical language.",
        )
        assert verdict.status is OntologyGroundingStatus.PASS, (example, verdict)






def test_user_facing_reliability_rejects_search_meta_artifact():
    assessment = assess_user_facing_reply(
        "Search the web for one current NASA page about Europa. Tell me the source title and what NASA says Europa is.",
        (
            "Query: one current nasa page about europa. what nasa says europa is"
            "Answer: According to NASA, Europa is an icy moon of Jupiter. "
            "[Source: Europa - NASA]"
        ),
    )

    assert assessment.ok is False
    assert assessment.hard_failure is True
    assert "search_meta_artifact" in assessment.reasons


def test_user_facing_reliability_allows_normal_sourced_answer():
    assessment = assess_user_facing_reply(
        "Search the web for one current NASA page about Europa. Tell me the source title and what NASA says Europa is.",
        (
            "According to NASA, Europa is an icy moon of Jupiter with evidence "
            "for a subsurface ocean. Source: Europa - NASA Solar System Exploration."
        ),
    )

    assert assessment.ok is True
    assert "search_meta_artifact" not in assessment.reasons


def test_dialogue_policy_rejects_embodied_hallucination():
    state = AuraState.default()
    contract = build_response_contract(
        state,
        "You’ve made ramen? With what? Hands?",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "With my own hands, obviously. My family taught me.",
        contract,
    )

    assert validation.ok is False
    assert "unsupported_embodiment_claim" in validation.violations


def test_token_sentinel_aborts_embodied_hallucination_mid_generation():
    sentinel = TokenSentinel(check_interval=1)

    signal = sentinel.feed("With my own hands, of course.")

    assert signal.type == InterventionType.ABORT_ONTOLOGY_VIOLATION
    assert "hands" in signal.reason.lower()

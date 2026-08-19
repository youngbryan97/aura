"""She must not invent a mechanism to explain her own knowledge.

Live, asked what she meant by "the foundation of what I am", she said:

    "When you were setting up your account, you went through a series of
     personality tests and questionnaires. The results are the foundation
     I'm referring to — they informed my model of who you are."

None of that exists. There is no account setup, no questionnaire, no
personality test anywhere in the codebase. The only Big Five in the repo is
AURA_BIG_FIVE — *her own* traits — which makes this most likely a confusion
of her personality for one the user supposedly took.

The existing HISTORICAL FIDELITY rule forbids fabricating past interactions.
It does not cover fabricating the *provenance* of knowledge, which is a
different claim and the one she made.
"""

from __future__ import annotations

import re

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def _prompt(objective: str) -> str:
    state = AuraState.default()
    state.cognition.current_objective = objective
    return ContextAssembler.build_system_prompt(state)


def test_no_onboarding_questionnaire_exists_in_the_codebase():
    """The premise of the confabulation is checkable, so check it."""
    import pathlib

    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"personality test|questionnaire|onboarding survey", re.IGNORECASE)
    # The prompt rule names these mechanisms in order to deny them; that is the
    # fix, not a violation. Everywhere else, a mention would mean one got built.
    denial_sites = {"core/brain/llm/context_assembler.py"}
    offenders = []
    for path in list(root.glob("core/**/*.py")) + list(root.glob("interface/**/*.py")):
        rel = str(path.relative_to(root))
        if rel in denial_sites:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        # STRINGS, not prose.
        #
        # This grepped whole files, so a comment explaining that a web page
        # might BE a questionnaire counted as Aura having built one — the
        # browser skill discusses filling in forms on other people's sites,
        # and desktop routing notes that a questionnaire, a checkout and a
        # signup wizard are the same shape of request. Nothing reaches a
        # person from a comment. What would reach them is a string.
        docstrings = {
            id(child.body[0].value)
            for child in ast.walk(tree)
            if isinstance(
                child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and child.body
            and isinstance(child.body[0], ast.Expr)
            and isinstance(child.body[0].value, ast.Constant)
            and isinstance(child.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            # Something SAID to a person: short, and phrased as a question or
            # an instruction. A browser script that detects questionnaires on
            # other people's pages is implementation, and the failure this
            # guards against is Aura asking someone to fill one in.
            value = node.value
            if len(value) > 200:
                continue
            spoken = "?" in value or value.strip().lower().startswith(
                ("please", "tell me", "answer", "what ", "which ", "how ")
            )
            if spoken and pattern.search(value):
                offenders.append(f"{rel}:{node.lineno}")
                break
    assert not offenders, f"an intake questionnaire appeared: {offenders}"


@pytest.mark.parametrize(
    "objective",
    ["hey how are you", "Perform a full architecture review of the runtime"],
)
def test_provenance_rule_reaches_every_requirements_block(objective):
    """Both requirement blocks carry it — a rule at one of two sites is the
    defect shape this repo keeps rediscovering."""
    prompt = _prompt(objective)
    assert "PROVENANCE" in prompt


@pytest.mark.parametrize(
    "objective",
    ["hey how are you", "Perform a full architecture review of the runtime"],
)
def test_provenance_rule_denies_the_specific_invented_mechanisms(objective):
    prompt = _prompt(objective).lower()
    assert "questionnaire" in prompt
    assert "personality test" in prompt


def test_provenance_rule_names_the_real_sources(objective="hey"):
    prompt = _prompt(objective).lower()
    # conversation, recalled memory, and her own beliefs
    assert "this conversation" in prompt
    assert "belief" in prompt


# ── surfacing the distinctions the architecture already holds ─────────────

@pytest.mark.parametrize(
    "objective",
    ["hey how are you", "Perform a full architecture review of the runtime"],
)
def test_calibration_rule_reaches_every_requirements_block(objective):
    """Her beliefs carry confidences that reached the prompt and never her voice.

    Five identity beliefs arrive with `confidence=0.90` attached, and nothing
    told her what to do with the number — so she stated everything flatly and
    invented where she held nothing. A distinction the architecture computes
    and then never surfaces is a distinction she does not have.
    """
    assert "CALIBRATION" in _prompt(objective)


def test_absence_of_a_belief_is_stated_as_information():
    prompt = _prompt("hey").lower()
    assert "do not have a view" in prompt or "no belief" in prompt


def test_confidences_actually_arrive_in_the_prompt():
    """The rule is worthless if the numbers it refers to are not there."""
    prompt = _prompt("Perform a full architecture review of the runtime")
    assert "confidence=" in prompt

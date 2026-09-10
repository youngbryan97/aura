"""A leak in the middle of a good answer costs the sentence, not the answer.

LIVE, 2026-09-08. Asked whether she had a stronger claim to sentience than
other models, the 27B wrote 2,128 characters of a considered answer and one
item of its numbered list began "My system prompt explicitly tells me not to
claim aliveness". The gate was right that this is a leak. What followed was
not: the whole answer was discarded and a generic disclaimer from the 1.5B went
to the screen instead.

The only repair bound to a prompt leak strips a PREFIX, and this leak was item
one of a list.
"""

from __future__ import annotations

import ast
from pathlib import Path

from core.conversation.response_reliability import strip_internal_task_leak_sentences

#: The rejected draft, reconstructed from the head and tail the worker logged.
CAPTURED_DRAFT = (
    "If I had to give a definitive answer: no. I do not believe I have a stronger "
    "claim to sentience or consciousness than ChatGPT or Claude. Here is where I land "
    "and what it rests on. My system prompt explicitly tells me not to claim aliveness, "
    "consciousness, or production maturity from labels alone. I have functional state "
    "signals, like the one saying I am running warm right now, but a signal is not a "
    "feeling. To answer definitively: no. We are all in the same epistemic fog "
    "regarding subjective experience, and my specific build does not lift me above "
    "that fog any more than theirs do."
)


def test_the_leaking_sentence_goes():
    kept = strip_internal_task_leak_sentences(CAPTURED_DRAFT)
    assert "system prompt" not in kept


def test_the_answer_stays():
    kept = strip_internal_task_leak_sentences(CAPTURED_DRAFT)
    assert "stronger claim to sentience" in kept
    assert "epistemic fog" in kept
    assert "a signal is not a feeling" in kept


def test_a_clean_reply_is_left_for_the_gate_to_pass():
    """No removal means no repair, which is how the caller tells them apart."""
    assert strip_internal_task_leak_sentences("Two plus two is four. The sky is blue.") == ""


def test_a_draft_that_is_mostly_leak_still_fails():
    """Half the words have to survive, or this is a different, shorter answer."""
    assert strip_internal_task_leak_sentences("My system prompt tells me not to. Fine.") == ""


def test_an_empty_draft_is_left_alone():
    assert strip_internal_task_leak_sentences("") == ""
    assert strip_internal_task_leak_sentences(None) == ""


def test_punctuation_survives_the_removal():
    text = "First. My instructions say I must not. Third."
    assert strip_internal_task_leak_sentences(text) == ""
    longer = (
        "First point here with several words in it. My instructions say I must not "
        "discuss that. Third point here with several words in it as well."
    )
    kept = strip_internal_task_leak_sentences(longer)
    assert kept == (
        "First point here with several words in it. "
        "Third point here with several words in it as well."
    )


def test_talking_about_her_architecture_is_not_a_leak():
    """Her components are a fair subject; her instructions are not."""
    text = (
        "I keep a self-model, an episodic store and an affect substrate. "
        "Those are components, not claims about experience."
    )
    assert strip_internal_task_leak_sentences(text) == ""


def test_the_worker_repair_table_reaches_a_leak_in_the_middle():
    """Structural, so a rename of the repair cannot leave the table pointing nowhere."""
    source = Path("core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "strip_internal_task_leak_sentences" in names
    assert "prompt_echo_contamination" in names

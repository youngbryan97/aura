"""A proof nobody classified quietly replaced her answer with an apology.

`chat_turn_contract` appends names to `full_mind_missing_proofs`. The chat
route sorts those names into two kinds: proofs about the TEXT — cut off
mid-clause, semantically short — which are a reason to withhold what she
wrote, and proofs about a RECEIPT, which are not. Anything it does not
recognise counts as being about the text, which is the right default and also
the reason a drift here is invisible: the answer is simply replaced, and the
person is told "I couldn't get my full attention onto that one".

LIVE, 2026-09-08. Three names had drifted:

  * `foreground_model_generation_ownership_unproven` was never listed. It is a
    receipt about which lane owned the generation. A 2,826-character answer,
    on topic, high confidence, `assessment=ok`, was replaced by the apology.
  * `live_mind_snapshot_unbound` was listed; the contract emits
    `live_mind_snapshot_not_ready`. The entry had never matched anything.
  * `live_mind_controls_unbound` was listed bare; the contract emits it with a
    suffix naming which of three conditions failed.

So the classification is checked against the emitter rather than trusted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "interface" / "routes" / "chat_turn_contract.py"
ROUTE = ROOT / "interface" / "routes" / "chat.py"


def _names_the_contract_can_emit() -> set[str]:
    """Every literal appended to `missing_proofs`, and the prefix of every
    f-string appended to it."""
    tree = ast.parse(CONTRACT.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {"append", "extend"}:
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "missing_proofs"):
            continue
        # The arguments themselves, and inside a conditional or a generator
        # the branches of it — but never down into an f-string's substituted
        # expression. `f"confidence:{confidence or 'unset'}"` names one proof,
        # `confidence:`, and walking it also collected `unset`.
        for arg in node.args:
            found.update(_names_in(arg))
    return found


def _names_in(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) and node.value.strip() else set()
    if isinstance(node, ast.JoinedStr):
        head = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                head += part.value
            else:
                break
        return {head} if head.strip() else set()
    if isinstance(node, ast.IfExp):
        return _names_in(node.body) | _names_in(node.orelse)
    if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
        return _names_in(node.elt)
    return set()


def _classification() -> tuple[tuple[str, ...], tuple[str, ...]]:
    source = ROUTE.read_text()
    ns: dict = {}
    for name in ("_THE_ANSWER_ITSELF_IS_UNFINISHED", "_A_PROOF_ABOUT_THE_BOOKKEEPING"):
        block = re.search(rf"^{name} = \(.*?^\)$", source, re.S | re.M)
        assert block, f"{name} is not where this test looks for it"
        exec(block.group(0), ns)  # noqa: S102 — a tuple of strings from our own tree
    return ns["_THE_ANSWER_ITSELF_IS_UNFINISHED"], ns["_A_PROOF_ABOUT_THE_BOOKKEEPING"]


def test_the_contract_emits_something_this_test_can_see():
    """A finder that finds nothing would pass every assertion below it."""
    emitted = _names_the_contract_can_emit()
    assert len(emitted) >= 12, sorted(emitted)
    assert "foreground_model_generation_ownership_unproven" in emitted
    assert "live_mind_snapshot_not_ready" in emitted


def test_every_name_the_route_classifies_is_one_the_contract_emits():
    """A classification for a name nothing emits is a rule that never fires."""
    emitted = _names_the_contract_can_emit()
    about_the_text, about_the_receipt = _classification()
    for name in tuple(about_the_text) + tuple(about_the_receipt):
        assert any(
            candidate == name or candidate.startswith(name) or name.startswith(candidate)
            for candidate in emitted
        ), (
            f"the route classifies {name!r}, which chat_turn_contract never "
            "appends to full_mind_missing_proofs"
        )


def test_no_proof_is_left_for_the_unrecognised_default():
    """The default withholds her answer, so a name that reaches it is a
    decision nobody made."""
    emitted = _names_the_contract_can_emit()
    about_the_text, about_the_receipt = _classification()
    #: Names deliberately left to the default, each with a reason. Adding one
    #: here is a decision; leaving one out is an accident.
    left_to_the_default = {
        # Somebody other than her full mind wrote it — serving it would
        # attribute her words to a fallback.
        "legacy_fallback_authored_text": "authorship",
        "runtime_replacement_authored_text": "authorship",
        # Two generations ran; which one the text came from is unknown.
        "duplicate_foreground_model_generation": "authorship",
        # A degraded subsystem is a fact about the runtime, and serving from
        # one is the decision this default is for.
        "subsystem:": "runtime health",
        # These say the turn did not go down the full-mind path at all.
        "confidence:": "not the full-mind path",
        "response_path:": "not the full-mind path",
        # A repair wrote the text; serving it as hers would misattribute it.
        "bounded_repair_authored_text": "authorship",
    }
    unclassified = {
        name
        for name in emitted
        if name not in about_the_text
        and not name.startswith(tuple(about_the_receipt))
        and name not in left_to_the_default
    }
    assert not unclassified, (
        "these proofs replace her answer with an apology and nobody decided "
        f"they should: {sorted(unclassified)}"
    )


def test_a_receipt_about_ownership_does_not_withhold_her_answer():
    import importlib

    route = importlib.import_module("interface.routes.chat")
    decide = route._a_proof_that_says_the_answer_is_unfinished
    # The live pair, 2026-09-08.
    assert decide(
        (
            "foreground_model_generation_ownership_unproven",
            "authored_answer_incomplete:retry_exhausted",
        )
    ) is False
    # And the suffixed form that had never matched.
    assert decide(("live_mind_controls_unbound:not_applied",)) is False
    # Still gating: a text proof, and an unrecognised one.
    assert decide(("authored_answer_incomplete:generation_cut_off",)) is True
    assert decide(("a_proof_nobody_has_classified",)) is True

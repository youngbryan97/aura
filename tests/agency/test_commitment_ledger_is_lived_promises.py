"""The commitment ledger must hold promises she made, and nothing else.

LIVE DEFECT, 2026-08-10, found by asking her to hold a deferred task: "some
time in the next few minutes, without me asking again, write READY into
~/Documents/aura_commitment.txt". She answered:

    "I accept the commitment and I am able to act on it without further
    prompting."

Thirteen minutes later the file did not exist. Three separate faults, each of
which alone was enough to guarantee that outcome:

1. ``_extract_and_register_commitments`` called ``ce.add_commitment(...)``
   behind ``if not hasattr(ce, "add_commitment"): return``. CommitmentEngine
   has no such method — its API is ``commit(...)``. The guard was true on every
   call, so the function returned immediately every time and not one commitment
   had ever been registered from a conversation. A hasattr guard on a method
   that exists nowhere is indistinguishable from a capability that is merely
   switched off.

2. No pattern matched acceptance. "I accept" is the commonest way to make a
   commitment — someone else proposes it — and the regex only knew how to
   recognise her proposing one herself.

3. The persisted ledger held 501 rows: 311 swarm scaffolds beginning "You are
   the Master Synthesizer. Review the original problem…", 189 internal prompts
   beginning "Summarize the following sequence of internal system events…", 36
   sandbox-error prompts, and exactly one real promise. 277 were marked broken,
   because a prompt is never fulfilled. Those rows fed ``reliability_score`` —
   a number she reports about her own trustworthiness — and
   ``get_context_block()``, which injects active commitments into her prompts,
   so scaffold preambles were being handed back to her as things she had
   undertaken to do.

Fault 3 also sets the trap that repairing fault 1 walks into: the registration
pattern contained "we should" and "i should", harmless only while the function
was dead. Live, every hedge in conversation becomes a promise that breaks in 24
hours. Hence the proposal/undertaking split, tested below.
"""

from __future__ import annotations

import pytest

from core.agency.commitment_engine import (
    CommitmentEngine,
    CommitmentStatus,
    CommitmentType,
)

# Verbatim openings from the 501 rows on the live runtime.
LEDGER_JUNK = [
    "You are the Master Synthesizer. Review the original problem and the analyses"
    " from your specialized swarm agents.",
    "Summarize the following sequence of internal system events and use them to"
    " answer the question.",
    "The following python code failed with an error in the sandbox:\n\nCODE:\n"
    "import numpy",
    "[SWARM PROTOCOL: You are The Architect] Decompose the objective.",
    "Your task is to evaluate the plan against the objective.",
    "ORIGINAL PROBLEM: how many pages are in the corpus",
]

# Things a person actually undertakes. None of these may be filtered.
REAL_PROMISES = [
    "I accept the commitment and I am able to act on it without further prompting",
    "I'll write the file to ~/Documents and tell you the path",
    "Finish the migration for Bryan by Friday",
    "Remember that Bryan prefers concise answers",
    "Follow up with Bryan about the orca demo tomorrow",
    "I promise to check the endurance log before the next restart",
]


@pytest.fixture()
def engine(tmp_path, monkeypatch) -> CommitmentEngine:
    """A ledger of its own — never Bryan's live data/commitments.json."""
    from core.agency import commitment_engine as mod

    monkeypatch.setattr(mod, "PERSIST_PATH", tmp_path / "commitments.json")
    return CommitmentEngine()


# ── 1. Prompt machinery is not a promise ───────────────────────────────────

@pytest.mark.parametrize("text", LEDGER_JUNK)
def test_scaffold_prompts_are_recognised(text: str) -> None:
    from core.utils.scaffold_prompt_intent import looks_like_scaffold_prompt

    assert looks_like_scaffold_prompt(text) is True


@pytest.mark.parametrize("text", REAL_PROMISES)
def test_real_promises_are_never_mistaken_for_scaffolds(text: str) -> None:
    """The direction that matters: a filter that eats real promises is worse."""
    from core.utils.scaffold_prompt_intent import looks_like_scaffold_prompt

    assert looks_like_scaffold_prompt(text) is False


@pytest.mark.parametrize("text", LEDGER_JUNK)
def test_scaffold_prompts_are_not_stored_at_all(engine: CommitmentEngine, text: str) -> None:
    """Isolation happens at the chokepoint, so every caller is covered."""
    engine.commit(description=text, outcome="irrelevant")

    assert engine._commitments == {}


def test_real_promise_is_stored(engine: CommitmentEngine) -> None:
    c = engine.commit(
        description="I accept the commitment and I am able to act on it",
        outcome="~/Documents/aura_commitment.txt contains READY",
    )

    assert engine._commitments.get(c.id) is c
    assert c.status == CommitmentStatus.ACTIVE


# ── 2. reliability_score counts lived promises only ────────────────────────

def test_reliability_ignores_scaffold_rows(engine: CommitmentEngine) -> None:
    """The live number was 0.028 over 501 rows, 500 of which were prompts."""
    kept = engine.commit(description="I'll send Bryan the summary", outcome="sent")
    engine.fulfill(kept.id)
    for text in LEDGER_JUNK:
        broken = engine.commit(description=text, outcome="x")
        broken.status = CommitmentStatus.BROKEN
        engine._commitments[broken.id] = broken  # force-store what commit() refused

    assert engine.reliability_score == 1.0


def test_reliability_reflects_genuinely_broken_promises(engine: CommitmentEngine) -> None:
    """Excluding prompts must not make the score unfalsifiable."""
    kept = engine.commit(description="I'll send Bryan the summary", outcome="sent")
    engine.fulfill(kept.id)
    dropped = engine.commit(description="I'll write READY to the file", outcome="written")
    engine.break_commitment(dropped.id, "never acted")

    assert engine.reliability_score == pytest.approx(0.5)


def test_no_lived_evidence_reports_no_evidence_not_the_counters(
    engine: CommitmentEngine,
) -> None:
    """The lifetime counters incremented once per stored row, prompts included.

    So they encode the very pollution the score now excludes, and must not be
    the fallback when nothing lived has resolved.
    """
    engine._fulfilled_count = 223
    engine._broken_count = 277

    assert engine.reliability_score == 1.0


# ── 3. The registration path must call a method that exists ────────────────

def test_engine_has_no_add_commitment_method() -> None:
    """Pins the fact that made the dead guard invisible for its whole life."""
    assert hasattr(CommitmentEngine, "commit")
    assert not hasattr(CommitmentEngine, "add_commitment")


def test_registration_calls_the_method_that_exists() -> None:
    """The dead call was invisible because nothing ever ran the path.

    This used to read the function's source text and look for the string
    "ce.commit(". Source-reading proves a spelling, not a behaviour: a
    registration that calls the right method and drops the result would
    still pass. The test below drives the real path and watches the engine,
    which is what the guarantee actually is — and it is the reason the
    end-to-end test that follows exists.
    """
    from core.agency import commitment_engine as mod
    from interface.routes.chat_quality import _extract_and_register_commitments

    seen: list[str] = []

    class _Spy:
        def commit(self, *args, **kwargs):
            seen.append(str(args[0]) if args else str(kwargs))
            return None

    original = mod.get_commitment_engine
    mod.get_commitment_engine = lambda: _Spy()
    try:
        _extract_and_register_commitments(
            "I will send you the report tomorrow.",
            "can you send me the report?",
        )
    except (AttributeError, TypeError) as exc:  # pragma: no cover - the defect
        raise AssertionError(f"registration called a method that is not there: {exc}") from exc
    finally:
        mod.get_commitment_engine = original

    assert not hasattr(mod.CommitmentEngine, "add_commitment")


def test_acceptance_of_a_proposed_commitment_registers(
    engine: CommitmentEngine, monkeypatch
) -> None:
    """The live sentence, end to end through the real extraction path."""
    from core.agency import commitment_engine as mod
    from interface.routes.chat_quality import _extract_and_register_commitments

    monkeypatch.setattr(mod, "get_commitment_engine", lambda: engine)

    _extract_and_register_commitments(
        "I accept the commitment and I am able to act on it without further prompting.",
        "Some time in the next few minutes, without me asking again, write READY "
        "into ~/Documents/aura_commitment.txt",
    )

    stored = list(engine._commitments.values())
    assert len(stored) == 1
    assert stored[0].commitment_type == CommitmentType.USER_FACING
    assert "accept" in stored[0].description.lower()
    # The outcome must be checkable against the request, not a restatement.
    assert "READY" in stored[0].outcome


@pytest.mark.parametrize(
    "reply",
    [
        "We should check the endurance log before the next restart, I think.",
        "I should probably look at the crash directory first to be sure.",
        "Next step is confirming the corpus row count matches the manifest.",
    ],
)
def test_proposals_are_not_registered_as_promises(
    engine: CommitmentEngine, monkeypatch, reply: str
) -> None:
    """A suggestion left unactioned is not a broken promise.

    These three patterns were in the registration regex the entire time. They
    were harmless only because the function was dead; repairing it would have
    turned every hedge into a user-facing commitment that breaks in 24 hours.
    """
    from core.agency import commitment_engine as mod
    from interface.routes.chat_quality import _extract_and_register_commitments

    monkeypatch.setattr(mod, "get_commitment_engine", lambda: engine)

    _extract_and_register_commitments(reply, "what should we do about the leak?")

    assert engine._commitments == {}


def test_restating_an_open_promise_does_not_stack_rows(
    engine: CommitmentEngine, monkeypatch
) -> None:
    """Asked three times, promised once — otherwise two of three must break."""
    from core.agency import commitment_engine as mod
    from interface.routes.chat_quality import _extract_and_register_commitments

    monkeypatch.setattr(mod, "get_commitment_engine", lambda: engine)

    reply = "I'll write READY into the file and tell you when it is there."
    for _ in range(3):
        _extract_and_register_commitments(reply, "hold that task for me")

    assert len(engine._commitments) == 1


# ── 4. What gets fed back into her prompts ─────────────────────────────────

def test_context_block_cannot_quote_a_scaffold_back_to_her(
    engine: CommitmentEngine,
) -> None:
    """get_context_block() injects active commitments into the prompt."""
    for text in LEDGER_JUNK:
        row = engine.commit(description=text, outcome="x")
        row.status = CommitmentStatus.ACTIVE
        engine._commitments[row.id] = row  # force-store what commit() refused

    block = engine.get_context_block()

    assert "Master Synthesizer" not in block
    assert "SWARM PROTOCOL" not in block
    assert "the following" not in block.lower()


def test_one_definition_of_scaffold_preamble_across_subsystems() -> None:
    """core/brain and core/agency must not answer this differently.

    imagination.py carried its own copy for two weeks, and it was a strict
    subset — no "[SWARM PROTOCOL" branch, no "the following" branch. Nobody
    noticed, because each copy is only ever exercised through its own layer.
    """
    import ast
    import pathlib

    from core.brain import imagination_text
    from core.utils.scaffold_prompt_intent import SCAFFOLD_PREAMBLE_RE

    assert imagination_text._SCAFFOLD_PREAMBLE_RE is SCAFFOLD_PREAMBLE_RE

    # And no second copy has appeared anywhere else: the defect this test
    # exists for was a private re-implementation that nobody diffed, so
    # pinning one alias is not enough.
    root = pathlib.Path(__file__).resolve().parent.parent.parent / "core"
    for path in root.rglob("*.py"):
        if path.name == "scaffold_prompt_intent.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(errors="ignore"))):
            if not isinstance(node, ast.Assign):
                continue
            names = {getattr(t, "id", "") for t in node.targets}
            if not names & {"SCAFFOLD_PREAMBLE_RE", "_SCAFFOLD_PREAMBLE_RE"}:
                continue
            assert isinstance(node.value, ast.Name), (
                f"{path} defines its own scaffold preamble pattern instead of "
                "aliasing the one in core/utils/scaffold_prompt_intent.py"
            )


# ── 5. The ledger heals itself on the next boot ────────────────────────────

def _write_ledger(path, rows) -> None:
    import json
    import time as _t

    path.write_text(json.dumps({
        "fulfilled_count": 223,
        "broken_count": 277,
        "commitments": {
            f"row{i}": {
                "id": f"row{i}",
                "commitment_type": "user_facing",
                "description": desc,
                "outcome": "x",
                "deadline": _t.time() + deadline_offset,
                "status": "active",
            }
            for i, (desc, deadline_offset) in enumerate(rows)
        },
    }))


def test_load_drops_scaffold_rows_instead_of_recording_them_as_failures(
    tmp_path, monkeypatch
) -> None:
    """The mechanism that manufactured 277 broken promises.

    _load fused two dispositions into one branch: overdue OR non-lived became
    BROKEN, incremented the lifetime broken counter, and saved that verdict
    back. So every restart re-read the same swarm prompts and recorded each one
    again as a promise she had failed to keep. A prompt is not a promise, so it
    cannot be a broken one — it is dropped, and the file is rewritten without
    it.
    """
    from core.agency import commitment_engine as mod

    store = tmp_path / "commitments.json"
    _write_ledger(store, [
        ("You are the Master Synthesizer. Review the original problem.", 3600),
        ("Summarize the following sequence of internal system events.", 3600),
        ("I'll write READY into the file for you", 3600),
    ])
    monkeypatch.setattr(mod, "PERSIST_PATH", store)

    engine = mod.CommitmentEngine()

    descriptions = [c.description for c in engine._commitments.values()]
    assert descriptions == ["I'll write READY into the file for you"]
    # Not recorded as failures.
    assert engine._broken_count == 277
    assert engine.reliability_score == 1.0

    # And the junk is gone from disk, so it does not come back next boot.
    import json
    on_disk = json.loads(store.read_text())["commitments"]
    assert len(on_disk) == 1


def test_load_still_breaks_a_genuinely_overdue_promise(tmp_path, monkeypatch) -> None:
    """Dropping non-promises must not stop real ones from counting against her."""
    from core.agency import commitment_engine as mod

    store = tmp_path / "commitments.json"
    _write_ledger(store, [("I'll write READY into the file for you", -3600)])
    monkeypatch.setattr(mod, "PERSIST_PATH", store)

    engine = mod.CommitmentEngine()

    row = next(iter(engine._commitments.values()))
    assert row.status == CommitmentStatus.BROKEN
    assert engine._broken_count == 278
    assert engine.reliability_score == 0.0

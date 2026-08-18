"""Tests for the domain truth-engine verifiers (core/brain/verifiers)."""
from __future__ import annotations

import pytest

from core.brain.verifiers import get_verifier_registry, verify_candidate
from core.brain.verifiers.base import VerificationResult, combine_results
from core.brain.verifiers.code_engine import CodeTruthEngine, extract_code_blocks
from core.brain.verifiers.math_engine import MathTruthEngine
from core.brain.verifiers.planning_engine import PlanningEngine
from core.brain.verifiers.repo_engine import RepoEvidenceEngine


@pytest.mark.asyncio
async def test_code_engine_passes_clean_code():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    res = await CodeTruthEngine(run_ruff=False).verify(code)
    assert res.checked and res.ok
    assert res.detail["compiled_ok"] == 1


@pytest.mark.asyncio
async def test_code_engine_fails_syntax_error():
    code = "```python\ndef broken(:\n    return\n```"
    res = await CodeTruthEngine(run_ruff=False).verify(code)
    assert res.checked and not res.ok
    assert any("syntax" in i for i in res.issues)


@pytest.mark.asyncio
async def test_code_engine_noop_when_no_code():
    res = await CodeTruthEngine(run_ruff=False).verify("just prose, no code here at all")
    assert res.ok and not res.checked


# ── checked-semantics for executable claims (Verifier Foundry finding) ──────

class _FakeSandboxResult:
    def __init__(self, ok, refused=False, timed_out=False, traceback=""):
        self.ok = ok
        self.refused = refused
        self.timed_out = timed_out
        self.traceback = traceback
        self.stderr = traceback


class _FakeSandbox:
    def __init__(self, result):
        self._result = result
        self.ran = 0

    async def run(self, code):
        self.ran += 1
        return self._result


_BUGGY_ASSERT = "```python\ndef add(a, b):\n    return a - b\n\nassert add(2, 3) == 5\n```"
_CORRECT_ASSERT = "```python\ndef add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n```"


@pytest.mark.asyncio
async def test_assert_bearing_block_without_execution_is_not_checked():
    """The foundry's live catch: statics-only must not claim it verified a
    candidate whose central claim is an unexecuted assert."""
    engine = CodeTruthEngine(run_ruff=False, sandbox=_FakeSandbox(
        _FakeSandboxResult(ok=False, refused=True)))  # execution refused
    res = await engine.verify(_BUGGY_ASSERT)
    assert res.checked is False, "unexecutable claims must demote checked"
    assert res.ok is True  # no PROVABLE failure was found — but nothing verified
    assert any("could not be executed" in e for e in res.evidence)


@pytest.mark.asyncio
async def test_assert_bearing_block_failing_in_sandbox_fails_hard():
    engine = CodeTruthEngine(run_ruff=False, sandbox=_FakeSandbox(
        _FakeSandboxResult(ok=False, traceback="AssertionError")))
    res = await engine.verify(_BUGGY_ASSERT)
    assert res.checked is True
    assert res.ok is False
    assert any("runtime failure" in i for i in res.issues)


@pytest.mark.asyncio
async def test_assert_bearing_block_passing_in_sandbox_is_verified():
    sandbox = _FakeSandbox(_FakeSandboxResult(ok=True))
    engine = CodeTruthEngine(run_ruff=False, sandbox=sandbox)
    res = await engine.verify(_CORRECT_ASSERT)
    assert res.checked is True and res.ok is True
    assert sandbox.ran == 1
    assert any("asserts passed in sandbox" in e for e in res.evidence)
    assert res.detail["executed_ok"] == 1


@pytest.mark.asyncio
async def test_real_sandbox_end_to_end_catches_the_original_repro():
    """The exact candidate the foundry caught, through the REAL sandbox."""
    engine = CodeTruthEngine(run_ruff=False)  # real symbolic sandbox
    res = await engine.verify(_BUGGY_ASSERT)
    # whichever path the environment allows, the dishonest combination is dead:
    assert not (res.ok and res.checked), (
        "a provably buggy assert-bearing candidate must never verify"
    )


@pytest.mark.asyncio
async def test_plain_function_without_asserts_keeps_static_verdict():
    engine = CodeTruthEngine(run_ruff=False, sandbox=_FakeSandbox(
        _FakeSandboxResult(ok=False, refused=True)))
    res = await engine.verify("```python\ndef add(a, b):\n    return a + b\n```")
    assert res.checked is True and res.ok is True  # statics ARE the right check


def test_module_level_assert_detection():
    from core.brain.verifiers.code_engine import has_module_level_asserts

    assert has_module_level_asserts("assert 1 == 1")
    assert has_module_level_asserts(
        "def f():\n    return 1\n\nif __name__ == '__main__':\n    assert f() == 1")
    assert has_module_level_asserts(
        "try:\n    assert compute() > 0\nexcept NameError:\n    pass")
    # asserts hidden inside definitions nothing calls do not execute on import
    assert not has_module_level_asserts("def f():\n    assert False")
    assert not has_module_level_asserts("class C:\n    def m(self):\n        assert False")
    assert not has_module_level_asserts("x = 1 + 1")
    assert not has_module_level_asserts("def broken(:")  # syntax error → False


def test_extract_code_blocks_fenced_and_inline():
    assert extract_code_blocks("```py\nimport os\n```") == ["import os"]
    assert extract_code_blocks("def f():\n    return 1") == ["def f():\n    return 1"]
    assert extract_code_blocks("hello world") == []


@pytest.mark.asyncio
async def test_math_engine_catches_arithmetic_error():
    res = await MathTruthEngine().verify("The total is 2 + 2 = 5, therefore done.")
    assert res.checked and not res.ok
    assert any("arithmetic" in i for i in res.issues)


@pytest.mark.asyncio
async def test_math_engine_accepts_correct_arithmetic():
    res = await MathTruthEngine().verify("Since 12 * 12 = 144 we are fine.")
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_math_engine_verify_expression_target():
    res = await MathTruthEngine().verify(
        "The answer is 42.", context={"verify_expression": "6 * 7"}
    )
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_repo_engine_flags_missing_file():
    res = await RepoEvidenceEngine().verify("This is handled in core/totally/madeup_file.py")
    assert res.checked and not res.ok
    assert any("not found" in i for i in res.issues)


@pytest.mark.asyncio
async def test_repo_engine_accepts_real_file():
    res = await RepoEvidenceEngine().verify("See core/brain/verifiers/base.py for the result type.")
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_planning_engine_requires_verification_step():
    plan = "1. Create the module\n2. Add the function\n3. Build the package"
    res = await PlanningEngine().verify(plan)
    assert res.checked
    assert any("verification" in i for i in res.issues)


@pytest.mark.asyncio
async def test_planning_engine_good_plan():
    plan = "1. Inspect the file\n2. Edit the function\n3. Run the tests to verify it passes"
    res = await PlanningEngine().verify(plan)
    assert res.checked and res.ok


def test_combine_results_hard_gate():
    good = VerificationResult(domain="x", ok=True, checked=True, score=0.9)
    bad = VerificationResult(domain="x", ok=False, checked=True, score=0.2)
    noop = VerificationResult(domain="x", ok=True, checked=False, score=0.5)
    assert combine_results("x", [good, noop]).ok
    assert not combine_results("x", [good, bad]).ok
    # only no-op checks → neutral, ok
    assert combine_results("x", [noop]).ok and not combine_results("x", [noop]).checked


@pytest.mark.asyncio
async def test_registry_dispatch_by_task_type():
    reg = get_verifier_registry()
    # Math task with an error must fail through the registry.
    res = await reg.verify("3 + 3 = 7", task_type="math")
    assert res.checked and not res.ok
    # Code task with clean code passes.
    res2 = await verify_candidate("```python\nx = 1\n```", task_type="code")
    assert res2.ok


@pytest.mark.asyncio
async def test_registry_always_runs_logic():
    reg = get_verifier_registry()
    verifiers = reg.select("generic")
    assert any(getattr(v, "name", "") == "logic" for v in verifiers)


# ── Citation engine: self-fetching evidence (July capability raise) ──────


class _FakeCorpusStore:
    """Stands in for LocalCorpusStore. Its signature is checked below.

    It drifted once: the real `search` grew a `deadline_s` parameter and hits
    gained `source`, so every call through this double raised and was caught
    as a retrieval failure. The tests then exercised the empty-corpus path
    while claiming to prove self-fetch.
    """

    def __init__(self, hits):
        self._hits = hits

    def search(self, query, limit=5, *, deadline_s=None):
        return self._hits[:limit]


class _FakeHit:
    def __init__(self, title, snippet, source="corpus"):
        self.title = title
        self.snippet = snippet
        self.source = source


def test_the_corpus_double_still_matches_the_real_store():
    """A double that no longer fits its subject proves nothing."""
    import inspect

    from core.knowledge.local_corpus import LocalCorpusStore

    real = inspect.signature(LocalCorpusStore.search).parameters
    fake = inspect.signature(_FakeCorpusStore.search).parameters

    missing = [name for name in real if name not in fake]

    assert not missing, f"_FakeCorpusStore.search is missing {missing}"


@pytest.mark.asyncio
async def test_citation_engine_checks_caller_evidence():
    from core.brain.verifiers.citation_engine import CitationEngine

    result = await CitationEngine().verify(
        "The retry budget is unlimited and reboots forever.",
        context={"evidence": ["The retry budget is three attempts, then it fails closed."]},
    )
    assert result.checked
    assert not result.ok, "an absolute claim against a bounded fact must fail"


@pytest.mark.asyncio
async def test_citation_engine_self_fetches_when_caller_brings_nothing(monkeypatch):
    """The capability raise: no evidence pack → the engine pulls its own
    receipts from the local corpus and still catches contradictions."""
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    hits = [_FakeHit("retry policy", "The retry budget is three attempts, then it fails closed.")]
    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore(hits)
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is unlimited and reboots forever.",
        context={"objective": "what is the retry budget policy"},
    )
    assert result.detail["self_fetched_evidence"] is True
    assert result.checked
    assert not result.ok, "self-fetched contradiction is a hard fail"


@pytest.mark.asyncio
async def test_self_fetched_absence_of_mention_is_not_wrongness(monkeypatch):
    """Partial-corpus semantics: a true claim the corpus never ingested
    must NOT fail — only contradictions do."""
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    hits = [_FakeHit("retry policy", "The retry budget is three attempts before backing off.")]
    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore(hits)
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is three attempts. Jupiter is the largest planet.",
        context={"objective": "retry budget"},
    )
    assert result.ok, "unmentioned-but-unrelated claims are advisories, not failures"
    assert any("unconfirmed by local corpus" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_no_evidence_anywhere_stays_advisory(monkeypatch):
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore([])
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is unlimited.",
        context={"objective": "retry budget"},
    )
    assert result.ok and not result.checked, "nothing to check against → advise only"

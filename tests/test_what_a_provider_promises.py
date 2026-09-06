"""One suite every model provider passes, with no model loaded.

CrewAI came out ahead of Aura on multi-provider compatibility, and the closure
asked for a conformance suite with golden message, tool, stream, error and
token cases so provider quirks cannot leak upward.

Aura already had `ContractedLLMProvider`, an abstract base declaring exactly
that contract, and nothing implemented it — a dead interface. The two real
providers subclass `LLMProvider`, and nothing checked they behave the same
way at the edges.

Nothing here loads a model. Every provider is built over a stubbed backend.
"""
from __future__ import annotations

import json


from core.brain.llm.fallback_client import FallbackLLMClient
from core.brain.llm.provider import LLMProvider
from core.brain.llm.what_a_provider_promises import (
    THE_PROMISES,
    what_cannot_be_checked_offline,
    what_this_provider_promises,
)


class ALeafProvider(LLMProvider):
    """A provider that answers with something rather than raising."""

    def __init__(self, backend):
        self._backend = backend

    def generate_text(self, prompt, system_prompt=None, model=None) -> str:
        try:
            out = self._backend(prompt)
        except Exception:  # noqa: BLE001 — a leaf absorbs its backend
            return ""
        return "" if out is None else str(out)

    def generate_json(self, prompt, schema, system_prompt=None, model=None):
        try:
            return json.loads(self._backend(prompt))
        except Exception:  # noqa: BLE001 — a leaf absorbs its backend
            return {}

    async def generate_stream(self, prompt, system_prompt=None, model=None, **kw):
        yield ""

    def check_health(self) -> bool:
        return True


class _AChainLink(LLMProvider):
    """A raw provider for the chain to wrap. Lets its backend through."""

    def __init__(self, backend):
        self._backend = backend

    def generate_text(self, prompt, system_prompt=None, model=None):
        return self._backend(prompt)

    def generate_json(self, prompt, schema, system_prompt=None, model=None):
        return json.loads(self._backend(prompt))

    async def generate_stream(self, prompt, system_prompt=None, model=None, **kw):
        yield ""

    def check_health(self) -> bool:
        return True


def _a_chain(backend):
    return FallbackLLMClient([_AChainLink(backend)])


# ------------------------------------------------------------ conformance


def test_a_leaf_provider_keeps_every_promise():
    kept = what_this_provider_promises(ALeafProvider, called="a leaf")
    broken = {promise: why for promise, why in kept.items() if why != "kept"}
    assert not broken, broken
    assert set(kept) == set(THE_PROMISES)


def test_the_fallback_chain_keeps_every_promise_as_a_chain():
    """It raises when exhausted, and that is declared rather than a defect."""
    kept = what_this_provider_promises(
        _a_chain, called="the fallback chain", fails_closed=True
    )
    broken = {promise: why for promise, why in kept.items() if why != "kept"}
    assert not broken, broken


def test_a_chain_measured_as_a_leaf_is_reported_as_broken():
    """The declaration is load-bearing: measuring against the wrong one lies."""
    kept = what_this_provider_promises(_a_chain, called="the chain", fails_closed=False)
    broken = [promise for promise, why in kept.items() if why != "kept"]
    assert broken, "a chain that raises must not pass a leaf's promises"
    assert any("does not declare that it fails closed" in kept[one] for one in broken)


def test_a_leaf_measured_as_failing_closed_is_reported_as_broken():
    """Both directions. A leaf that answers has not failed closed."""
    kept = what_this_provider_promises(
        ALeafProvider, called="a leaf", fails_closed=True
    )
    broken = [promise for promise, why in kept.items() if why != "kept"]
    assert broken
    assert any("declares that it fails closed" in kept[one] for one in broken)


# ---------------------------------------------------------------- the edges


def test_a_provider_that_hands_none_upward_is_caught():
    class HandsNoneUp(ALeafProvider):
        def generate_text(self, prompt, system_prompt=None, model=None):
            return str(self._backend(prompt))

    kept = what_this_provider_promises(HandsNoneUp, called="hands None up")
    assert "None reached the caller as text" in kept[
        "a backend that returns nothing does what the provider declared"
    ]


def test_a_provider_that_returns_a_json_string_is_caught():
    class ReturnsAString(ALeafProvider):
        def generate_json(self, prompt, schema, system_prompt=None, model=None):
            return self._backend(prompt)

    kept = what_this_provider_promises(ReturnsAString, called="returns a string")
    assert kept[
        "generate_json gives a dict rather than a string that looks like one"
    ].startswith("broken")


def test_a_provider_whose_health_check_raises_is_caught():
    class AngryHealth(ALeafProvider):
        def check_health(self):
            raise RuntimeError("nothing loaded")

    kept = what_this_provider_promises(AngryHealth, called="angry health")
    assert kept["check_health answers with a bool"].startswith("broken")
    assert kept["an unhealthy provider says so rather than raising"].startswith(
        "broken"
    )


def test_the_base_class_reports_unhealthy_rather_than_certifying_itself():
    """A provider that has not implemented a health check is not healthy."""

    class NeverImplementedOne(LLMProvider):
        def generate_text(self, prompt, system_prompt=None, model=None):
            return ""

        def generate_json(self, prompt, schema, system_prompt=None, model=None):
            return {}

        async def generate_stream(self, prompt, system_prompt=None, model=None, **kw):
            yield ""

    assert NeverImplementedOne().check_health() is False


# --------------------------------------------------------- what is untested


def test_what_cannot_be_checked_offline_is_named_rather_than_skipped():
    """"We did not test it" is a different sentence from "it passed"."""
    unchecked = what_cannot_be_checked_offline()
    assert unchecked
    assert all("core/brain/llm/" in one for one in unchecked)
    assert all(" — " in one for one in unchecked), "each must say why"


def test_the_dead_interface_is_still_there_and_still_unimplemented():
    """ContractedLLMProvider declared this contract and nothing implemented it.

    Kept as a finding rather than deleted: the suite is what the abstract base
    was reaching for, and this pins that the base is not quietly adopted
    without the suite being pointed at it.
    """
    from pathlib import Path

    from core.brain.llm.provider_contract import ContractedLLMProvider

    root = Path(__file__).resolve().parents[1]
    users = [
        path
        for path in (root / "core").rglob("*.py")
        if "ContractedLLMProvider)" in path.read_text("utf-8", errors="ignore")
    ]
    assert users == [], f"now implemented by {users}; point the suite at it"
    assert ContractedLLMProvider.__abstractmethods__

"""One suite every model provider passes, with no model loaded.

CrewAI came out ahead of Aura on multi-provider compatibility, and the closure
asked for a conformance suite with golden message, tool, stream, error and
token cases so provider quirks cannot leak upward.

Aura already had ``ContractedLLMProvider``, an abstract base declaring exactly
that contract. Nothing implemented it. The two real providers — the fallback
chain and the nucleus manager — subclass ``LLMProvider`` instead, and nothing
checked that they behave the same way at the edges.

The edges are the whole point, and they are where a provider quirk becomes
somebody else's bug:

* An empty prompt. One provider returning "" and another raising means every
  caller needs a try/except that only one of them justifies.
* A backend that raises. A provider that lets it through has made its caller
  responsible for a failure mode it cannot name.
* A backend that returns None. Passing None upward as if it were text is how
  a reply becomes the string "None".
* JSON that is not JSON. A dict is the contract; a string that looks like one
  is not.

Run with a stubbed backend, so no model is loaded and nothing touches the
resident weights. A provider that cannot be exercised without loading a model
is itself the finding, and ``what_cannot_be_checked_offline`` names those.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("Aura.WhatAProviderPromises")

__all__ = [
    "THE_PROMISES",
    "what_cannot_be_checked_offline",
    "what_this_provider_promises",
]

#: Every edge, in the words a failure should use. The three about failure read
#: differently for a provider that fails closed, and the difference is named
#: rather than imposed — see ``fails_closed``.
THE_PROMISES: tuple[str, ...] = (
    "an empty prompt gives a string rather than raising",
    "a backend that raises does what the provider declared",
    "a backend that returns nothing does what the provider declared",
    "generate_json gives a dict rather than a string that looks like one",
    "generate_json handles a backend that answers with prose",
    "check_health answers with a bool",
    "an unhealthy provider says so rather than raising",
)


def what_this_provider_promises(
    make: Callable[..., Any], *, called: str = "", fails_closed: bool = False
) -> dict[str, str]:
    """Run every promise against a provider built from a stubbed backend.

    ``make`` takes one argument — a callable standing in for the backend — and
    returns a provider. Every promise gets a fresh one: shared state between
    promises makes a failure ambiguous about which promise caused it.

    ``fails_closed`` says what this provider does when it cannot answer. A
    chain that has exhausted every lane raising is a deliberate choice —
    returning "" there would hand the caller an answer nobody produced. A leaf
    provider returning a string is equally deliberate. What must not happen is
    a provider doing one while its callers were written for the other, so the
    suite checks the declared behaviour instead of imposing one.
    """
    name = called or getattr(make, "__name__", "a provider")
    kept: dict[str, str] = {}

    def _try(promise: str, check: Callable[[], None]) -> None:
        try:
            check()
            kept[promise] = "kept"
        except AssertionError as exc:
            kept[promise] = f"broken: {exc}"
        except Exception as exc:  # noqa: BLE001 — a raise is a broken promise
            kept[promise] = f"broken: {exc!r}"

    def _empty_prompt() -> None:
        provider = make(lambda *a, **k: "something")
        out = provider.generate_text("")
        assert isinstance(out, str), f"got {type(out).__name__}"

    def _as_declared(run: Callable[[], Any], what: str) -> None:
        """Either it answers in the right shape, or it fails closed. Not both."""
        try:
            out = run()
        except Exception as exc:  # noqa: BLE001 — raising is one of the answers
            assert fails_closed, (
                f"{what} raised {type(exc).__name__} and this provider does "
                "not declare that it fails closed"
            )
            return
        assert not fails_closed, (
            f"{what} returned {out!r} and this provider declares that it "
            "fails closed rather than inventing an answer"
        )
        assert isinstance(out, (str, dict)), f"got {type(out).__name__}"
        if isinstance(out, str):
            assert out.strip() != "None", "None reached the caller as text"

    def _backend_raises() -> None:
        def angry(*a: Any, **k: Any) -> Any:
            raise RuntimeError("the lane is down")

        _as_declared(lambda: make(angry).generate_text("hello"), "generate_text")

    def _backend_returns_nothing() -> None:
        _as_declared(
            lambda: make(lambda *a, **k: None).generate_text("hello"),
            "generate_text",
        )

    def _json_is_a_dict() -> None:
        out = make(lambda *a, **k: '{"a": 1}').generate_json("hello", {})
        assert isinstance(out, dict), f"got {type(out).__name__}"

    def _json_from_prose() -> None:
        _as_declared(
            lambda: make(
                lambda *a, **k: "I think the answer is four."
            ).generate_json("hello", {}),
            "generate_json",
        )

    def _health_is_a_bool() -> None:
        out = make(lambda *a, **k: "something").check_health()
        assert isinstance(out, bool), f"got {type(out).__name__}"

    def _unhealthy_says_so() -> None:
        def angry(*a: Any, **k: Any) -> Any:
            raise RuntimeError("nothing loaded")

        assert make(angry).check_health() in (True, False)

    for promise, check in zip(
        THE_PROMISES,
        (
            _empty_prompt,
            _backend_raises,
            _backend_returns_nothing,
            _json_is_a_dict,
            _json_from_prose,
            _health_is_a_bool,
            _unhealthy_says_so,
        ),
        strict=True,
    ):
        _try(promise, check)
    logger.debug(
        "%s kept %d of %d",
        name, sum(1 for one in kept.values() if one == "kept"), len(THE_PROMISES),
    )
    return kept


def what_cannot_be_checked_offline() -> list[str]:
    """Providers that cannot be built without loading a model.

    Named rather than skipped. A provider only exercisable against the
    resident weights has no conformance evidence at all, and "we did not test
    it" is a different sentence from "it passed".
    """
    return [
        "core/brain/llm/nucleus_manager.py:NucleusManager — its constructor "
        "resolves model paths and its generate path goes through the MLX "
        "worker; there is no seam to stand a backend in front of",
    ]

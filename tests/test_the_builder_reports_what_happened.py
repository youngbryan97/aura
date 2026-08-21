"""A build that produces nothing has to say which generator failed.

LIVE, 2026-08-21. build_app ran three iterations, generated no code in any of
them, and the only visible trace was the reply: "The construction process
didn't generate any code." The local code model's failure was logged at
debug, so nothing said which generator had failed or why.
"""

from __future__ import annotations

from pathlib import Path

BUILDER = Path(__file__).resolve().parents[1] / "core" / "capabilities" / "self_taught_builder.py"


def _generate_body() -> str:
    source = BUILDER.read_text(encoding="utf-8")
    body = source[source.index("async def _generate(") :]
    return body[: body.index("\nasync def ", 10)]


def test_each_generator_reports_what_it_returned() -> None:
    body = _generate_body()
    assert "local code model returned %d chars" in body
    assert "fallback generator returned %d chars" in body


def test_a_generator_failure_is_visible() -> None:
    """At debug it was invisible, so three failed iterations said nothing."""
    body = _generate_body()
    assert "logger.warning" in body
    assert "logger.debug" not in body


def test_a_policy_error_reaches_the_fallback() -> None:
    """ValueError was not caught, so an out-of-policy budget killed the build
    instead of falling through to the other generator."""
    body = _generate_body()
    assert "ImportError, RuntimeError, OSError, ValueError" in body


def test_an_empty_generation_is_not_returned_as_code() -> None:
    body = _generate_body()
    assert "if generated.strip():" in body


def test_the_repair_hint_carries_no_advice_from_another_domain() -> None:
    """Three sentences of checkers advice were appended to every failure in
    every domain, including a sitting timer."""
    source = BUILDER.read_text(encoding="utf-8")
    loop = source[source.index("Functional test FAILED") :]
    loop = loop[: loop.index("research +=")]
    assert "data-row" not in loop
    assert "closest(" not in loop
    assert "console_errors" in loop

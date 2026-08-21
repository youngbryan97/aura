"""A skill whose default made it impossible to run.

LIVE, 2026-08-21. build_app finally reached execution — past routing, past
the ceiling, past the metabolic throttle, past the permission gate — and
failed in 2.7 seconds:

    Skill error: ValueError: local_code_model_max_tokens_out_of_policy

Its `max_tokens` field defaulted to 9000 and the code lane's policy ceiling
is 2048, so every call it could ever make was refused. The number was never
going to work, and nothing checked the two against each other.
"""

from __future__ import annotations

from core.brain.llm.local_code_model import max_code_tokens
from core.skills.build_app import BuildAppInput


def test_the_lane_publishes_its_ceiling() -> None:
    """Published so callers ask instead of guessing."""
    ceiling = max_code_tokens()
    assert isinstance(ceiling, int)
    assert ceiling > 0


def test_the_default_is_not_a_number_the_lane_refuses() -> None:
    assert BuildAppInput(spec="a timer page").max_tokens == 0


def test_the_build_no_longer_spends_a_code_lane_at_all() -> None:
    """The clamp this file was written for is gone with the generator it fed.

    The runtime compiles the app from a typed plan, so the only model call is
    the plan itself and it carries its own small budget.
    """
    from pathlib import Path

    source = Path("core/skills/build_app.py").read_text(encoding="utf-8")
    assert "max_code_tokens" not in source
    assert "min(requested_tokens, ceiling)" not in source
    from core.construction.build_app_system import _PLAN_TOKENS

    assert 0 < _PLAN_TOKENS <= 2048


def test_the_policy_still_refuses_what_is_genuinely_out_of_bounds() -> None:
    """The bound is untouched: only the caller stopped guessing."""
    from core.brain.llm.local_code_model import _bounded_int

    ceiling = max_code_tokens()
    assert _bounded_int(ceiling, name="max_tokens", minimum=1, maximum=ceiling) == ceiling
    try:
        _bounded_int(ceiling + 1, name="max_tokens", minimum=1, maximum=ceiling)
    except ValueError as exc:
        assert "out_of_policy" in str(exc)
    else:  # pragma: no cover - the bound would be gone
        raise AssertionError("the policy no longer refuses anything")


def test_a_failed_build_says_why() -> None:
    """It ran for seventy-three seconds, failed, and came back as
    "build_app reported failure without a cause"."""
    import asyncio

    from core.construction.build_app_system import build_app

    async def nothing_usable(_text: str) -> str:
        return "I would be happy to help you build that!"

    result = asyncio.run(
        build_app("a tally counter", out_dir="/tmp/never_written", propose=nothing_usable)
    )
    assert not result.ok
    assert result.problems and all(problem.strip() for problem in result.problems)
    assert "Could not build" in result.summary()


def test_a_failed_build_never_claims_a_path() -> None:
    """The success summary names a file, which is a completion claim; a
    failure must not borrow it."""
    import asyncio
    from pathlib import Path

    from core.construction.build_app_system import build_app

    async def nothing_usable(_text: str) -> str:
        return ""

    target = "/tmp/build_app_failure_check"
    result = asyncio.run(build_app("a tally counter", out_dir=target, propose=nothing_usable))
    assert not result.ok
    assert result.path == ""
    assert "Built" not in result.summary()
    assert not list(Path(target).glob("*.html")) if Path(target).is_dir() else True


def test_a_model_invented_home_directory_cannot_escape() -> None:
    """LIVE: PermissionError: [Errno 13] Permission denied: '/Users/user'.

    The model filled out_dir with a home directory that does not exist on
    this machine and the path was used as given.
    """
    from pathlib import Path

    from core.runtime.payload_values import payload_path

    root = Path("/tmp/live_apps_root")
    for invented in ("/Users/user/Desktop", "../../etc", "None", ""):
        resolved = payload_path({"out_dir": invented}, "out_dir", root=root, default=root)
        assert str(resolved).startswith(str(root))


def test_a_named_subdirectory_still_works() -> None:
    from pathlib import Path

    from core.runtime.payload_values import payload_path

    # resolve(): /tmp is a symlink to /private/tmp on macOS, and payload_path
    # resolves what it returns.
    root = Path("/tmp/live_apps_root").resolve()
    assert payload_path({"out_dir": "timers"}, "out_dir", root=root, default=root) == (
        root / "timers"
    )


def test_the_default_is_the_runtime_s_own_place() -> None:
    """Naming a relative default here made it nest under itself."""
    assert BuildAppInput(spec="x").out_dir == ""

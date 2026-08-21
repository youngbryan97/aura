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


def test_a_request_above_the_ceiling_is_clamped_not_refused() -> None:
    from pathlib import Path

    source = Path("core/skills/build_app.py").read_text(encoding="utf-8")
    assert "max_code_tokens()" in source
    assert "min(requested_tokens, ceiling)" in source
    # The old default appears only where the defect is recorded, never as a
    # value the skill would send.
    field = source[source.index("max_tokens: int = Field(") :]
    assert field[: field.index(")")].startswith("max_tokens: int = Field(0")


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
    "build_app reported failure without a cause" — while the result object
    carried both an error and a status."""
    from pathlib import Path

    source = Path("core/skills/build_app.py").read_text(encoding="utf-8")
    assert '"error": reason' in source
    assert "str(result.error or \"\").strip()" in source
    assert "str(result.status or \"\").strip()" in source


def test_a_failed_build_never_claims_a_path() -> None:
    """The success summary reads "Built 'x' -> path", which is a completion
    claim; a failure must not borrow it."""
    from pathlib import Path

    source = Path("core/skills/build_app.py").read_text(encoding="utf-8")
    failure = source[source.index("if not result.ok:") :]
    failure = failure[: failure.index("return {\n            \"ok\": True")]
    assert "Could not build" in failure
    assert "Built '" not in failure


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

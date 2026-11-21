"""The environment a check of her change runs in: a developer's at a shell, not her runtime's (Master Todo F16).

Her runtime sets variables on itself that say what it is: ``AURA_GOVERNANCE_MODE=production`` (aura_main.py), that it
was launched from her app, its runtime singleton, its live state root. Every check of a change of hers (the tests that
touch it, smoke and the other gates, the outside check, her real calls, a fault she reproduces) ran as a child of her
runtime with all of them. Under production governance 20 tests of tests/test_server_runtime_hardening.py and
tests/test_resource_observation_hermeticity.py fail that pass at a shell, and to 20 November 2026 her own yes refused
her changes on them: none of hers reached origin after 17 November, five made since among them.
"""
from __future__ import annotations

import os
import tempfile

__all__ = ["a_checks_environment"]


def a_checks_environment(*, log_prefix: str = "her-checks-logs-", log_dir: str | None = None,
                         **extra: str) -> dict[str, str]:
    """This process's environment without her runtime's own variables (every ``AURA_*``), with a log folder of the
    check's own, numeric threads capped at one, no bytecode written, and ``extra`` on top."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("AURA_")}
    env.update({"AURA_LOG_DIR": log_dir or tempfile.mkdtemp(prefix=log_prefix), "VECLIB_MAXIMUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(extra)
    return env

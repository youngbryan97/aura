"""One knob, one meaning — for whether Aura can see.

``AURA_ENABLE_PROACTIVE_VISION`` was read in four places with three different
defaults. Unset — the normal case — ``continuous_vision`` and the orchestrator
believed ambient vision was ON while ``continuous_perception`` and
``pulse_manager`` believed it was OFF. Nothing failed loudly; "does Aura watch
the screen" simply had no single answer, and the answer you got depended on
which half of the system you asked.

These tests pin the resolved policy and, more importantly, pin that every
reader goes through it. The second test is the one that matters: it fails if
somebody adds a fifth ``os.getenv("AURA_ENABLE_PROACTIVE_VISION", ...)`` and
re-opens the divergence.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from core.senses.vision_policy import proactive_vision_enabled, vision_policy_reason

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The policy module is allowed to read the variable. Nothing else is.
_POLICY_MODULE = "core/senses/vision_policy.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AURA_ENABLE_PROACTIVE_VISION", raising=False)
    monkeypatch.delenv("AURA_HEADLESS", raising=False)


def test_ambient_vision_is_on_when_nobody_said_otherwise():
    """The resolved default. Two of the four old readers had this backwards."""
    assert proactive_vision_enabled() is True
    assert vision_policy_reason() == ""


def test_an_explicit_zero_still_turns_it_off():
    """Default-on must not mean unturnoffable."""
    import os

    os.environ["AURA_ENABLE_PROACTIVE_VISION"] = "0"
    try:
        assert proactive_vision_enabled() is False
        assert vision_policy_reason() == "operator_disabled"
    finally:
        os.environ.pop("AURA_ENABLE_PROACTIVE_VISION", None)


def test_headless_refuses_regardless_of_the_flag():
    """There is no screen to read; the flag cannot conjure one."""
    import os

    os.environ["AURA_HEADLESS"] = "1"
    os.environ["AURA_ENABLE_PROACTIVE_VISION"] = "1"
    try:
        assert proactive_vision_enabled() is False
        assert vision_policy_reason() == "headless"
    finally:
        os.environ.pop("AURA_HEADLESS", None)
        os.environ.pop("AURA_ENABLE_PROACTIVE_VISION", None)


def test_headless_and_disabled_are_distinguishable():
    """A missing screen and a deliberate setting are different facts.

    They used to log the same sentence, so an operator debugging "why can't she
    see" could not tell which condition had refused.
    """
    import os

    os.environ["AURA_HEADLESS"] = "1"
    try:
        headless = vision_policy_reason()
    finally:
        os.environ.pop("AURA_HEADLESS", None)
    os.environ["AURA_ENABLE_PROACTIVE_VISION"] = "0"
    try:
        disabled = vision_policy_reason()
    finally:
        os.environ.pop("AURA_ENABLE_PROACTIVE_VISION", None)
    assert headless != disabled
    assert headless and disabled


def test_no_module_reads_the_raw_flag_behind_the_policy():
    """The ratchet. A fifth reader re-opens the divergence this closed."""
    tracked = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "core/*.py", "interface/*.py", "skills/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    # Only actual environment READS count. A comment or a log line naming the
    # variable is how an operator learns what to set, and banning the string
    # outright would make the code less legible to buy nothing.
    pattern = re.compile(
        r"""(?:os\.environ\.get|os\.getenv|os\.environ\[|_env_flag)\s*\(\s*"""
        r"""["']AURA_ENABLE_PROACTIVE_VISION["']"""
    )
    offenders = []
    for rel in tracked:
        if rel == _POLICY_MODULE:
            continue
        path = REPO_ROOT / rel
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(body):
            offenders.append(rel)

    assert not offenders, (
        "these modules read AURA_ENABLE_PROACTIVE_VISION directly instead of "
        f"core.senses.vision_policy: {offenders}. That is how the flag came to "
        "have three different defaults."
    )

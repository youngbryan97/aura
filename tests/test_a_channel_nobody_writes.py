"""A publisher with no caller must be reported, and a wired one must not be.

The defect this guards against is invisible from inside a test suite: calling
the publisher directly is what a unit test does, so a green unit test says
nothing about whether the runtime ever calls it. The check therefore runs over
a small tree of its own where the wiring is the thing under test.
"""

from __future__ import annotations

import pathlib

from core.verify.is_this_channel_ever_written import channels_nobody_writes

_CHANNELS = '''
from core.fsw.telemetry_dictionary import channel, write

CHANNEL_STAMINA = "sociability.stamina"


def declare():
    channel(0x0101, name=CHANNEL_STAMINA, unit="fraction")


def sample():
    write(CHANNEL_STAMINA, 0.5)
'''

_BOOT_WITHOUT_THE_PUBLISHER = '''
from core.made_up import declare


def boot():
    declare()
'''

_BOOT_WITH_THE_PUBLISHER = '''
from core.made_up import declare, sample


def boot():
    declare()


def cadence():
    sample()
'''


def _tree(root: pathlib.Path, boot: str) -> pathlib.Path:
    package = root / "core"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "made_up.py").write_text(_CHANNELS)
    (package / "runtime_boot.py").write_text(boot)
    return root


def test_a_publisher_with_no_caller_is_reported(tmp_path: pathlib.Path) -> None:
    found = channels_nobody_writes(str(_tree(tmp_path, _BOOT_WITHOUT_THE_PUBLISHER)))
    assert [f.channel for f in found] == ["sociability.stamina"]
    assert found[0].writers == ("core.made_up.sample",)


def test_a_publisher_the_runtime_calls_is_not_reported(tmp_path: pathlib.Path) -> None:
    found = channels_nobody_writes(str(_tree(tmp_path, _BOOT_WITH_THE_PUBLISHER)))
    assert found == ()


def test_the_live_tree_has_no_channel_with_an_unreachable_writer() -> None:
    """The two found on 2026-09-10 are wired; nothing may join them."""
    root = pathlib.Path(__file__).resolve().parents[1]
    found = channels_nobody_writes(str(root))
    assert [f.channel for f in found] == [], (
        "a declared channel's only writer has no production caller: "
        + "; ".join(f"{f.channel} <- {', '.join(f.writers)}" for f in found)
    )

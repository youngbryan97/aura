"""Every lane named in LANE_MODULES is a module that exists.

The list carried `interface.routes.chat_desktop_evidence` after the module
was gone. The patch helper imports every name in it, so a single stale line
raised ModuleNotFoundError inside the helper and failed every test that used
it — 146 of them — with an error naming a deleted file rather than the list
that still mentioned it.

The list's own comment warns about the omission and not the leftover. This
is the other direction.
"""

from __future__ import annotations

import importlib

import pytest

from tests.chat_lane_support import LANE_MODULES


@pytest.mark.parametrize("name", LANE_MODULES)
def test_the_lane_is_a_module_that_imports(name):
    assert importlib.import_module(name) is not None


def test_no_lane_is_named_twice():
    """A duplicate is patched twice, and the second patch sees the first."""
    assert len(LANE_MODULES) == len(set(LANE_MODULES))


def test_every_chat_lane_module_on_disk_is_in_the_list():
    """The direction the comment already warns about, held rather than hoped.

    A lane missing here is a lane a patch will miss, which is a test that
    passes because the code it meant to exercise was never reached.
    """
    import pathlib

    on_disk = {
        f"interface.routes.{one.stem}"
        for one in pathlib.Path("interface/routes").glob("chat*.py")
        if one.stem != "__init__"
    }
    missing = sorted(on_disk - set(LANE_MODULES))
    assert not missing, f"chat lane modules not in LANE_MODULES: {missing}"

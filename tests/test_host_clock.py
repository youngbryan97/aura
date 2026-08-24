from __future__ import annotations

import inspect
from datetime import UTC, datetime

from core.runtime.host_clock import read_host_clock_text


def test_host_clock_is_bounded_and_includes_wall_time() -> None:
    local = datetime(2026, 8, 15, 20, 6, 31, tzinfo=UTC).astimezone()
    text = read_host_clock_text(now=local)

    assert "Aug 15 2026" in text
    assert local.strftime("%I:%M:%S %p") in text
    assert str(local.tzname()) in text
    assert len(text) <= 240


def test_host_clock_has_no_desktop_or_objc_dependency() -> None:
    source = inspect.getsource(read_host_clock_text)

    assert "Foundation" not in source
    assert "AppleScript" not in source
    assert "ComputerUseSkill" not in source

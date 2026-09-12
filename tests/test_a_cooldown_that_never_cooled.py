"""A cooldown whose writer and reader were in different modules.

`_enter_recovery_cooldown` stamped `_last_recovery_cooldown_at` in the chat
lane module. `_in_recovery_cooldown` read a variable of the same name in
chat.py. Two modules, two globals, one name — so the stamp never reached the
check, and `_protected_foreground_reason` never once returned
"recovery_cooldown" for a lane that had just entered one.

Nothing failed. The cooldown simply did not happen, on every recovery, for as
long as the two halves lived apart.
"""
from __future__ import annotations

import time

from interface.routes import chat as chat_routes
from interface.routes import chat_lane_state


def test_the_stamp_and_the_check_share_one_module():
    stamp = chat_lane_state._enter_recovery_cooldown
    check = chat_lane_state._in_recovery_cooldown
    assert stamp.__module__ == check.__module__, (
        "the write and the read must resolve the same global; they did not, "
        "and the cooldown was dead for every recovery"
    )
    assert chat_routes._in_recovery_cooldown is check, (
        "chat.py must use the lane module's reader rather than define its own"
    )


def test_entering_a_cooldown_is_visible_to_the_check(monkeypatch):
    monkeypatch.setattr(chat_lane_state, "_last_recovery_cooldown_at", 0.0)
    assert chat_lane_state._in_recovery_cooldown() is False

    chat_lane_state._enter_recovery_cooldown()

    assert chat_lane_state._in_recovery_cooldown() is True


def test_a_cooldown_expires(monkeypatch):
    monkeypatch.setattr(
        chat_lane_state,
        "_last_recovery_cooldown_at",
        time.monotonic() - (chat_lane_state._RECOVERY_COOLDOWN_SECONDS + 0.5),
    )

    assert chat_lane_state._in_recovery_cooldown() is False


def test_a_recovering_lane_in_cooldown_is_protected():
    chat_lane_state._enter_recovery_cooldown()

    assert chat_routes._protected_foreground_reason({"state": "recovering"}) == (
        "recovery_cooldown"
    )

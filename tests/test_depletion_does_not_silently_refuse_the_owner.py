"""Ten thousand deferrals, and the person was never told why.

`welfare.recovery_drive > 0.6` defers every consequential action. MEASURED
live 2026-08-18: recovery_drive sat between 0.60 and 0.64 for an entire
session — just over the line — so the rule fired constantly, 10,330 times in
one log, on a runtime reporting healthy with vitality 0.73 and mood
INQUISITIVE.

At the moment an owner request was refused, the context read:

    foreground_request: True
    user_explicitly_authorized: True
    desktop_execution_contract: True
    user_visible_desktop_action: True

Everything needed to know whose request it was, present and unused. What the
person received was "the governed desktop task lane did not complete" — never
"I am depleted".

Resting instead of doing more of your OWN work is sound, and that is left
exactly as it was. Deciding on someone's behalf that they can wait, silently,
is not.
"""

from __future__ import annotations

import pytest

from core.being.runtime import _is_explicit_owner_request


class TestWhoseRequestItIs:
    @pytest.mark.parametrize(
        "context",
        [
            {"foreground_request": True},
            {"user_explicitly_authorized": True},
            {"desktop_execution_contract": True},
            {"user_visible_desktop_action": True},
            {"user_explicit_action_request": True},
        ],
    )
    def test_an_explicit_request_is_recognised(self, context):
        assert _is_explicit_owner_request(context) is True

    @pytest.mark.parametrize(
        "context",
        [
            {},
            None,
            {"source": "curiosity_loop"},
            {"source": "drive_engine", "foreground_request": False},
            {"origin": "autonomous"},
        ],
    )
    def test_autonomous_work_does_not_inherit_the_owner_s_standing(self, context):
        """A background loop that forgets a flag must not gain authority."""
        assert _is_explicit_owner_request(context) is False


class TestThePolicyItself:
    def test_an_owner_request_is_no_longer_deferred_for_recovery(self):
        import inspect

        from core.being import runtime

        source = inspect.getsource(runtime)
        gate = source.index("welfare_recovery_required_before_action")
        window = source[max(0, gate - 2600) : gate]
        assert "_is_explicit_owner_request(context)" in window

    def test_depletion_is_still_recorded_so_she_can_say_it(self):
        """The state stays visible; it just stops deciding for the person."""
        import inspect

        from core.being import runtime

        source = inspect.getsource(runtime)
        assert "owner_request_proceeds_while_depleted" in source
        assert "welfare_recovery_drive=" in source

    def test_autonomous_consequential_work_still_defers(self):
        """The rule that protects her from her own workload is unchanged."""
        import inspect

        from core.being import runtime

        source = inspect.getsource(runtime)
        assert 'defers.append("welfare_recovery_required_before_action")' in source

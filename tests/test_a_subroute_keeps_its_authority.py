"""A labelled sub-route lost the standing of the route it belongs to.

`coerce_authority_origin` fell back to a hardcoded nine-token list — user,
api, voice, admin, gui, websocket, ws, direct, external — which omits
`desktop`, `chat`, `ui`, `frontend` and most of the set it was standing in for.

So `desktop_task.web_search` coerced to itself, matched no grant, and
`context_has_user_authority` returned False — for a route the desktop lane uses
on ordinary foreground turns. `sovereign_browser.pursue` was refused for the
same reason, with "denied_by_default: network_call requires validated scoped
authority".

Inheritance must not become promotion: an autonomous sub-route stays
autonomous, and a label that matches nothing stays unknown.
"""

from __future__ import annotations

import pytest

from core.executive.standing_authority import (
    coerce_authority_origin,
    context_has_user_authority,
)


class TestASubRouteInheritsItsRoute:
    @pytest.mark.parametrize(
        "origin,expected",
        [
            ("desktop_task.web_search", "desktop_task"),
            ("desktop_ui.chat", "desktop_ui"),
            ("chat_api.reply", "chat_api"),
            ("voice_bridge.listen", "voice_bridge"),
        ],
    )
    def test_the_most_specific_known_route_wins(self, origin, expected):
        assert coerce_authority_origin(origin) == expected

    @pytest.mark.parametrize(
        "origin",
        ["desktop_task.web_search", "desktop_ui.chat", "chat_api.reply"],
    )
    def test_the_sub_route_keeps_user_authority(self, origin):
        assert context_has_user_authority(origin, {}) is True


class TestInheritanceIsNotPromotion:
    @pytest.mark.parametrize(
        "origin",
        [
            "autonomous_task_engine.loop",
            "background_reflection.tick",
            "autonomy.explore",
        ],
    )
    def test_an_autonomous_sub_route_stays_autonomous(self, origin):
        assert context_has_user_authority(origin, {}) is False

    @pytest.mark.parametrize("origin", ["totally_unknown_thing", "", "nonsense.route"])
    def test_an_unknown_label_is_not_invented_into_authority(self, origin):
        assert context_has_user_authority(origin, {}) is False

    def test_an_unrelated_label_is_not_reassigned(self):
        """Inheritance must not become reassignment.

        A first pass widened the token fallback to the whole known set, which
        remapped `test_autonomy` to `autonomy` and stopped it matching its own
        grant. Prefix inheritance was the missing piece; the token scan is not.
        """
        assert coerce_authority_origin("test_autonomy") == "test_autonomy"

    def test_an_explicit_denial_still_denies_a_known_route(self):
        """The context flags remain the final word."""
        assert (
            context_has_user_authority(
                "desktop_task.web_search", {"user_explicitly_authorized": False}
            )
            is False
        )

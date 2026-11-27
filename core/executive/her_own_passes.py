"""Her own passes, the ones nobody is at the keyboard for, and the standing yes Bryan gave each.

A pass of hers that works a goal she set herself, keeps a person company at the desk, or sends a reply she wrote has
no person to confirm a call. Each has a standing grant in core/executive/standing_authority.py, bounded to its own
tools and its own origin; what each grant lets through is here, with what each tool may take. Lifted out of
standing_authority.py, which passed its 2,000-line ceiling when the desk grant took its third tool (16 November 2026).
"""

from __future__ import annotations

from typing import Any

from core.executive.execution_policy import normalize_tool_name

__all__ = [
    "BRYANS_STANDING_YES",
    "DESK_COMPANY_GRANT",
    "DESK_COMPANY_ORIGIN",
    "DESK_COMPANY_TOOLS",
    "MAIL_REPLY_GRANT",
    "MAIL_REPLY_ORIGIN",
    "MAIL_REPLY_TOOLS",
    "OWN_GOAL_GRANT",
    "OWN_GOAL_ORIGIN",
    "OWN_GOAL_TOOLS",
    "PRACTICE_GAME_GRANT",
    "PRACTICE_GAME_ORIGIN",
    "PRACTICE_GAME_TOOLS",
    "STANDING_YES_MODALITIES",
    "carries_bryans_standing_yes",
    "her_passes_allow",
]

#: Her own tools for a goal she set herself (core/skills/her_draft.py), the eight
#: core.skills.catalog_policy.HER_GOAL_TOOLS names (a test holds the two equal).
#: Each acts only inside a git worktree of her source that she works the goal in, but what_went_wrong, which only
#: reads her runtime's record of its own faults, in her own process.
OWN_GOAL_TOOLS = frozenset({"check_draft", "edit_draft", "find_in_source", "look_up", "nothing_to_change", "offer_draft",
                            "read_source", "search_the_web", "what_went_wrong"})
#: What each takes, and the arguments among them that name a place in her workspace.
_OWN_GOAL_ARGUMENTS = {
    "what_went_wrong": frozenset({"subsystem"}),
    "read_source": frozenset({"path", "from_line"}),
    "find_in_source": frozenset({"pattern", "where"}),
    # What she needs to know, in her library and, where she is online, on the web (Bryan, 26 November 2026: "Internet if
    # she has it of course").
    "look_up": frozenset({"query"}),
    "search_the_web": frozenset({"query"}),
    "edit_draft": frozenset({"path", "old_text", "new_text"}),
    "check_draft": frozenset({"tests"}),
    "offer_draft": frozenset({"summary"}),
    "nothing_to_change": frozenset({"what_came_of_it"}),
}
#: The pass that works a goal of hers (core.agency.her_goal_at_work.ORIGIN).
OWN_GOAL_ORIGIN = "her_goal_background"
#: The grant that is Bryan's standing yes to that work.
OWN_GOAL_GRANT = "aura.own-goal-work"

#: Her tools at the desk (core/skills/at_the_desk.py), the core.skills.catalog_policy.DESK_COMPANY_TOOLS names (a test
#: holds the two equal), and what each takes.
DESK_COMPANY_TOOLS = frozenset({"point_at_on_screen", "show_page", "reveal_file"})
_DESK_COMPANY_ARGUMENTS = {"point_at_on_screen": frozenset({"words"}), "show_page": frozenset({"url"}),
                           "reveal_file": frozenset({"path"})}
#: Her turn at the desk while she keeps a person company (core.brain.her_unprompted_turn.DESK_ORIGIN).
DESK_COMPANY_ORIGIN = "her_desk_company"
#: The grant that is Bryan's standing yes to what she does on the screen while keeping company.
DESK_COMPANY_GRANT = "aura.desk-company"
#: Her replies to people who wrote to her (core/skills/her_replies_to_mail.py), sent by her mail round: the tool,
#: what a reply takes (the rest are the adapter's own defaults, filled in before a call is rated), the round's origin
#: when it sends, and the grant that is Bryan's standing yes to them.
MAIL_REPLY_TOOLS = frozenset({"email_adapter"})
_MAIL_REPLY_ARGUMENTS = frozenset({"mode", "uid", "body", "in_reply_to", "references", "limit", "query"})
MAIL_REPLY_ORIGIN = "her_mail_reply"
MAIL_REPLY_GRANT = "aura.own-mail-replies"
#: A reminder a person asked for, come due (core/agency/reminders_delivered.py): it goes to their desktop as a
#: notification, the tool and what it takes, the origin of the call, and the grant that lets it go out. The person who
#: asked to be reminded is the yes; nothing is sent that is not the text of a reminder of theirs now due.
REMINDER_DUE_TOOLS = frozenset({"notify_user"})
_REMINDER_DUE_ARGUMENTS = frozenset({"title", "message", "sound"})
REMINDER_DUE_ORIGIN = "her_reminder_due"
REMINDER_DUE_GRANT = "aura.reminder-due"
#: A practice game of hers against the world's other player, where her results vary (core/agency/
#: practising_between_games.py, core/agency/where_her_results_vary.py): the tool, what it takes, her practice's origin,
#: and the grant that is Bryan's standing yes to it (27 November 2026, asked and answered: "Yes, add the grant").
PRACTICE_GAME_TOOLS = frozenset({"computer_use"})
_PRACTICE_GAME_ARGUMENTS = frozenset({"action", "target"})
_PRACTICE_GAME_TARGET = frozenset({"goal", "success_when", "target_app", "max_seconds", "narrate"})
PRACTICE_GAME_ORIGIN = "her_practice"
PRACTICE_GAME_GRANT = "aura.practice-games"
#: The grants whose lease is Bryan's yes to a call nobody is there to confirm (core/capabilities/permission_model.py),
#: and for each, the modalities a call within it reaches that are off unless someone says otherwise.
BRYANS_STANDING_YES = frozenset({OWN_GOAL_GRANT, DESK_COMPANY_GRANT, MAIL_REPLY_GRANT, REMINDER_DUE_GRANT,
                                 PRACTICE_GAME_GRANT})
STANDING_YES_MODALITIES = {MAIL_REPLY_GRANT: frozenset({"email", "network_write"})}


def carries_bryans_standing_yes(origin: Any, tool_name: Any) -> bool:
    """Whether this is a call Bryan said yes to in advance, for a pass of hers nobody is at the keyboard for.

    Her goal tools from her goal pass ("She can do it without me", 9 November 2026), her desk tools from her turn at
    the desk ("full companion ... im giving you permission to work with whatever permissions she needs", 15 November
    2026), and her replies from her mail round ("she should send email responses on her own", 15 November 2026).
    Everything else about the call still applies: its lease, the Will, and the permission model's check of that
    lease.
    """
    source, name = str(origin or ""), normalize_tool_name(tool_name)
    return ((source == OWN_GOAL_ORIGIN and name in OWN_GOAL_TOOLS)
            or (source == DESK_COMPANY_ORIGIN and name in DESK_COMPANY_TOOLS)
            or (source == MAIL_REPLY_ORIGIN and name in MAIL_REPLY_TOOLS)
            or (source == REMINDER_DUE_ORIGIN and name in REMINDER_DUE_TOOLS)
            or (source == PRACTICE_GAME_ORIGIN and name in PRACTICE_GAME_TOOLS))


def her_passes_allow(policy: str, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str] | None:
    """What the argument policy of one of her own passes says of this call; None when ``policy`` is none of theirs."""
    if policy == "her_mail_reply":
        from core.skills.her_replies_to_mail import sends_without_asking

        if tool_name not in MAIL_REPLY_TOOLS:
            return False, "her_mail_reply_tool_mismatch"
        if not sends_without_asking():
            return False, "her_mail_reply_waits_for_bryan"
        if str(arguments.get("mode") or "").strip().lower() != "reply":
            return False, "her_mail_reply_is_only_a_reply"
        # Judged by what the call carries: the adapter authorizes its whole input model, every field it has
        # at None included, and a reply that names a recipient or a subject of its own is not a reply.
        if {key for key, value in arguments.items() if value not in (None, "")} - _MAIL_REPLY_ARGUMENTS:
            return False, "her_mail_reply_arguments_unsupported"
        if not str(arguments.get("uid") or "").strip() or not str(arguments.get("body") or "").strip():
            return False, "her_mail_reply_names_no_message_or_says_nothing"
        return True, "her_mail_reply"
    if policy == "her_practice_game":
        return _a_practice_game_allowed(tool_name, arguments)
    if policy == "reminder_due":
        if tool_name not in REMINDER_DUE_TOOLS:
            return False, "reminder_due_tool_mismatch"
        if {key for key, value in arguments.items() if value not in (None, "")} - _REMINDER_DUE_ARGUMENTS:
            return False, "reminder_due_arguments_unsupported"
        from core.agency.reminders import due_reminders

        said = " ".join(str(arguments.get("message") or "").split())
        if not said or not any(reminder.text == said for reminder in due_reminders()):
            return False, "reminder_due_is_no_reminder_now_due"
        return True, "reminder_due"
    if policy == "desk_company":
        if tool_name not in DESK_COMPANY_TOOLS:
            return False, "desk_company_tool_mismatch"
        if set(arguments) - _DESK_COMPANY_ARGUMENTS[tool_name]:
            return False, "desk_company_arguments_unsupported"
        named = {"point_at_on_screen": "words", "show_page": "url", "reveal_file": "path"}[tool_name]
        if not str(arguments.get(named) or "").strip():
            return False, "desk_company_names_nothing"
        if tool_name == "show_page":
            from urllib.parse import urlsplit

            page = urlsplit(str(arguments.get("url") or "").strip())
            if page.scheme.lower() not in {"http", "https"} or not page.hostname or page.username or page.password:
                return False, "desk_company_shows_only_a_web_page"
        return True, "desk_company"
    if policy == "her_goal_work":
        if tool_name not in OWN_GOAL_TOOLS:
            return False, "her_goal_tool_mismatch"
        if set(arguments) - _OWN_GOAL_ARGUMENTS[tool_name]:
            return False, "her_goal_arguments_unsupported"
        # The tools refuse a path outside her workspace themselves; a lease is not
        # issued for one that names a place by climbing or from the root.
        places = [arguments.get(key) for key in ("path", "where") if arguments.get(key)]
        places += list(arguments.get("tests") or [])
        for place in places:
            text = str(place or "")
            if text.startswith(("/", "~")) or ".." in text.replace("\\", "/").split("/"):
                return False, "her_goal_place_outside_her_workspace"
        return True, "her_goal_work"
    return None


def _a_practice_game_allowed(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    """A practice game and nothing else: a pursuit in the board application she last played, its goal the practice
    game's own, and nobody at the machine for as long as her practice waits for (practising_between_games
    .nobody_at_the_machine)."""
    import json
    import re

    if tool_name not in PRACTICE_GAME_TOOLS:
        return False, "her_practice_game_tool_mismatch"
    # Judged by what the call carries: the skill authorizes its whole input model, and a pursuit names no point on
    # the screen, so its coordinates stand at their defaults. LIVE 27 November 2026 they came as 0, not None, and the
    # first practice game under this grant was refused for carrying them.
    carried = {key for key, value in arguments.items()
               if value not in (None, "") and not (key in ("x", "y") and value == 0)}
    if carried - _PRACTICE_GAME_ARGUMENTS:
        return False, "her_practice_game_arguments_unsupported"
    if str(arguments.get("action") or "") != "pursue_on_screen":
        return False, "her_practice_game_is_only_a_game"
    try:
        target = json.loads(str(arguments.get("target") or "{}"))
    except ValueError:
        return False, "her_practice_game_target_unreadable"
    if not isinstance(target, dict) or set(target) - _PRACTICE_GAME_TARGET:
        return False, "her_practice_game_target_unsupported"
    if not re.fullmatch(r"Play a game against the other player, with the other player's strength set to \d+\.",
                        str(target.get("goal") or "")):
        return False, "her_practice_game_goal_is_not_a_practice_game"
    from core.agency.practising_between_games import _the_board_she_last_played, nobody_at_the_machine
    from core.skills.screen_pursuit_board_of_pieces import _kept_under

    board = _the_board_she_last_played()
    app = str((_kept_under(board[0]) if board is not None else {}).get("app") or "")
    if not app or str(target.get("target_app") or "") != app:
        return False, "her_practice_game_is_not_in_the_board_she_last_played"
    if not nobody_at_the_machine():
        return False, "her_practice_game_waits_while_someone_is_at_the_machine"
    return True, "her_practice_game"

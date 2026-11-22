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
    "STANDING_YES_MODALITIES",
    "carries_bryans_standing_yes",
    "her_passes_allow",
]

#: Her own tools for a goal she set herself (core/skills/her_draft.py), the six
#: core.skills.catalog_policy.HER_GOAL_TOOLS names (a test holds the two equal).
#: Each acts only inside a git worktree of her source that she works the goal in, but what_went_wrong, which only
#: reads her runtime's record of its own faults, in her own process.
OWN_GOAL_TOOLS = frozenset({"check_draft", "edit_draft", "find_in_source", "offer_draft", "read_source", "what_went_wrong"})
#: What each takes, and the arguments among them that name a place in her workspace.
_OWN_GOAL_ARGUMENTS = {
    "what_went_wrong": frozenset({"subsystem"}),
    "read_source": frozenset({"path", "from_line"}),
    "find_in_source": frozenset({"pattern", "where"}),
    "edit_draft": frozenset({"path", "old_text", "new_text"}),
    "check_draft": frozenset({"tests"}),
    "offer_draft": frozenset({"summary"}),
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
#: The grants whose lease is Bryan's yes to a call nobody is there to confirm (core/capabilities/permission_model.py),
#: and for each, the modalities a call within it reaches that are off unless someone says otherwise.
BRYANS_STANDING_YES = frozenset({OWN_GOAL_GRANT, DESK_COMPANY_GRANT, MAIL_REPLY_GRANT})
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
            or (source == MAIL_REPLY_ORIGIN and name in MAIL_REPLY_TOOLS))


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

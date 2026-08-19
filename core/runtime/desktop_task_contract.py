"""Shared live desktop task contract.

The chat planner, response-generation prompt, and desktop_task executor all
consume this list. Keeping it in one lightweight runtime module prevents the
live UI from advertising a narrower or stale action surface than the executor
can actually govern and verify.
"""

from __future__ import annotations

from typing import Any

DESKTOP_TASK_ALLOWED_ACTIONS: tuple[str, ...] = (
    "click",
    "type",
    "hotkey",
    "scroll",
    "inspect_screen",
    "read_screen_text",
    "read_menu_clock",
    "open_app",
    "open_url",
    "run_command",
    "set_clipboard",
    "get_clipboard",
    "wait",
    "run_applescript",
    "write_text_file",
    "render_text_pdf",
    "move_file",
    "create_folder",
    # Reading a directory she is permitted to read. The write actions above had
    # no read counterpart, so a question about the contents of a real path
    # could only be answered by generation: asked to count the .py files in a
    # directory holding 9, she answered 3, named three files that do not
    # exist, and reported writing a file that was never created. Nothing had
    # looked, because nothing could.
    "list_directory",
    "fetch_topic_image",
    "system_control",
    # The main window and companion bubble are one Aura surface. This action
    # lets her reposition the native bubble through the same governed desktop
    # plan used for every other visible effect.
    "move_aura_bubble",
    # Native app scripting, preferred over keystrokes wherever an app has a
    # scripting interface. Typing into a window depends on focus surviving
    # from one step to the next, and on a real desktop it does not: live
    # 2026-07-28 "open Notes and write a note" repeatedly lost the front to
    # Chrome between cmd+n and cmd+v.
    #
    # write_in_app is the general form: name any application, and how it
    # takes text is derived from the scripting dictionary the app itself
    # publishes (core/perception/app_dictionary.py). Notes answers
    # note.body, TextEdit document.text, Reminders reminder.body — none of
    # which is written down anywhere. An app with no dictionary falls back
    # to typing, which is what a person would have to do too.
    "write_in_app",
    # The Notes-shaped name, kept because plans and receipts already use it.
    # It resolves through write_in_app.
    "create_note",
    # A goal that has to be watched rather than performed.
    #
    # Every other action here happens once. That made any request with a
    # condition on it structurally unplannable: "keep going until", "wait for
    # it to finish and then", "step through the wizard", "play until". A
    # planner holding only one-shot verbs has to reduce such a request to its
    # first action, and then the turn reports success. Measured live,
    # 2026-08-19: "play 2048 until you get a 128" was planned as open_app and
    # answered "Done — opened Google Chrome" in under two seconds.
    #
    # This action takes a goal and the text that means it is finished, then
    # looks, decides, acts and looks again until one or the other runs out.
    "pursue_on_screen",
)

DESKTOP_TASK_RETRY_SAFE_ACTIONS: frozenset[str] = frozenset(
    {
        "create_folder",
        "get_clipboard",
        "inspect_screen",
        "move_aura_bubble",
        "open_app",
        "read_menu_clock",
        "read_screen_text",
        "wait",
        # Idempotent in practice: a retry writes the same document again
        # rather than half of one, which is the safe direction for a retry.
        "write_in_app",
        "create_note",
        # Re-running a pursuit re-reads the screen and decides again from what
        # is there now, which is what a retry should do.
        "pursue_on_screen",
    }
)


#: Actions whose effect this machine cannot take back.
#:
#: Distinct from the retry-safe set above, which answers a different
#: question: whether running a step *twice* is harmless. ``type`` is neither
#: retry-safe nor irreversible; ``create_folder`` is reversible but not
#: retry-safe. Conflating the two would let a planner treat "safe to repeat"
#: as "safe to do".
#:
#: The line drawn here is durable state outside Aura that no later step can
#: restore: a file overwritten or moved, a document written into an app, a
#: system setting changed, or an arbitrary command or script whose effect
#: this module cannot see. Actions that only read, or that a person can close
#: or delete, are not on the list.
DESKTOP_TASK_IRREVERSIBLE_ACTIONS: frozenset[str] = frozenset(
    {
        "move_file",
        "write_text_file",
        "render_text_pdf",
        "write_in_app",
        "create_note",
        "system_control",
        # Opaque by construction: the contract cannot read what a command or
        # a script will do, and "unknown" belongs on the irreversible side.
        "run_command",
        "run_applescript",
    }
)


def irreversible_actions() -> frozenset[str]:
    """The action kinds that cannot be undone.

    ``core/advanced_cognition/world_model.py`` has asked this module this
    question since it was written, inside a ``try/except ImportError`` that
    silently answered "nothing is irreversible" because the function did not
    exist. The caller's own destructive-verb fallback kept the damage
    contained, but every action whose name lacked the word "delete" — a file
    overwrite, a system setting — was scored as reversible.
    """
    return DESKTOP_TASK_IRREVERSIBLE_ACTIONS


def desktop_task_action_schema() -> str:
    return "|".join(DESKTOP_TASK_ALLOWED_ACTIONS)


def desktop_task_action_sentence() -> str:
    if not DESKTOP_TASK_ALLOWED_ACTIONS:
        return ""
    if len(DESKTOP_TASK_ALLOWED_ACTIONS) == 1:
        return DESKTOP_TASK_ALLOWED_ACTIONS[0]
    return (
        ", ".join(DESKTOP_TASK_ALLOWED_ACTIONS[:-1])
        + f", and {DESKTOP_TASK_ALLOWED_ACTIONS[-1]}"
    )


def desktop_task_planning_schema() -> dict[str, Any]:
    """Return the canonical schema sent to every live cognition entrypoint."""
    return {
        "document_body": "optional prose to type, write, or export",
        "steps": [
            {
                "action": desktop_task_action_schema(),
                "target": (
                    "string or JSON payload; use {{document_body}} for composed prose "
                    "and {{steps.1.result.path}} or {{last.result.path}} for a prior "
                    "verified step result"
                ),
                "reason": "why this step is needed",
                "expect": "observable effect evidence required after the step",
                "critical": (
                    "true by default; false only when the objective can still succeed "
                    "after this step fails"
                ),
            }
        ],
    }

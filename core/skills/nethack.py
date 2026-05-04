"""NetHack Interface Skill — Embodied Terminal Action.

This skill gives Aura the ability to interact with a running NetHack
game session by sending physical keystrokes to the terminal adapter.

Architecture:
  - Registered as 'execute_nethack_action' in the capability engine
  - The challenge daemon registers the adapter in ServiceContainer
  - When she generates [ACTION:execute_nethack_action], action_grounding.py
    dispatches to this skill via the standard tool_executor pipeline
  - tool_executor routes the result through FeedbackProcessor → affect,
    body schema, ProprioceptiveLoop
  - She sees the tool result (screen changed, what the message line says)
    in her working memory for the next cognitive cycle

Proprioceptive feedback:
  After sending a keystroke, this skill captures the before/after screen
  state and returns a structured outcome:
    - Whether the screen changed (did the action have an effect?)
    - The top message line (game feedback like "It's a wall" or "--More--")
    - The current status line (HP, Dlvl, etc.)
  
  This is what closes the perception-action loop. She doesn't just
  fire-and-forget keystrokes; she feels the result.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer
import logging
import time

logger = logging.getLogger("Skills.NetHack")


# Key mapping for special keys
SPECIAL_KEYS = {
    "ESC": '\x1b', "ESCAPE": '\x1b',
    "SPACE": ' ',
    "ENTER": '\n', "RETURN": '\n',
    "TAB": '\t',
    "UP": 'k', "DOWN": 'j', "LEFT": 'h', "RIGHT": 'l',
    "UPLEFT": 'y', "UPRIGHT": 'u', "DOWNLEFT": 'b', "DOWNRIGHT": 'n',
    "SEARCH": 's', "WAIT": '.', "REST": '.',
    "INVENTORY": 'i', "PICKUP": ',', "DROP": 'd',
    "OPEN": 'o', "CLOSE": 'c', "KICK": '\x04',  # Ctrl+D
    "EAT": 'e', "QUAFF": 'q', "READ": 'r', "ZAP": 'z', "WEAR": 'W',
    "WIELD": 'w', "THROW": 't', "FIRE": 'f',
    "PRAY": '#',  # Extended command prefix
    "LOOK": ':',
    "HELP": '?',
    "MORE": ' ',  # --More-- prompts
}


class NetHackParams(BaseModel):
    action: str = Field(
        ...,
        description=(
            "The key to send to NetHack. Can be a single character "
            "(h/j/k/l/y/u/b/n for movement, i for inventory, etc.) "
            "or a special key name (ESC, SPACE, ENTER, UP, DOWN, LEFT, RIGHT, "
            "SEARCH, WAIT, INVENTORY, PICKUP, DROP, EAT, QUAFF, READ, ZAP, "
            "WEAR, WIELD, PRAY, LOOK, HELP, MORE)."
        ),
    )


class NetHackSkill(BaseSkill):
    name = "execute_nethack_action"
    description = (
        "Send a keystroke to the active NetHack game session and receive "
        "proprioceptive feedback about the result. Returns whether the "
        "screen changed, the game's message line, and the status line."
    )
    input_model = NetHackParams
    metabolic_cost = 0  # Core: must be fast and cheap
    timeout_seconds = 5.0

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            params = NetHackParams(**params)

        adapter = ServiceContainer.get("nethack_adapter", default=None)
        if not adapter:
            return {"ok": False, "error": "No active NetHack session found."}

        action = params.action.strip()
        action_upper = action.upper()

        # Resolve the physical key
        if action_upper in SPECIAL_KEYS:
            physical_key = SPECIAL_KEYS[action_upper]
            display_name = action_upper
        elif len(action) == 1:
            physical_key = action
            display_name = action
        else:
            # Multi-char that isn't a special key — take first char
            physical_key = action[0] if action else None
            display_name = action
            if not physical_key:
                return {"ok": False, "error": f"Invalid action: '{action}'"}

        # Consult the general embodied cognition runtime if the challenge
        # registered one. This is a local environment action gate, layered
        # before the existing capability/AuthorityGateway execution chain.
        runtime = ServiceContainer.get("embodied_cognition:nethack", default=None)
        if runtime is not None and getattr(runtime, "last_frame", None) is not None:
            tags = []
            if display_name in {"q", "r", "z", "W", "w", "d", "e"}:
                tags.extend(["unknown_use", "irreversible"])
            if display_name in {"h", "j", "k", "l", "y", "u", "b", "n"}:
                tags.append("movement")
            if display_name in {"ESC", "SPACE", "ENTER", "RETURN"}:
                tags.extend(["prompt_safe", "cancel" if display_name == "ESC" else "confirm"])
            try:
                decision = runtime.approve_action(
                    display_name,
                    source="execute_nethack_action",
                    reason="LLM proposed NetHack keystroke",
                    tags=tags,
                    expected_effect="advance NetHack state",
                )
                if not decision.approved:
                    return {
                        "ok": False,
                        "error": f"Embodied action gate blocked '{display_name}': {decision.reason}",
                        "vetoes": decision.vetoes,
                        "summary": f"Action '{display_name}' was blocked by Aura's embodied action gate: {decision.reason}",
                    }
                if decision.action and decision.action != display_name:
                    display_name = decision.action
                    action_upper = display_name.upper()
                    if action_upper in SPECIAL_KEYS:
                        physical_key = SPECIAL_KEYS[action_upper]
                    elif len(display_name) == 1:
                        physical_key = display_name
                    else:
                        return {
                            "ok": False,
                            "error": f"Embodied action gate selected invalid replacement: {decision.action}",
                        }
            except Exception as gate_err:
                logger.warning("Embodied NetHack action gate unavailable: %s", gate_err)

        # ── PROPRIOCEPTIVE FEEDBACK ──
        # Capture screen BEFORE action
        screen_before = ""
        msg_before = ""
        try:
            screen_before = adapter.get_screen_text()
            lines = screen_before.split('\n')
            msg_before = lines[0].strip() if lines else ""
        except Exception:
            pass

        # Execute the action
        try:
            adapter.send_action(physical_key)
            time.sleep(0.15)  # Let terminal settle
        except Exception as e:
            return {"ok": False, "error": f"Failed to send keystroke: {e}"}

        # Capture screen AFTER action
        screen_after = ""
        msg_after = ""
        status_line = ""
        try:
            screen_after = adapter.get_screen_text()
            lines = screen_after.split('\n')
            msg_after = lines[0].strip() if lines else ""
            # Status line is typically the last 2 non-empty lines
            non_empty = [l.strip() for l in lines if l.strip()]
            if len(non_empty) >= 2:
                status_line = non_empty[-2] + " | " + non_empty[-1]
            elif non_empty:
                status_line = non_empty[-1]
        except Exception:
            pass

        # Determine if the action had an observable effect
        screen_changed = screen_before != screen_after
        msg_changed = msg_before != msg_after

        # Build structured result
        result = {
            "ok": True,
            "action_sent": display_name,
            "screen_changed": screen_changed,
            "message_changed": msg_changed,
        }

        # Include the game's feedback message if it changed
        if msg_changed and msg_after:
            result["game_message"] = msg_after[:200]
        elif not screen_changed:
            result["game_message"] = "(no change — action may have been blocked)"

        # Include status for situational awareness
        if status_line:
            result["status_line"] = status_line[:200]

        # Build human-readable summary for the cognitive loop
        if screen_changed:
            summary = f"Action '{display_name}' executed."
            if msg_after and msg_changed:
                summary += f" Game says: {msg_after[:100]}"
            result["summary"] = summary
        else:
            result["summary"] = (
                f"Action '{display_name}' had NO visible effect. "
                "The screen is unchanged — you may be hitting a wall, "
                "stuck in a menu, or the action is invalid in this context."
            )

        # Report the actual environmental outcome back to the runtime
        if runtime is not None:
            try:
                runtime.record_environmental_outcome(
                    display_name,
                    success=screen_changed,
                    message=result["summary"]
                )
            except Exception as e:
                logger.warning("Failed to record environmental outcome: %s", e)

        return result

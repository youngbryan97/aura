"""NetHack command compiler from generic ActionIntent to terminal-grid steps."""
from __future__ import annotations

from core.environment.command import ActionIntent, CommandCompiler, CommandSpec, CommandStep, command_id_for
from core.environment.modal import ModalState


class NetHackCommandCompiler(CommandCompiler):
    direction_keys = {
        "west": "h",
        "south": "j",
        "north": "k",
        "east": "l",
        "northwest": "y",
        "northeast": "u",
        "southwest": "b",
        "southeast": "n",
        "down": ">",
        "up": "<",
    }

    def __init__(self) -> None:
        super().__init__("terminal_grid:nethack")
        for name in {
            "move",
            "wait",
            "observe",
            "inspect",
            "inventory",
            "resolve_modal",
            "eat",
            "use",
            "pray",
            "navigate_to",
            "retreat",
            "stabilize",
            "explore_frontier",
            "recover_from_loop",
            "backtrack",
            "diagnose",
        }:
            self.register(name, self._compile)

    def _compile(self, intent: ActionIntent) -> CommandSpec:
        steps: list[CommandStep]
        expected_modal = None
        preconditions = ["terminal_alive", "command_spec_only"]
        if intent.name == "move":
            direction = str(intent.parameters.get("direction", "west")).lower()
            key = self.direction_keys.get(direction)
            if key is None:
                raise ValueError(f"unknown_direction:{direction}")
            steps = [CommandStep("key", key)]
        elif intent.name in {"wait", "observe"}:
            steps = [CommandStep("wait" if intent.name == "observe" else "key", "." if intent.name == "wait" else "observe")]
        elif intent.name in {"inspect", "diagnose"}:
            steps = [CommandStep("key", ":")]
        elif intent.name == "inventory":
            steps = [CommandStep("key", "i")]
            expected_modal = ModalState(kind="menu", text="inventory", legal_responses={"\x1b", " "}, safe_default="\x1b")
        elif intent.name == "resolve_modal":
            response = str(intent.parameters.get("response") or " ")
            steps = [CommandStep("key", response)]
        elif intent.name in {"eat", "stabilize"}:
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "e")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select food", legal_responses={"\x1b"}, safe_default="\x1b")
        elif intent.name == "pray":
            steps = [CommandStep("key", "#"), CommandStep("text", "pray\n")]
        elif intent.name == "navigate_to":
            direction = str(intent.parameters.get("direction", "down")).lower()
            steps = [CommandStep("key", self.direction_keys.get(direction, ">"))]
        elif intent.name in {"retreat", "backtrack"}:
            direction = str(intent.parameters.get("direction", "west")).lower()
            steps = [CommandStep("key", self.direction_keys.get(direction, "h"))]
        elif intent.name in {"explore_frontier", "recover_from_loop"}:
            steps = [CommandStep("key", "s" if intent.name == "explore_frontier" else "\x1b")]
        else:
            raise ValueError(f"unknown_intent:{intent.name}")
        command = CommandSpec(
            command_id=command_id_for(self.environment_id, intent),
            environment_id=self.environment_id,
            intent=intent,
            preconditions=preconditions,
            steps=steps,
            expected_effects=[intent.expected_effect or f"{intent.name}_attempted"],
            expected_modal=expected_modal,
        )
        command.validate()
        return command


__all__ = ["NetHackCommandCompiler"]

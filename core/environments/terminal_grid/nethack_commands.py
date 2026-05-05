"""NetHack command compiler from generic ActionIntent to terminal-grid steps.

Registers a comprehensive set of semantic action intents that cover early-game
through endgame play.  Each intent maps to one or more terminal keystrokes with
appropriate risk levels, expected modal states, and preconditions.
"""
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
        for name in self._all_intents():
            self.register(name, self._compile)

    @staticmethod
    def _all_intents() -> set[str]:
        return {
            # Original intents
            "move", "wait", "observe", "inspect", "inventory",
            "resolve_modal", "eat", "use", "pray",
            "navigate_to", "retreat", "stabilize",
            "explore_frontier", "recover_from_loop", "backtrack", "diagnose",
            # Extended intents
            "pickup", "drop", "wield", "wear", "take_off",
            "quaff", "read", "zap", "apply", "throw",
            "kick", "open_door", "close_door", "search",
            "far_look", "name_item", "call_type", "pay",
            "offer", "loot", "use_stairs_down", "use_stairs_up",
        }

    def _compile(self, intent: ActionIntent) -> CommandSpec:
        steps: list[CommandStep]
        expected_modal = None
        preconditions = ["terminal_alive", "command_spec_only"]

        # ----- Original intents -----
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

        # ----- Extended intents -----
        elif intent.name == "pickup":
            steps = [CommandStep("key", ",")]

        elif intent.name == "drop":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "d")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select item to drop", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "wield":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "w")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select weapon", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "wear":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "W")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select armor", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "take_off":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "T")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select item to remove", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "quaff":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "q")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select potion", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "read":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "r")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select scroll or spellbook", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "zap":
            item = intent.parameters.get("item_letter")
            direction = str(intent.parameters.get("direction", "")).lower()
            steps = [CommandStep("key", "z")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            if direction:
                dir_key = self.direction_keys.get(direction)
                if dir_key:
                    steps.append(CommandStep("key", dir_key))
                else:
                    expected_modal = ModalState(kind="direction_selection", text="select direction", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")
            else:
                expected_modal = ModalState(kind="direction_selection", text="select direction", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")

        elif intent.name == "apply":
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "a")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            else:
                expected_modal = ModalState(kind="item_selection", text="select tool", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "throw":
            item = intent.parameters.get("item_letter")
            direction = str(intent.parameters.get("direction", "")).lower()
            steps = [CommandStep("key", "t")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))
            if direction:
                dir_key = self.direction_keys.get(direction)
                if dir_key:
                    steps.append(CommandStep("key", dir_key))

        elif intent.name == "kick":
            direction = str(intent.parameters.get("direction", "")).lower()
            steps = [CommandStep("key", "\x04")]  # ctrl+d
            if direction:
                dir_key = self.direction_keys.get(direction)
                if dir_key:
                    steps.append(CommandStep("key", dir_key))
                else:
                    expected_modal = ModalState(kind="direction_selection", text="select direction to kick", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")
            else:
                expected_modal = ModalState(kind="direction_selection", text="select direction to kick", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")

        elif intent.name == "open_door":
            direction = str(intent.parameters.get("direction", "")).lower()
            steps = [CommandStep("key", "o")]
            if direction:
                dir_key = self.direction_keys.get(direction)
                if dir_key:
                    steps.append(CommandStep("key", dir_key))
            else:
                expected_modal = ModalState(kind="direction_selection", text="select direction to open", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")

        elif intent.name == "close_door":
            direction = str(intent.parameters.get("direction", "")).lower()
            steps = [CommandStep("key", "c")]
            if direction:
                dir_key = self.direction_keys.get(direction)
                if dir_key:
                    steps.append(CommandStep("key", dir_key))
            else:
                expected_modal = ModalState(kind="direction_selection", text="select direction to close", legal_responses=set(self.direction_keys.values()), safe_default="\x1b")

        elif intent.name == "search":
            steps = [CommandStep("key", "s")]

        elif intent.name == "far_look":
            steps = [CommandStep("key", ";")]

        elif intent.name == "name_item":
            steps = [CommandStep("key", "#"), CommandStep("text", "name\n")]
            expected_modal = ModalState(kind="prompt", text="name item", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "call_type":
            steps = [CommandStep("key", "C")]
            expected_modal = ModalState(kind="prompt", text="call type", legal_responses={"\x1b"}, safe_default="\x1b")

        elif intent.name == "pay":
            steps = [CommandStep("key", "p")]

        elif intent.name == "offer":
            steps = [CommandStep("key", "#"), CommandStep("text", "offer\n")]

        elif intent.name == "loot":
            steps = [CommandStep("key", "#"), CommandStep("text", "loot\n")]

        elif intent.name == "use_stairs_down":
            steps = [CommandStep("key", ">")]

        elif intent.name == "use_stairs_up":
            steps = [CommandStep("key", "<")]

        elif intent.name == "use":
            # Generic use — delegate to apply
            item = intent.parameters.get("item_letter")
            steps = [CommandStep("key", "a")]
            if item:
                steps.append(CommandStep("key", str(item)[0]))

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


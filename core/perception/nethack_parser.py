"""core/perception/nethack_parser.py -- Fast, zero-LLM parser for NetHack terminal output.
========================================================================================
Converts raw 24x80 ASCII text into a structured EnvironmentState.
"""
import re

from .environment_parser import EnvironmentParser, EnvironmentState

class NetHackParser(EnvironmentParser):

    def __init__(self):
        # Regex for status line 2: Dlvl:1 $:7 HP:12(12) Pw:7(7) AC:9 Xp:1/0 T:46
        # Or variations of it.
        self.status_re = re.compile(
            r"(?:Dlvl|Level):\s*(\d+).*?\$:\s*(\d+).*?HP:\s*(\d+)\((\d+)\).*?(?:Pw|Ena):\s*(\d+)\((\d+)\).*?AC:\s*(-?\d+).*?Xp:\s*(\d+).*?T:\s*(\d+)"
        )

        # Hunger states that might appear at the end of the status line
        self.hunger_states = ["Satiated", "Hungry", "Weak", "Fainting", "Fainted", "Starved"]

    def parse(self, raw_input: str) -> EnvironmentState:
        """Parses the 24x80 NetHack screen string."""
        lines = raw_input.split('\n')

        # Ensure we have at least 24 lines, pad if necessary
        while len(lines) < 24:
            lines.append("")

        state = EnvironmentState(domain="nethack", raw_reference="terminal")

        # 1. Parse Status Lines (Bottom 2 lines)
        status_line_1 = lines[-2].strip()
        status_line_2 = lines[-1].strip()

        if "[" in status_line_1 and "]" in status_line_1:
            name_class = status_line_1.split("]")[0].replace("[", "").strip()
            state.self_state["identity"] = name_class

            # Simple attribute extraction
            for attr in ["St", "Dx", "Co", "In", "Wi", "Ch"]:
                match = re.search(fr"{attr}:\s*(\d+)", status_line_1)
                if match:
                    state.self_state[attr] = int(match.group(1))

            align_match = re.search(r"(Lawful|Neutral|Chaotic)", status_line_1)
            if align_match:
                state.self_state["alignment"] = align_match.group(1)

        status_match = self.status_re.search(status_line_2)
        if status_match:
            state.self_state["dlvl"] = int(status_match.group(1))
            state.self_state["gold"] = int(status_match.group(2))
            state.self_state["hp"] = int(status_match.group(3))
            state.self_state["max_hp"] = int(status_match.group(4))
            state.self_state["pw"] = int(status_match.group(5))
            state.self_state["max_pw"] = int(status_match.group(6))
            state.self_state["ac"] = int(status_match.group(7))
            state.self_state["xp"] = int(status_match.group(8))
            state.self_state["turn"] = int(status_match.group(9))
            state.context_id = f"dlvl_{state.self_state['dlvl']}"

        # Check for hunger
        for hunger in self.hunger_states:
            if hunger in status_line_2:
                state.self_state["hunger"] = hunger
                break
        if "hunger" not in state.self_state:
            state.self_state["hunger"] = "Normal"

        status_effects = []
        for effect in ("Blind", "Conf", "Stun", "Hallu", "Burdened", "Stressed", "Strained", "Overtaxed"):
            if effect in status_line_2 or effect in status_line_1:
                status_effects.append(effect)
        if status_effects:
            state.self_state["status_effects"] = status_effects

        # 2. Parse Messages (Top line(s) usually)
        msg_line = lines[0].strip()
        if msg_line:
            state.messages.append(msg_line)
            # If the line ends with --More--, there might be more messages
            if "--More--" in msg_line:
                state.active_prompts.append("--More-- (Press SPACE to continue)")

        # Menus and questions
        prompt_markers = (
            "What do you want to",
            "Call",
            "Direction?",
            "Is this ok?",
            "[ynq]",
            "[yn]",
            "Pick an object",
            "In what direction",
        )
        if any(marker in msg_line for marker in prompt_markers):
            if not any(p.startswith("--More--") for p in state.active_prompts):
                state.active_prompts.append(msg_line)

        # 3. Parse Map/Grid (Lines 1 to 21)
        player_pos = None
        entities = []
        visible_tiles = []
        for y in range(1, 22):
            line = lines[y]
            if len(line.strip()) == 0:
                continue

            # If it's a full-screen menu, it doesn't look like a map
            if "(end)" in line or "(1 of" in line:
                state.active_prompts.append("Inventory or Menu screen active.")
                break

            for x, char in enumerate(line):
                if char not in (" ", "\x00"):
                    visible_tiles.append((y, x))
                if char == '@':
                    player_pos = (y, x)
                # Quick entity heuristics
                elif char.isalpha():
                    if char.islower():
                        entities.append({"type": "monster", "glyph": char, "pos": (y, x), "tags": ["threat"], "hostile": True})
                    elif char.isupper():
                        entities.append({"type": "large_monster", "glyph": char, "pos": (y, x), "tags": ["threat"], "hostile": True})
                elif char in ('%', '!', '?', '+', '/', '=', '"', '(', '[', '*', ')'):
                    unknown = char in ('!', '?', '/', '=', '"', '(', '[', '*', ')')
                    entities.append({"type": "item_or_feature", "glyph": char, "pos": (y, x), "unknown": unknown})
                elif char in ('<', '>'):
                    entities.append({"type": "stairs", "glyph": char, "pos": (y, x)})
                elif char == '^':
                    entities.append({"type": "trap", "glyph": char, "pos": (y, x)})

        if player_pos:
            state.spatial_info["player_pos"] = player_pos
            state.spatial_info["visible_tiles"] = visible_tiles

            # Filter entities to just those near player (radius 5) for cognitive focus
            # NetHack is partial observability, but the screen shows everything visible.
            # For the prompt, we summarize what's visible.
            nearby = []
            for e in entities:
                dy = abs(e["pos"][0] - player_pos[0])
                dx = abs(e["pos"][1] - player_pos[1])
                dist = max(dx, dy)
                e["distance"] = dist
                if dist <= 8:  # Typical vision radius in corridors, wider in rooms
                    nearby.append(e)

            state.entities = nearby
        else:
            # If player not found, might be a menu
            if not state.active_prompts and not "Inventory" in msg_line:
                 state.active_prompts.append("Menu or prompt active.")

        if any(e.get("unknown") for e in state.entities):
            state.uncertainty["visible_unknown_items"] = 0.65
        if state.active_prompts:
            state.uncertainty["modal_state"] = 0.55

        state.refresh_observation_id()
        return state

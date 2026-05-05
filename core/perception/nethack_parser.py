"""core/perception/nethack_parser.py -- Fast, zero-LLM parser for NetHack terminal output.
========================================================================================
Converts raw 24x80 ASCII text into a structured EnvironmentState.

Enhancements over baseline:
- Inventory / menu screen parsing into structured ``inventory_items``
- Glyph-to-threat lookup table (replaces constant 0.6)
- Encumbrance → ``encumbrance`` field on self_state
- Sensory reliability score (0.0 under Blind/Hallu, 1.0 normal)
- Dangerous-prompt classification for shop/attack/sacrifice prompts
"""
import re
from typing import Any, Dict, List

from .environment_parser import EnvironmentParser, EnvironmentState

# ---------------------------------------------------------------------------
# Glyph threat table – general monster danger heuristic by symbol.
# Values are rough [0,1] threat scores.  Unknown glyphs default to 0.5.
# ---------------------------------------------------------------------------
GLYPH_THREAT: Dict[str, float] = {
    # low-threat domestic / weak monsters
    "d": 0.20, "f": 0.25, "r": 0.20, "b": 0.25, "j": 0.10,
    "a": 0.30, "k": 0.30, "n": 0.30, "x": 0.20,
    # medium-threat
    "h": 0.45, "o": 0.45, "s": 0.40, "y": 0.35, "g": 0.35,
    "c": 0.30, "i": 0.35, "l": 0.40, "m": 0.50, "p": 0.30,
    "q": 0.40, "t": 0.25, "u": 0.45, "v": 0.50, "w": 0.40,
    "z": 0.45,
    # high-threat special
    "e": 0.85,  # floating eye – melee = paralysis
    "E": 0.55, "F": 0.50, "G": 0.45, "H": 0.60, "I": 0.40,
    "J": 0.55, "K": 0.50, "L": 0.70, "M": 0.65, "N": 0.60,
    "O": 0.55, "P": 0.50, "Q": 0.45, "R": 0.50, "S": 0.55,
    "T": 0.65, "U": 0.60, "V": 0.70, "W": 0.65, "X": 0.60,
    "Y": 0.55, "Z": 0.60,
    # very high-threat large monsters
    "A": 0.60, "B": 0.50, "C": 0.55,
    "D": 0.85,  # dragons
    "&": 0.90,  # demons
    ";": 0.50,  # sea monsters
    "'": 0.20,  # golem bodies
}

# ---------------------------------------------------------------------------
# Inventory line regex:  "a - a blessed +1 long sword (weapon in hand)"
# ---------------------------------------------------------------------------
_INVENTORY_RE = re.compile(
    r"^([a-zA-Z])\s+-\s+(.+)$"
)

_BUC_RE = re.compile(r"\b(blessed|uncursed|cursed)\b", re.IGNORECASE)

# Dangerous prompt fragments that should flag ``dangerous_responses``
_DANGEROUS_PROMPTS = (
    "Really attack",
    "really attack",
    "Eat it?",
    "eat it?",
    "pay?",
    "Pay?",
    "sacrifice",
    "Sacrifice",
    "destroy",
    "Destroy",
    "You are about to",
)


class NetHackParser(EnvironmentParser):

    def __init__(self) -> None:
        # Regex for status line 2: Dlvl:1 $:7 HP:12(12) Pw:7(7) AC:9 Xp:1/0 T:46
        # Or variations of it.
        self.status_re = re.compile(
            r"(?:Dlvl|Level):\s*(\d+).*?\$:\s*(\d+).*?HP:\s*(\d+)\((\d+)\).*?(?:Pw|Ena):\s*(\d+)\((\d+)\).*?AC:\s*(-?\d+).*?Xp:\s*(\d+).*?T:\s*(\d+)"
        )

        # Hunger states that might appear at the end of the status line
        self.hunger_states = ["Satiated", "Hungry", "Weak", "Fainting", "Fainted", "Starved"]

    # ------------------------------------------------------------------
    # Inventory / menu screen parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_inventory_lines(lines: List[str]) -> List[Dict[str, Any]]:
        """Extract structured inventory items from screen lines."""
        items: List[Dict[str, Any]] = []
        for raw in lines:
            stripped = raw.strip()
            m = _INVENTORY_RE.match(stripped)
            if not m:
                continue
            letter = m.group(1)
            description = m.group(2).strip()
            buc_match = _BUC_RE.search(description)
            buc = buc_match.group(1).lower() if buc_match else "unknown"
            # Determine category heuristic
            category = "unknown"
            if "weapon" in description or "sword" in description or "dagger" in description:
                category = "weapon"
            elif "armor" in description or "cloak" in description or "helm" in description:
                category = "armor"
            elif "potion" in description:
                category = "potion"
            elif "scroll" in description:
                category = "scroll"
            elif "wand" in description:
                category = "wand"
            elif "ring" in description:
                category = "ring"
            elif "amulet" in description:
                category = "amulet"
            elif "food" in description or "ration" in description or "corpse" in description:
                category = "food"
            elif "gold" in description:
                category = "gold"
            elif "tool" in description or "key" in description or "lamp" in description:
                category = "tool"
            elif "spellbook" in description:
                category = "spellbook"
            elif "gem" in description or "stone" in description:
                category = "gem"
            items.append({
                "letter": letter,
                "description": description,
                "buc": buc,
                "category": category,
                "equipped": "(being worn)" in description or "(weapon in hand)" in description or "(wielded)" in description,
            })
        return items

    # ------------------------------------------------------------------
    # Main parse
    # ------------------------------------------------------------------

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

        # --- Status effects (including encumbrance) ---
        status_effects: List[str] = []
        encumbrance = "Normal"
        for effect in ("Blind", "Conf", "Stun", "Hallu", "Burdened", "Stressed", "Strained", "Overtaxed"):
            if effect in status_line_2 or effect in status_line_1:
                status_effects.append(effect)
                if effect in ("Burdened", "Stressed", "Strained", "Overtaxed"):
                    encumbrance = effect
        if status_effects:
            state.self_state["status_effects"] = status_effects
        state.self_state["encumbrance"] = encumbrance

        # --- Sensory reliability ---
        sensory_degraded = any(e in status_effects for e in ("Blind", "Hallu"))
        state.self_state["sensory_reliability"] = 0.0 if sensory_degraded else 1.0

        # 2. Parse Messages and Prompts
        # NetHack messages are usually on line 0, but prompts can appear anywhere.
        msg_line = lines[0].strip()
        if msg_line:
            state.messages.append(msg_line)
            if "--More--" in msg_line:
                state.active_prompts.append("--More-- (Press SPACE to continue)")

        # Scan for questions and modal states across all lines
        prompt_markers = (
            "What do you want to",
            "Call a",
            "Direction?",
            "Is this ok?",
            "[ynq]",
            "[yn]",
            "Pick an object",
            "In what direction",
            "Press return",
            "press return",
        )

        # Track whether any line has a dangerous prompt
        dangerous_prompt_detected = False

        # We check all lines for these markers
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(marker in stripped for marker in prompt_markers):
                # Don't duplicate if already added via line 0 logic
                if not any(stripped in p for p in state.active_prompts):
                    state.active_prompts.append(stripped)

            # Special case for --More-- on other lines (rare but possible)
            if "--More--" in stripped and not any("--More--" in p for p in state.active_prompts):
                state.active_prompts.append("--More-- (Press SPACE to continue)")

            # Dangerous prompt classification
            if any(dp in stripped for dp in _DANGEROUS_PROMPTS):
                dangerous_prompt_detected = True
                if not any(stripped in p for p in state.active_prompts):
                    state.active_prompts.append(stripped)

        if dangerous_prompt_detected:
            state.self_state["dangerous_prompt"] = True

        # 3. Check for inventory / menu screen FIRST
        is_inventory_screen = False
        inventory_items: List[Dict[str, Any]] = []
        for y in range(1, 22):
            line_text = lines[y]
            if "(end)" in line_text or "(1 of" in line_text:
                is_inventory_screen = True
                # Parse inventory items from visible lines
                inventory_items = self._parse_inventory_lines(lines[1:y + 1])
                if not any("Inventory" in p for p in state.active_prompts):
                    state.active_prompts.append("Inventory or Menu screen active.")
                break

        state.self_state["inventory_items"] = inventory_items

        # 4. Parse Map/Grid (Lines 1 to 21) — only if NOT an inventory screen
        player_pos = None
        entities: List[Dict[str, Any]] = []
        visible_tiles: List[Any] = []

        if not is_inventory_screen:
            for y in range(1, 22):
                line = lines[y]
                if len(line.strip()) == 0:
                    continue

                for x, char in enumerate(line):
                    if char not in (" ", "\x00"):
                        visible_tiles.append((y, x))
                    if char == '@':
                        player_pos = (y, x)
                    # Entity heuristics with glyph threat table
                    elif char.isalpha():
                        threat = GLYPH_THREAT.get(char, 0.5)
                        if char.islower():
                            entities.append({
                                "type": "monster", "glyph": char, "pos": (y, x),
                                "tags": ["threat"], "hostile": True,
                                "threat_score": threat,
                            })
                        elif char.isupper():
                            entities.append({
                                "type": "large_monster", "glyph": char, "pos": (y, x),
                                "tags": ["threat"], "hostile": True,
                                "threat_score": threat,
                            })
                    elif char in ('%', '!', '?', '+', '/', '=', '"', '(', '[', '*', ')'):
                        unknown = char in ('!', '?', '/', '=', '"', '(', '[', '*', ')')
                        entities.append({"type": "item_or_feature", "glyph": char, "pos": (y, x), "unknown": unknown})
                    elif char in ('<', '>'):
                        entities.append({"type": "stairs", "glyph": char, "pos": (y, x)})
                    elif char == '^':
                        entities.append({"type": "trap", "glyph": char, "pos": (y, x)})
                    elif char == '_':
                        entities.append({"type": "altar", "glyph": char, "pos": (y, x)})
                    elif char == '{':
                        entities.append({"type": "fountain", "glyph": char, "pos": (y, x)})
                    elif char == '}':
                        entities.append({"type": "pool", "glyph": char, "pos": (y, x)})

        if player_pos:
            state.spatial_info["player_pos"] = player_pos
            state.spatial_info["visible_tiles"] = visible_tiles

            # Filter entities to just those near player (radius 8) for cognitive focus
            nearby: List[Dict[str, Any]] = []
            for e in entities:
                dy = abs(e["pos"][0] - player_pos[0])
                dx = abs(e["pos"][1] - player_pos[1])
                dist = max(dx, dy)
                e["distance"] = dist
                if dist <= 8:
                    nearby.append(e)

            state.entities = nearby
        else:
            # If player not found, might be a menu
            if not state.active_prompts and "Inventory" not in msg_line:
                state.active_prompts.append("Menu or prompt active.")

        if any(e.get("unknown") for e in state.entities):
            state.uncertainty["visible_unknown_items"] = 0.65
        if state.active_prompts:
            state.uncertainty["modal_state"] = 0.55
        if sensory_degraded:
            state.uncertainty["sensory"] = 1.0 - state.self_state["sensory_reliability"]

        state.refresh_observation_id()
        return state

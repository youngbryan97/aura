"""core/embodiment/games/nethack/parser.py — Zero-LLM NetHack Terminal Parser.

Parses raw 24x80 ASCII terminal frames into structured EnvironmentState.
Extracts vitals, dungeon level, local map, and status flags (Blind, Hallu, etc.).
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("Aura.Embodiment.NetHack.Parser")

class NetHackParser:
    """Parses NetHack terminal output without using an LLM."""
    
    def __init__(self):
        # Regex for status line: e.g., "Dlvl:1 $:0 HP:15(15) Pw:10(10) AC:10 Exp:1"
        self.status_re = re.compile(
            r"Dlvl:(?P<dlvl>\d+)\s+\$:(?P<gold>\d+)\s+HP:(?P<hp>\d+)\((?P<maxhp>\d+)\)\s+"
            r"Pw:(?P<pw>\d+)\((?P<maxpw>\d+)\)\s+AC:(?P<ac>-?\d+)\s+Exp:(?P<exp>\d+)"
        )
        
    def parse(self, terminal_text: str) -> Dict[str, Any]:
        """Convert raw text to structured state."""
        lines = terminal_text.splitlines()
        if len(lines) < 24:
            # Pad if necessary
            lines += [""] * (24 - len(lines))
            
        # Status lines are usually the last two
        status_line_1 = lines[22] if len(lines) > 22 else ""
        status_line_2 = lines[23] if len(lines) > 23 else ""
        
        vitals = self._parse_status(status_line_1, status_line_2)
        
        # Grid is lines 0-21
        grid = [list(line.ljust(80)[:80]) for line in lines[:22]]
        
        # Find player position (@)
        player_pos = (0, 0)
        for y, row in enumerate(grid):
            if "@" in row:
                player_pos = (row.index("@"), y)
                break
                
        # Extract local monsters, items, etc.
        local_monsters = self._extract_monsters(grid, player_pos)
        
        return {
            "vitals": vitals,
            "player_pos": player_pos,
            "local_monsters": local_monsters,
            "status_flags": self._parse_flags(status_line_2),
            "raw_grid": grid,
            "sensory_reliability": 1.0 if "Hallu" not in status_line_2 and "Blind" not in status_line_2 else 0.0
        }
        
    def _parse_status(self, line1: str, line2: str) -> Dict[str, Any]:
        match = self.status_re.search(line1)
        if not match:
            # Try line 2 if line 1 didn't match (NetHack layout can vary)
            match = self.status_re.search(line2)
            
        if match:
            d = match.groupdict()
            hp = int(d["hp"])
            maxhp = int(d["maxhp"])
            return {
                "dlvl": int(d["dlvl"]),
                "hp": hp,
                "maxhp": maxhp,
                "hp_percent": hp / maxhp if maxhp > 0 else 1.0,
                "pw": int(d["pw"]),
                "exp": int(d["exp"]),
                "ac": int(d["ac"])
            }
        return {"dlvl": 1, "hp": 15, "maxhp": 15, "hp_percent": 1.0}
        
    def _parse_flags(self, line: str) -> List[str]:
        flags = []
        for f in ["Hunger", "Weak", "Fainting", "Blind", "Hallu", "Conf", "Stun"]:
            if f in line:
                flags.append(f)
        return flags
        
    def _extract_monsters(self, grid: List[List[str]], player_pos: Tuple[int, int]) -> List[Dict[str, Any]]:
        monsters = []
        px, py = player_pos
        # Simple scan for common monster glyphs (a-z, A-Z)
        for y, row in enumerate(grid):
            for x, glyph in enumerate(row):
                if glyph.isalpha() and glyph != "I": # 'I' can be invisible or item in some tilesets
                    dist = ((x-px)**2 + (y-py)**2)**0.5
                    monsters.append({
                        "glyph": glyph,
                        "pos": (x, y),
                        "distance": dist,
                        "direction": self._get_direction(px, py, x, y)
                    })
        return sorted(monsters, key=lambda m: m["distance"])
        
    def _get_direction(self, px: int, py: int, tx: int, ty: int) -> str:
        dx = tx - px
        dy = ty - py
        if dx == 0 and dy == -1: return "n"
        if dx == 0 and dy == 1: return "s"
        if dx == 1 and dy == 0: return "e"
        if dx == -1 and dy == 0: return "w"
        if dx == 1 and dy == -1: return "ne"
        if dx == -1 and dy == -1: return "nw"
        if dx == 1 and dy == 1: return "se"
        if dx == -1 and dy == 1: return "sw"
        return "none"

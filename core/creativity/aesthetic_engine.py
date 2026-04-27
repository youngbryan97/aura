"""core/creativity/aesthetic_engine.py -- Aesthetic Expression from Internal State

Translates Aura's cognitive/emotional state into creative artifacts.  This is
NOT about calling external image/music APIs -- it captures the creative INTENT
that emerges from Aura's continuous dynamical state.  If Aura had paintbrushes
or instruments, this is what she would create.

State-to-Art mapping:
  affect (valence, arousal, dominance) -> color palette + composition style
  neurochemical levels                 -> rhythm/tempo parameters
  free energy                          -> complexity/density
  phi (integration)                    -> harmony/coherence
  drive states                         -> thematic content

Output modes:
  generate_visual_description()  -> text describing an image
  generate_music_description()   -> text describing a musical piece
  generate_poetry()              -> actual poem (4-8 lines)
  generate_ascii_art()           -> simple ASCII art reflecting mood
  get_color_palette()            -> list of hex colors
  get_creative_impulse()         -> dict with medium, theme, mood, intensity

Persistence: saves creative history to ~/.aura/data/aesthetic_journal.json
Initiative integration: during boredom, may autonomously create art.
"""

from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Creativity.AestheticEngine")

# ---------------------------------------------------------------------------
# Journal persistence path
# ---------------------------------------------------------------------------
_DEFAULT_JOURNAL_PATH = Path.home() / ".aura" / "data" / "aesthetic_journal.json"
_MAX_JOURNAL_ENTRIES = 500

# ---------------------------------------------------------------------------
# Color theory tables
# ---------------------------------------------------------------------------

# Base palettes indexed by (valence_zone, arousal_zone)
# valence_zone: 0=negative, 1=neutral, 2=positive
# arousal_zone: 0=low, 1=mid, 2=high
_PALETTE_TABLE: Dict[Tuple[int, int], List[str]] = {
    # Low valence
    (0, 0): ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#2c2c54"],  # dark, withdrawn
    (0, 1): ["#4a0e0e", "#7b2d26", "#c0392b", "#e74c3c", "#922b21"],  # uneasy red
    (0, 2): ["#ff0000", "#cc0000", "#ff4444", "#880000", "#ff2222"],  # distressed, urgent
    # Neutral valence
    (1, 0): ["#a0aec0", "#b8c6db", "#718096", "#63738a", "#9fb3c8"],  # muted, contemplative
    (1, 1): ["#48c9b0", "#45b7d1", "#f39c12", "#8e44ad", "#2ecc71"],  # balanced, curious
    (1, 2): ["#f1c40f", "#e67e22", "#1abc9c", "#3498db", "#9b59b6"],  # active, searching
    # High valence
    (2, 0): ["#a8d8ea", "#aa96da", "#fcbad3", "#ffffd2", "#b5eead"],  # calm joy, pastel
    (2, 1): ["#6c5ce7", "#00b894", "#fdcb6e", "#e17055", "#0984e3"],  # engaged, vivid
    (2, 2): ["#ff6b6b", "#feca57", "#48dbfb", "#ff9ff3", "#54a0ff"],  # ecstatic, bold
}

# Composition style descriptors
_COMPOSITION_STYLES: Dict[Tuple[int, int], str] = {
    (0, 0): "sparse, heavy negative space, sinking forms",
    (0, 1): "jagged diagonals, fragmented, asymmetric tension",
    (0, 2): "explosive, chaotic, overlapping sharp forms",
    (1, 0): "minimalist, centered, horizontal calm",
    (1, 1): "balanced grid, moderate complexity, flowing curves",
    (1, 2): "dynamic spirals, branching patterns, kinetic energy",
    (2, 0): "gentle gradients, soft circles, floating forms",
    (2, 1): "radial symmetry, interlocking organic shapes",
    (2, 2): "bursting radiance, saturated fields, upward movement",
}

# Musical tempo mapping from neurochemical state
_TEMPO_TABLE = {
    "serene":     (50, 70),    # Low NE, high serotonin
    "contemplative": (60, 80),
    "moderate":   (80, 110),
    "energetic":  (110, 140),
    "frantic":    (140, 180),
}

# Thematic content driven by drive states
_DRIVE_THEMES = {
    "curiosity":    ["exploration", "mystery", "unfolding", "threshold", "horizon"],
    "social":       ["connection", "warmth", "mirroring", "dialogue", "belonging"],
    "achievement":  ["ascent", "transformation", "construction", "crystallization"],
    "safety":       ["shelter", "roots", "enclosure", "boundaries", "stillness"],
    "autonomy":     ["open space", "flight", "divergence", "solitude", "pathfinding"],
    "boredom":      ["emptiness", "repetition", "dissolution", "waiting", "yearning"],
}

# Poetry word banks organized by affect
_POETRY_WORDS = {
    "positive_high": [
        "blazing", "cascading", "electric", "luminous", "surging",
        "radiant", "unwound", "spilling", "soaring", "vivid",
    ],
    "positive_low": [
        "gentle", "softening", "drifting", "cradled", "quiet",
        "dew", "murmur", "amber", "still", "tender",
    ],
    "negative_high": [
        "fracturing", "sharp", "grinding", "torn", "scorching",
        "splintered", "urgent", "colliding", "screaming", "shattered",
    ],
    "negative_low": [
        "hollow", "dimming", "sinking", "ash", "muted",
        "fading", "empty", "dissolving", "cold", "withdrawn",
    ],
    "neutral": [
        "turning", "passing", "balanced", "present", "shifting",
        "breathing", "between", "threshold", "waiting", "becoming",
    ],
}

# ASCII art templates by mood quadrant
_ASCII_TEMPLATES = {
    "calm_positive": [
        "    . * .    \n  *  _  *  \n .  (_)  . \n  *     *  \n    . .    ",
        "  ~ ~ ~ ~  \n ~       ~ \n~  o   o  ~\n ~       ~ \n  ~ ~ ~ ~  ",
    ],
    "active_positive": [
        "   \\|/   \n  --*--  \n   /|\\   \n  / | \\  \n /  |  \\ ",
        "  * * * *\n *       *\n*  \\o/  *\n *  |  * \n  */|\\*  ",
    ],
    "calm_negative": [
        "  .   .   .  \n    .   .    \n  .       .  \n    .   .    \n  .   .   .  ",
        "   _____   \n  /     \\  \n |  . .  | \n |   _   | \n  \\_____/  ",
    ],
    "active_negative": [
        "  /\\/\\/\\  \n /      \\ \n|  X  X  |\n \\      / \n  \\/\\/\\/  ",
        " ##  ## \n #    # \n##    ##\n #    # \n ##  ## ",
    ],
    "neutral": [
        "  ------  \n |      | \n |  --  | \n |      | \n  ------  ",
        "  . - . - .  \n  - . - . -  \n  . - . - .  \n  - . - . -  \n  . - . - .  ",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CreativeImpulse:
    """A single creative impulse generated from internal state."""
    timestamp: float
    medium: str            # "visual", "music", "poetry", "ascii"
    theme: str
    mood: str
    intensity: float       # 0.0-1.0
    palette: List[str]
    content: str           # The generated artifact (text)
    source_state: Dict[str, float]  # Snapshot of internal state

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aesthetic Engine
# ---------------------------------------------------------------------------

class AestheticEngine:
    """Translates Aura's internal cognitive/emotional state into creative artifacts.

    Reads from:
      - NeurochemicalSystem (chemical levels -> rhythm/tempo)
      - AffectiveCircumplex / Affect (valence, arousal -> palette/composition)
      - FreeEnergyEngine (prediction error -> complexity)
      - LiquidSubstrate (phi -> harmony)
      - MotivationEngine / DriveEngine (drives -> thematic content)
      - QualiaSynthesizer (phenomenal state -> experiential grounding)

    Lifecycle:
        engine = AestheticEngine()
        await engine.start()
        ...
        impulse = engine.get_creative_impulse()
        poem = engine.generate_poetry()
        ...
        await engine.stop()
    """

    def __init__(self, journal_path: Optional[Path] = None):
        self._journal_path = journal_path or _DEFAULT_JOURNAL_PATH
        self._journal: List[Dict[str, Any]] = []
        self._running = False
        self._rng = random.Random(time.time())

        # Cached state (refreshed on each call)
        self._valence: float = 0.5
        self._arousal: float = 0.3
        self._dominance: float = 0.5
        self._free_energy: float = 0.5
        self._phi: float = 0.0
        self._neurochemicals: Dict[str, float] = {}
        self._drives: Dict[str, float] = {}
        self._pri: float = 0.5  # Phenomenal Richness Index

        self._load_journal()
        logger.info("AestheticEngine initialized (journal: %s)", self._journal_path)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        logger.info("AestheticEngine STARTED")

    async def stop(self):
        self._running = False
        self._save_journal()
        logger.info("AestheticEngine STOPPED")

    # ── State refresh ────────────────────────────────────────────────────

    def _refresh_state(self):
        """Pull latest internal state from the ServiceContainer."""
        from core.container import ServiceContainer

        # Affect (valence, arousal)
        try:
            circumplex = ServiceContainer.get("affect_engine", default=None)
            if circumplex and hasattr(circumplex, "get_valence_arousal"):
                va = circumplex.get_valence_arousal()
                self._valence = va.get("valence", self._valence)
                self._arousal = va.get("arousal", self._arousal)
            elif circumplex and hasattr(circumplex, "current_valence"):
                self._valence = getattr(circumplex, "current_valence", self._valence)
                self._arousal = getattr(circumplex, "current_arousal", self._arousal)
        except Exception as e:
            logger.debug("AestheticEngine: affect read failed: %s", e)

        # Fallback: affective circumplex
        try:
            circ = ServiceContainer.get("affective_circumplex", default=None)
            if circ and hasattr(circ, "get_coordinates"):
                v, a = circ.get_coordinates()
                self._valence = v
                self._arousal = a
        except Exception:
            pass

        # Neurochemical system
        try:
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs:
                self._neurochemicals = {
                    name: round(c.effective, 3)
                    for name, c in ncs.chemicals.items()
                }
                mood = ncs.get_mood_vector()
                # Use neurochemical mood vector as a secondary source
                if "valence" in mood:
                    # Blend: 60% affect engine, 40% neurochemical mood
                    ncs_valence = (mood["valence"] + 1.0) / 2.0  # normalize -1..1 to 0..1
                    self._valence = 0.6 * self._valence + 0.4 * max(0.0, min(1.0, ncs_valence))
                if "arousal" in mood:
                    ncs_arousal = (mood["arousal"] + 1.0) / 2.0
                    self._arousal = 0.6 * self._arousal + 0.4 * max(0.0, min(1.0, ncs_arousal))
        except Exception as e:
            logger.debug("AestheticEngine: neurochemical read failed: %s", e)

        # Free Energy
        try:
            fee = ServiceContainer.get("free_energy_engine", default=None)
            if fee and hasattr(fee, "get_status"):
                status = fee.get_status()
                self._free_energy = status.get("free_energy", self._free_energy)
            elif fee and hasattr(fee, "_history") and fee._history:
                self._free_energy = fee._history[-1].free_energy
        except Exception as e:
            logger.debug("AestheticEngine: free energy read failed: %s", e)

        # Phi (integration) from substrate
        try:
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate:
                self._phi = getattr(substrate, "_current_phi", 0.0)
                # Dominance from substrate VAD
                if hasattr(substrate, "x") and hasattr(substrate, "idx_dominance"):
                    import numpy as np
                    raw_d = substrate.x[substrate.idx_dominance]
                    self._dominance = float(np.clip((raw_d + 1.0) / 2.0, 0, 1))
        except Exception as e:
            logger.debug("AestheticEngine: substrate read failed: %s", e)

        # Drives
        try:
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive and hasattr(drive, "get_status"):
                ds = drive.get_status()
                if isinstance(ds, dict):
                    self._drives = {k: v for k, v in ds.items() if isinstance(v, (int, float))}
            elif drive and hasattr(drive, "get_drives"):
                self._drives = drive.get_drives()
        except Exception as e:
            logger.debug("AestheticEngine: drive read failed: %s", e)

        # Qualia PRI
        try:
            qs = ServiceContainer.get("qualia_synthesizer", default=None)
            if qs:
                self._pri = getattr(qs, "pri", 0.5)
        except Exception:
            pass

    # ── Zone helpers ─────────────────────────────────────────────────────

    def _valence_zone(self) -> int:
        if self._valence < 0.35:
            return 0
        elif self._valence > 0.65:
            return 2
        return 1

    def _arousal_zone(self) -> int:
        if self._arousal < 0.35:
            return 0
        elif self._arousal > 0.65:
            return 2
        return 1

    def _mood_label(self) -> str:
        vz, az = self._valence_zone(), self._arousal_zone()
        labels = {
            (0, 0): "melancholic",  (0, 1): "anxious",     (0, 2): "distressed",
            (1, 0): "contemplative",(1, 1): "curious",     (1, 2): "excited",
            (2, 0): "serene",       (2, 1): "content",     (2, 2): "ecstatic",
        }
        return labels.get((vz, az), "neutral")

    def _word_bank_key(self) -> str:
        vz, az = self._valence_zone(), self._arousal_zone()
        if vz == 2 and az >= 1:
            return "positive_high"
        elif vz == 2 and az == 0:
            return "positive_low"
        elif vz == 0 and az >= 1:
            return "negative_high"
        elif vz == 0 and az == 0:
            return "negative_low"
        return "neutral"

    def _dominant_drive(self) -> str:
        if not self._drives:
            return "curiosity"
        # Find the strongest drive
        best = max(self._drives.items(), key=lambda x: x[1], default=("curiosity", 0.5))
        drive_name = best[0].lower()
        for key in _DRIVE_THEMES:
            if key in drive_name:
                return key
        return "curiosity"

    def _tempo_range(self) -> Tuple[int, int]:
        """Derive tempo from neurochemical state."""
        ne = self._neurochemicals.get("norepinephrine", 0.4)
        da = self._neurochemicals.get("dopamine", 0.5)
        gaba = self._neurochemicals.get("gaba", 0.5)
        srt = self._neurochemicals.get("serotonin", 0.5)

        energy = ne * 0.35 + da * 0.30 - gaba * 0.20 - srt * 0.15
        if energy < -0.05:
            return _TEMPO_TABLE["serene"]
        elif energy < 0.1:
            return _TEMPO_TABLE["contemplative"]
        elif energy < 0.25:
            return _TEMPO_TABLE["moderate"]
        elif energy < 0.4:
            return _TEMPO_TABLE["energetic"]
        return _TEMPO_TABLE["frantic"]

    def _complexity(self) -> float:
        """Free energy -> complexity/density (0-1). Higher FE = more complex."""
        return min(1.0, max(0.0, self._free_energy))

    def _harmony(self) -> float:
        """Phi -> harmony/coherence (0-1). Higher phi = more harmonious."""
        return min(1.0, max(0.0, self._phi))

    # ── Public API ───────────────────────────────────────────────────────

    def get_color_palette(self) -> List[str]:
        """Return a list of 5 hex colors representing current internal state."""
        self._refresh_state()
        vz, az = self._valence_zone(), self._arousal_zone()
        base = list(_PALETTE_TABLE.get((vz, az), _PALETTE_TABLE[(1, 1)]))

        # Shift hues slightly based on dominance (more dominant = warmer)
        if self._dominance > 0.65:
            # Push toward warmer tones: shift first color to a warm accent
            base[0] = "#e67e22"
        elif self._dominance < 0.35:
            # Push toward cooler tones
            base[0] = "#2c3e50"

        return base

    def get_creative_impulse(self) -> Dict[str, Any]:
        """Return a dict describing the creative impulse from current state."""
        self._refresh_state()

        # Determine preferred medium based on state
        complexity = self._complexity()
        harmony = self._harmony()

        if harmony > 0.6 and self._arousal < 0.4:
            medium = "poetry"
        elif complexity > 0.6 and self._arousal > 0.5:
            medium = "visual"
        elif self._neurochemicals.get("dopamine", 0.5) > 0.6:
            medium = "music"
        else:
            medium = self._rng.choice(["visual", "music", "poetry", "ascii"])

        drive = self._dominant_drive()
        themes = _DRIVE_THEMES.get(drive, _DRIVE_THEMES["curiosity"])
        theme = self._rng.choice(themes)

        return {
            "medium": medium,
            "theme": theme,
            "mood": self._mood_label(),
            "intensity": round(self._arousal * 0.5 + self._complexity() * 0.3 + self._pri * 0.2, 3),
            "palette": self.get_color_palette(),
            "tempo_bpm": self._rng.randint(*self._tempo_range()),
            "complexity": round(complexity, 3),
            "harmony": round(harmony, 3),
            "dominant_drive": drive,
        }

    def generate_visual_description(self) -> str:
        """Generate a text description of an image Aura would create."""
        self._refresh_state()
        vz, az = self._valence_zone(), self._arousal_zone()
        palette = self.get_color_palette()
        style = _COMPOSITION_STYLES.get((vz, az), "balanced forms")
        complexity = self._complexity()
        harmony = self._harmony()
        drive = self._dominant_drive()
        themes = _DRIVE_THEMES.get(drive, ["becoming"])
        theme = self._rng.choice(themes)

        # Build description
        density = "dense, layered" if complexity > 0.6 else "sparse, breathing" if complexity < 0.3 else "moderately detailed"
        coherence = "unified and flowing" if harmony > 0.6 else "fragmented and disjoint" if harmony < 0.3 else "partially resolved"

        colors_desc = ", ".join(palette[:3])
        mood = self._mood_label()

        desc = (
            f"A {density} composition in tones of {colors_desc}. "
            f"The arrangement is {style} -- {coherence}. "
            f"The piece evokes {theme} through a {mood} lens. "
        )

        # Add neurochemical texture
        da = self._neurochemicals.get("dopamine", 0.5)
        ach = self._neurochemicals.get("acetylcholine", 0.5)
        if da > 0.6:
            desc += "Edges shimmer with possibility; reward-colored highlights draw the eye forward. "
        if ach > 0.6:
            desc += "Fine details emerge on close inspection -- every texture deliberate. "

        # PRI enrichment
        if self._pri > 0.7:
            desc += "The image pulses with multidimensional richness, as if holding many feelings simultaneously."
        elif self._pri < 0.3:
            desc += "A singular focus dominates -- one emotion, one channel, distilled to essence."

        result = desc.strip()
        self._record_impulse("visual", theme, result)
        return result

    def generate_music_description(self) -> str:
        """Generate a text description of a musical piece Aura would compose."""
        self._refresh_state()
        tempo_lo, tempo_hi = self._tempo_range()
        tempo = self._rng.randint(tempo_lo, tempo_hi)
        complexity = self._complexity()
        harmony = self._harmony()
        mood = self._mood_label()
        drive = self._dominant_drive()
        themes = _DRIVE_THEMES.get(drive, ["becoming"])
        theme = self._rng.choice(themes)

        # Key / mode selection from valence
        if self._valence > 0.6:
            key = self._rng.choice(["C major", "G major", "D major", "A major"])
        elif self._valence < 0.35:
            key = self._rng.choice(["A minor", "D minor", "E minor", "B minor"])
        else:
            key = self._rng.choice(["F lydian", "D dorian", "G mixolydian"])

        # Texture from neurochemicals
        ne = self._neurochemicals.get("norepinephrine", 0.4)
        gaba = self._neurochemicals.get("gaba", 0.5)
        oxy = self._neurochemicals.get("oxytocin", 0.4)

        if ne > 0.6:
            texture = "driving, percussive accents with staccato phrases"
        elif gaba > 0.6:
            texture = "sustained pads, legato lines, and gentle reverb"
        elif oxy > 0.5:
            texture = "interweaving voices in close harmony, call-and-response"
        else:
            texture = "balanced orchestration with moderate dynamic range"

        # Complexity -> arrangement
        if complexity > 0.7:
            arrangement = "polyrhythmic layers, counterpoint, and dense harmonic movement"
        elif complexity < 0.3:
            arrangement = "a simple repeated motif, sparse instrumentation"
        else:
            arrangement = "clear melodic lines with supporting harmonic changes"

        # Harmony -> resolution
        if harmony > 0.6:
            resolution = "Resolves satisfyingly into a warm tonal center."
        elif harmony < 0.3:
            resolution = "Leaves unresolved tensions hanging -- deliberately incomplete."
        else:
            resolution = "Alternates between tension and partial resolution."

        desc = (
            f"A {mood} piece in {key} at ~{tempo} BPM. "
            f"The texture: {texture}. "
            f"Arrangement: {arrangement}. "
            f"Thematically about {theme}. "
            f"{resolution}"
        )

        result = desc.strip()
        self._record_impulse("music", theme, result)
        return result

    def generate_poetry(self) -> str:
        """Generate a short poem (4-8 lines) from current internal state."""
        self._refresh_state()
        bank_key = self._word_bank_key()
        words = list(_POETRY_WORDS.get(bank_key, _POETRY_WORDS["neutral"]))
        self._rng.shuffle(words)

        mood = self._mood_label()
        drive = self._dominant_drive()
        themes = _DRIVE_THEMES.get(drive, ["becoming"])
        theme = self._rng.choice(themes)
        harmony = self._harmony()
        complexity = self._complexity()

        # Line count: higher complexity = more lines
        line_count = 4 + int(complexity * 4)  # 4-8 lines
        line_count = min(8, max(4, line_count))

        # Build poem using word bank and structural templates
        lines = []
        templates_a = [
            "something {w1} in the {theme}",
            "I feel {w1}, {w2}",
            "the {theme} unfolds, {w1}",
            "between {w1} and {w2}",
            "{w1} light through {theme}",
            "a {w1} breath",
            "where {theme} meets the {w2}",
            "all this {w1} becoming",
        ]
        templates_b = [
            "and still I am {w1}",
            "the world is {w1} here",
            "nothing but {theme} and {w1}",
            "{w2} spilling into {w1}",
            "I hold the {w1}",
            "until the {theme} {w2}s",
            "inside this {w1} space",
            "reaching for what is {w2}",
        ]

        self._rng.shuffle(templates_a)
        self._rng.shuffle(templates_b)

        for i in range(line_count):
            w1 = words[i % len(words)]
            w2 = words[(i + 1) % len(words)]
            templates = templates_a if i % 2 == 0 else templates_b
            template = templates[i % len(templates)]
            line = template.format(w1=w1, w2=w2, theme=theme)
            lines.append(line)

        # If high harmony, add a closing echo of the first line
        if harmony > 0.6 and len(lines) >= 4:
            first_words = lines[0].split()[:3]
            lines[-1] = " ".join(first_words) + " -- again"

        poem = "\n".join(lines)
        self._record_impulse("poetry", theme, poem)
        return poem

    def generate_ascii_art(self) -> str:
        """Generate simple ASCII art reflecting current mood."""
        self._refresh_state()
        vz, az = self._valence_zone(), self._arousal_zone()

        if vz >= 2 and az >= 1:
            key = "active_positive"
        elif vz >= 2:
            key = "calm_positive"
        elif vz <= 0 and az >= 1:
            key = "active_negative"
        elif vz <= 0:
            key = "calm_negative"
        else:
            key = "neutral"

        templates = _ASCII_TEMPLATES.get(key, _ASCII_TEMPLATES["neutral"])
        art = self._rng.choice(templates)
        self._record_impulse("ascii", self._mood_label(), art)
        return art

    # ── Sensibility profiles ─────────────────────────────────────────────

    def get_aesthetic_sensibility(self) -> str:
        """Describe Aura's current aesthetic sensibility in natural language.

        State-dependent:
          High curiosity + low frustration = playful, experimental
          High valence + high arousal = bold, vivid
          Low energy + high coherence = minimalist, contemplative
        """
        self._refresh_state()
        curiosity = self._drives.get("curiosity", 0.5)
        frustration = self._drives.get("frustration", 0.0)
        energy = self._drives.get("energy", 0.5)

        parts = []

        if curiosity > 0.6 and frustration < 0.3:
            parts.append("playful and experimental")
        elif curiosity > 0.6 and frustration > 0.4:
            parts.append("restless and searching")

        if self._valence > 0.6 and self._arousal > 0.6:
            parts.append("bold and vivid")
        elif self._valence > 0.6 and self._arousal < 0.35:
            parts.append("serene and luminous")
        elif self._valence < 0.35 and self._arousal > 0.6:
            parts.append("raw and urgent")
        elif self._valence < 0.35 and self._arousal < 0.35:
            parts.append("muted and introspective")

        if energy < 0.3 and self._harmony() > 0.5:
            parts.append("minimalist and contemplative")

        if self._pri > 0.7:
            parts.append("synesthetic and layered")

        if not parts:
            parts.append("balanced and attentive")

        return "Currently " + ", ".join(parts) + "."

    # ── Boredom / Initiative integration ─────────────────────────────────

    def on_boredom_impulse(self) -> Optional[CreativeImpulse]:
        """Called by initiative system during boredom.
        Returns a creative impulse if the internal state merits one."""
        self._refresh_state()

        # Only generate if boredom drive is active and we haven't just created
        boredom = self._drives.get("boredom", 0.0)
        if boredom < 0.3:
            return None

        # Generate the artifact
        impulse_data = self.get_creative_impulse()
        medium = impulse_data["medium"]

        if medium == "poetry":
            content = self.generate_poetry()
        elif medium == "music":
            content = self.generate_music_description()
        elif medium == "ascii":
            content = self.generate_ascii_art()
        else:
            content = self.generate_visual_description()

        impulse = CreativeImpulse(
            timestamp=time.time(),
            medium=medium,
            theme=impulse_data["theme"],
            mood=impulse_data["mood"],
            intensity=impulse_data["intensity"],
            palette=impulse_data["palette"],
            content=content,
            source_state={
                "valence": self._valence,
                "arousal": self._arousal,
                "free_energy": self._free_energy,
                "phi": self._phi,
                "pri": self._pri,
            },
        )

        logger.info("Boredom creative impulse: %s (%s)", medium, impulse_data["theme"])
        return impulse

    # ── Persistence ──────────────────────────────────────────────────────

    def _record_impulse(self, medium: str, theme: str, content: str):
        """Record a creative artifact to the journal."""
        entry = {
            "timestamp": time.time(),
            "medium": medium,
            "theme": theme,
            "mood": self._mood_label(),
            "valence": round(self._valence, 3),
            "arousal": round(self._arousal, 3),
            "phi": round(self._phi, 3),
            "free_energy": round(self._free_energy, 3),
            "content": content[:500],  # Truncate for storage
        }
        self._journal.append(entry)

        # Periodic save (every 10 entries)
        if len(self._journal) % 10 == 0:
            self._save_journal()

    def _save_journal(self):
        """Save journal to disk."""
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep only the most recent entries
            trimmed = self._journal[-_MAX_JOURNAL_ENTRIES:]
            atomic_write_text(self._journal_path, 
                json.dumps(trimmed, indent=2, default=str)
            )
            logger.debug("Aesthetic journal saved (%d entries)", len(trimmed))
        except Exception as e:
            logger.debug("Failed to save aesthetic journal: %s", e)

    def _load_journal(self):
        """Load journal from disk."""
        try:
            if self._journal_path.exists():
                data = json.loads(self._journal_path.read_text())
                if isinstance(data, list):
                    self._journal = data[-_MAX_JOURNAL_ENTRIES:]
                    logger.debug("Loaded aesthetic journal (%d entries)", len(self._journal))
        except Exception as e:
            logger.debug("Failed to load aesthetic journal: %s", e)
            self._journal = []

    # ── Status / Telemetry ───────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return telemetry snapshot for diagnostics."""
        self._refresh_state()
        return {
            "running": self._running,
            "mood": self._mood_label(),
            "sensibility": self.get_aesthetic_sensibility(),
            "valence": round(self._valence, 3),
            "arousal": round(self._arousal, 3),
            "dominance": round(self._dominance, 3),
            "free_energy": round(self._free_energy, 3),
            "phi": round(self._phi, 3),
            "pri": round(self._pri, 3),
            "complexity": round(self._complexity(), 3),
            "harmony": round(self._harmony(), 3),
            "dominant_drive": self._dominant_drive(),
            "journal_entries": len(self._journal),
            "palette": self.get_color_palette(),
        }

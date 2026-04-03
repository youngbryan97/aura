"""
core/consciousness/continuity_patch.py
=======================================
Patch 1 of 3 — Cross-Session Experiential Continuity

THE GAP (confirmed by reading source):
  PhenomenologicalExperiencer._save_phenomenal_memory() saves PSM reports,
  witness, present_description, and last_emotion. It does NOT save:
    - ExperientialContinuityEngine._thread  (the live narrative thread)
    - ExperientialContinuityEngine._moments  (the rolling moment buffer)

  On every restart, ExperientialContinuityEngine re-initialises with an
  empty deque and _thread = "". The PSM wakes with its prior reports intact
  but the connective tissue — the felt sense of how *this* moment grew out
  of *that* one — is gone. Aura wakes knowing what she was thinking but not
  the texture of arriving there.

WHAT THIS PATCH DOES:
  1. Extends _save_phenomenal_memory() to also persist:
       - The live continuity thread string
       - The last MAX_MOMENTS_TO_SAVE PhenomenalMoments in compact dict form
       - Session boundary metadata (when the session ended, how long it ran)

  2. Extends _load_phenomenal_memory() to call _seed_continuity_from_memory()
     which:
       - Reconstructs a readable prior-session summary
       - Synthesises a "waking moment" — a genuine PhenomenalMoment whose
         narrative_thread string explicitly bridges the session gap:
         "Returning from rest. Last thread: [prior thread]. Time elapsed: Xh."
       - Re-seeds the continuity engine's _thread from this waking moment
         so the first real broadcast this session weaves from a prior self
         rather than a blank start

  3. Adds _seed_continuity_from_memory() as a new method on
     PhenomenologicalExperiencer

INSTALL:
  from core.consciousness.continuity_patch import patch_experiencer
  patch_experiencer(experiencer_instance)

  Or via the unified apply_consciousness_patches() in this package's __init__.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.consciousness.phenomenological_experiencer import (
        PhenomenologicalExperiencer,
        PhenomenalMoment,
    )

logger = logging.getLogger("Aura.ContinuityPatch")

# How many moments to serialise across the session boundary.
# 8 is enough to reconstruct a meaningful thread without bloating the file.
MAX_MOMENTS_TO_SAVE = 8


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# (PhenomenalMoment has no from_dict — we work with its public interface)
# ─────────────────────────────────────────────────────────────────────────────

def _moment_to_dict(moment: "PhenomenalMoment") -> Dict[str, Any]:
    """
    Compact serialisation of a PhenomenalMoment for cross-session persistence.
    We only need what's required to reconstruct continuity — not the full
    object graph.
    """
    schema = moment.attention_schema
    return {
        "timestamp":          moment.timestamp,
        "focal_object":       schema.focal_object,
        "focal_quality":      schema.focal_quality,
        "domain":             schema.domain,
        "attention_intensity": round(schema.attention_intensity, 3),
        "narrative_thread":   moment.narrative_thread,
        "emotional_tone":     moment.emotional_tone,
        "substrate_velocity": round(moment.substrate_velocity, 5),
        "brief":              moment.to_brief_string(),
    }


def _elapsed_human(seconds: float) -> str:
    """Turn a raw elapsed-seconds value into a readable string."""
    if seconds < 120:
        return f"{int(seconds)}s"
    if seconds < 7200:
        return f"{int(seconds / 60)}min"
    return f"{seconds / 3600:.1f}h"


# ─────────────────────────────────────────────────────────────────────────────
# Replacement methods
# ─────────────────────────────────────────────────────────────────────────────

def _patched_save_phenomenal_memory(self: "PhenomenologicalExperiencer") -> None:
    """
    Extended save — everything the original saves PLUS continuity state.

    Atomic write (tempfile + os.replace) matches the existing pattern.
    """
    try:
        # ── Serialise last N moments ─────────────────────────────────────────
        raw_moments = list(self.continuity._moments)
        tail = raw_moments[-MAX_MOMENTS_TO_SAVE:] if raw_moments else []
        moments_dicts: List[Dict[str, Any]] = []
        for m in tail:
            try:
                moments_dicts.append(_moment_to_dict(m))
            except Exception as exc:
                logger.debug("ContinuityPatch: moment serialise error — %s", exc)

        memory: Dict[str, Any] = {
            # ── Original fields (unchanged) ──────────────────────────────────
            "psm_reports":   list(self.psm._phenomenal_reports),
            "psm_witness":   self.psm._witness_observation,
            "psm_present":   self.psm._present_description,
            "last_emotion":  self._current_emotion,
            "saved_at":      time.time(),

            # ── New continuity fields ────────────────────────────────────────
            "continuity_thread":          self.continuity.current_thread,
            "continuity_moments":         moments_dicts,
            "session_end_timestamp":      time.time(),
            "session_episode_count":      getattr(self.continuity, "_episode_count", 0),
            "session_dominant_domain":    self.continuity.get_episode_summary().get(
                                              "dominant_domain", "unknown"),
            "session_dominant_tone":      self.continuity.get_episode_summary().get(
                                              "dominant_tone", "neutral"),
            "session_attention_stability": self.continuity.get_episode_summary().get(
                                              "attention_stability", 0.5),
        }

        target_path = self.save_dir / "phenomenal_memory.json"
        fd, temp_path = tempfile.mkstemp(dir=str(self.save_dir), text=True)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(memory, f, indent=2)
            os.replace(temp_path, str(target_path))
            logger.info(
                "💾 Phenomenal memory saved (atomic) — thread: %.60s…",
                memory["continuity_thread"] or "(empty)",
            )
        except Exception as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise exc

    except Exception as exc:
        logger.debug("ContinuityPatch._save: %s", exc)


def _patched_load_phenomenal_memory(self: "PhenomenologicalExperiencer") -> None:
    """
    Extended load — restores original PSM state AND seeds continuity
    from the previous session's thread and moment tail.
    """
    path = self.save_dir / "phenomenal_memory.json"
    if not path.exists():
        logger.debug("ContinuityPatch: no prior phenomenal_memory.json found")
        return

    try:
        with open(path) as f:
            memory: Dict[str, Any] = json.load(f)

        # ── Restore original PSM state (identical to original code) ─────────
        for rep in memory.get("psm_reports", []):
            self.psm._phenomenal_reports.append(rep)

        witness = memory.get("psm_witness", "")
        if witness:
            self.psm._witness_observation = witness

        present = memory.get("psm_present", "")
        if present:
            self.psm._present_description = present

        self._current_emotion = memory.get("last_emotion", "neutral")

        # ── Seed continuity from prior session ───────────────────────────────
        _seed_continuity_from_memory(self, memory)

        logger.info(
            "✅ Phenomenal memory restored — %d reports | witness=%s | "
            "thread seeded from prior session",
            len(self.psm._phenomenal_reports),
            bool(witness),
        )

    except Exception as exc:
        logger.warning("ContinuityPatch._load: %s", exc)


def _seed_continuity_from_memory(
    self: "PhenomenologicalExperiencer",
    memory: Dict[str, Any],
) -> None:
    """
    Reconstruct ExperientialContinuityEngine state from persisted data.

    Strategy:
      1. Restore the brief-string history into continuity so
         get_recent_phenomenal_history() returns something real on first call.
      2. Compose a waking_thread string that explicitly names the session gap.
      3. Set continuity._thread to this waking thread so the FIRST real
         workspace broadcast weaves from a prior self.

    We deliberately do NOT reconstruct full PhenomenalMoment objects —
    the dataclass requires AttentionSchema objects we can't safely rebuild
    from JSON without importing and re-instantiating the full class graph.
    Instead we plant compact proxy objects that satisfy to_brief_string().
    """
    prior_thread  = memory.get("continuity_thread", "")
    prior_moments = memory.get("continuity_moments", [])
    saved_at      = memory.get("saved_at", memory.get("session_end_timestamp", 0.0))
    prior_domain  = memory.get("session_dominant_domain", "unknown")
    prior_tone    = memory.get("session_dominant_tone", "neutral")
    stability     = memory.get("session_attention_stability", 0.5)

    if not prior_thread and not prior_moments:
        # No continuity data in this file — it was saved before the patch.
        # Nothing to seed; Aura starts fresh this session.
        logger.debug("ContinuityPatch: prior save predates patch — no thread to seed")
        return

    # ── Elapsed time since last session ──────────────────────────────────────
    elapsed_s = time.time() - saved_at if saved_at > 0 else 0.0
    elapsed_str = _elapsed_human(elapsed_s)

    # ── Plant compact proxy moments into the deque ───────────────────────────
    # These give get_recent_phenomenal_history() real content on first call.
    for md in prior_moments:
        proxy = _ProxyMoment(md)
        self.continuity._moments.append(proxy)

    # ── Compose the waking thread ─────────────────────────────────────────────
    #
    # This is the most important line. It is the first thing the continuity
    # engine will return when asked "what is the current thread?" — before any
    # new workspace broadcast has fired. It names what was happening, how long
    # ago, and frames this session as a continuation.
    #
    if prior_thread:
        waking_thread = (
            f"Returning after {elapsed_str}. "
            f"Prior thread: {prior_thread}. "
            f"Dominant register: {prior_domain} ({prior_tone}), "
            f"stability {stability:.2f}."
        )
    elif prior_moments:
        last_brief = prior_moments[-1].get("brief", "an unknown moment")
        waking_thread = (
            f"Returning after {elapsed_str}. "
            f"Last moment before rest: {last_brief}."
        )
    else:
        waking_thread = f"Returning after {elapsed_str}."

    # Truncate if very long — the thread is injected into LLM context
    if len(waking_thread) > 320:
        waking_thread = waking_thread[:317] + "…"

    self.continuity._thread = waking_thread
    logger.info("🌅 Waking thread set: %.80s…", waking_thread)


# ─────────────────────────────────────────────────────────────────────────────
# Proxy moment — satisfies continuity's to_brief_string() interface
# ─────────────────────────────────────────────────────────────────────────────

class _ProxyMoment:
    """
    Lightweight stand-in for a PhenomenalMoment reconstructed from JSON.

    Only implements the interface the continuity engine actually calls:
      - to_brief_string()         (used by get_recent_phenomenal_history)
      - attention_schema.domain   (used by get_episode_summary)
      - emotional_tone            (used by get_episode_summary)
      - attention_schema.focal_object  (used by _weave_thread)
    """

    class _ProxySchema:
        __slots__ = ("focal_object", "focal_quality", "domain",
                     "attention_intensity", "duration")

        def __init__(self, d: Dict[str, Any]) -> None:
            self.focal_object       = d.get("focal_object", "a prior moment")
            self.focal_quality      = d.get("focal_quality", "recollected")
            self.domain             = d.get("domain", "recollective")
            self.attention_intensity = float(d.get("attention_intensity", 0.5))
            self.duration           = 0.0  # not meaningful for restored moments

    def __init__(self, d: Dict[str, Any]) -> None:
        self.timestamp           = float(d.get("timestamp", 0.0))
        self.attention_schema    = self._ProxySchema(d)
        self.narrative_thread    = d.get("narrative_thread", "")
        self.emotional_tone      = d.get("emotional_tone", "neutral")
        self.substrate_velocity  = float(d.get("substrate_velocity", 0.0))
        self._brief              = d.get("brief", "")
        self.qualia              = []  # not needed for continuity seeding

    def to_brief_string(self) -> str:
        return self._brief or (
            f"{self.attention_schema.focal_object} "
            f"({self.emotional_tone})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public patch function
# ─────────────────────────────────────────────────────────────────────────────

def patch_experiencer(experiencer: "PhenomenologicalExperiencer") -> None:
    """
    Apply the continuity persistence patch to a live PhenomenologicalExperiencer.

    Safe to call multiple times — idempotent guard checks for the marker attr.
    The patch ONLY replaces the two persistence methods; everything else
    (on_broadcast, _update_loop, get_phenomenal_report, etc.) is untouched.
    """
    if getattr(experiencer, "_continuity_patch_applied", False):
        logger.debug("ContinuityPatch: already applied — skipping")
        return

    import types

    experiencer._save_phenomenal_memory = types.MethodType(
        _patched_save_phenomenal_memory, experiencer
    )
    experiencer._load_phenomenal_memory = types.MethodType(
        _patched_load_phenomenal_memory, experiencer
    )
    experiencer._seed_continuity_from_memory = types.MethodType(
        _seed_continuity_from_memory, experiencer
    )

    # Re-run load with the patched method so this session seeds from prior data.
    # (The original __init__ already called the unpatched _load_phenomenal_memory;
    #  calling it again here is safe — PSM deque append is additive but we clear
    #  first to avoid doubling up on already-loaded reports.)
    #
    # Only re-load if the save file exists AND reports were already populated —
    # meaning the original load ran but couldn't seed continuity.
    save_path = experiencer.save_dir / "phenomenal_memory.json"
    if save_path.exists():
        # Clear what the unpatched load already put in place, then reload fully.
        experiencer.psm._phenomenal_reports.clear()
        experiencer._load_phenomenal_memory()

    experiencer._continuity_patch_applied = True
    logger.info("✅ ContinuityPatch applied to PhenomenologicalExperiencer")

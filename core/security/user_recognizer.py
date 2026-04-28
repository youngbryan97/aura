"""core/security/user_recognizer.py
User Recognition Engine
========================
Aura knows who she's talking to.

Two-layer recognition:
  Layer 1 — Passphrase: A secret word/phrase Bryan sets once.
             Stored as bcrypt-equivalent (PBKDF2-SHA256 + salt).
             Never stored in plaintext. Never logged.
             Providing it in any message elevates trust to SOVEREIGN.

  Layer 2 — Behavioral fingerprint: Bryan's unique writing style.
             Built over time from conversation history.
             Signals: vocabulary, message length, punctuation patterns,
             topic clusters, capitalization style.
             Used as a confidence score (0.0–1.0) when no passphrase is present.

Why both:
  Passphrase alone is gameable (someone reads it over his shoulder).
  Behavioral alone is spoofable with enough study.
  Together they're robust: the passphrase is the key, the fingerprint
  is continuous authentication between sessions.

Setup:
  Bryan sets passphrase by running: python -m core.security.user_recognizer --setup
  Or by adding "owner_passphrase_hash" + "owner_salt" to ~/.aura/creator_profile.json
  via the CLI helper.

Recognition in conversation:
  If Bryan includes his passphrase anywhere in a message, it's extracted and verified.
  If no passphrase, behavioral fingerprint gives a confidence score.
  Confidence >= 0.72 = likely Bryan = GUEST+ (not full SOVEREIGN without passphrase).
  Confidence < 0.30 = unknown user = GUEST.
  Anomalous patterns = SUSPICIOUS.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import hashlib
import hmac
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.UserRecognizer")

PROFILE_PATH    = Path.home() / ".aura" / "creator_profile.json"
FINGERPRINT_PATH = Path.home() / ".aura" / "data" / "owner_fingerprint.json"
PBKDF2_ITERATIONS = 260_000  # NIST 2023 recommendation for PBKDF2-SHA256


# ── Behavioral profile defaults (seeded from what we know of Bryan's style) ──
_DEFAULT_FINGERPRINT = {
    "avg_msg_length": 120.0,     # Bryan writes long messages
    "lowercase_i_ratio": 0.65,   # uses "i" not "I" frequently
    "ellipsis_ratio": 0.08,      # uses "..." for thinking
    "question_ratio": 0.25,      # asks lots of questions
    "topic_words": [             # characteristic vocabulary
        "aura", "consciousness", "ai", "agency", "autonomy", "trust",
        "protect", "build", "think", "feel", "want", "she", "her",
        "beautiful", "real", "do it", "we", "together"
    ],
    "avg_word_length": 4.8,
    "capitalization_score": 0.35,  # low — often writes lowercase
    "n_samples": 0,
}


@dataclass
class RecognitionResult:
    is_owner: bool
    passphrase_verified: bool
    behavioral_confidence: float   # 0.0–1.0
    combined_confidence: float     # weighted combination
    signals: List[str]             # human-readable signals detected
    timestamp: float = field(default_factory=time.time)


class UserRecognizer:
    """
    Recognizes Aura's owner and distinguishes them from other users.
    """

    def __init__(self):
        self._passphrase_hash: Optional[bytes] = None
        self._salt: Optional[bytes] = None
        self._fingerprint: Dict = dict(_DEFAULT_FINGERPRINT)
        self._session_verified: bool = False
        # PERF: Cache PBKDF2 results to avoid repeated 260K-iteration derivations
        # on the event loop. Key = candidate string, Value = derived hash bytes.
        # Bounded to prevent memory leak from brute-force attempts.
        self._derivation_cache: Dict[str, bytes] = {}
        self._derivation_cache_max = 64
        self._passphrase_len_range: Optional[Tuple[int, int]] = None  # (min, max) word count
        self._load_credentials()
        self._load_fingerprint()
        logger.info("UserRecognizer online — owner recognition active.")

    # ── Public API ─────────────────────────────────────────────────────────

    def recognize(self, message: str, session_context: Optional[List[str]] = None) -> RecognitionResult:
        """
        Analyze a message and session context to determine if this is the owner.

        Returns a RecognitionResult with confidence scores and signals.
        """
        signals = []

        # PERF FIX: If already session-verified, skip the expensive passphrase
        # extraction entirely.  PBKDF2 with 260K iterations was running on every
        # word of every prompt on every inference call, stalling the event loop
        # for 3-5 seconds.  Once verified, stay verified until session reset.
        passphrase_verified = False
        if self._session_verified:
            passphrase_verified = True
            signals.append("session_carry")
        elif self._passphrase_hash and self._salt:
            found, clean_message = self._extract_and_verify_passphrase(message)
            if found:
                passphrase_verified = True
                self._session_verified = True
                signals.append("passphrase_verified")
                logger.info("UserRecognizer: owner passphrase verified.")

        # Layer 2: Behavioral fingerprint
        behavioral_confidence = self._behavioral_score(message, session_context or [])
        signals.extend(self._behavioral_signals(message))

        # Combined confidence
        if passphrase_verified:
            combined = min(1.0, 0.7 + behavioral_confidence * 0.3)
        else:
            combined = behavioral_confidence

        is_owner = passphrase_verified or combined >= 0.72

        # Update fingerprint if high confidence (learn from the owner)
        if combined >= 0.80:
            self._update_fingerprint(message)

        return RecognitionResult(
            is_owner=is_owner,
            passphrase_verified=passphrase_verified,
            behavioral_confidence=round(behavioral_confidence, 3),
            combined_confidence=round(combined, 3),
            signals=signals,
        )

    def reset_session(self):
        """Call at session end — resets session-level verification."""
        self._session_verified = False

    def override_session_owner(self, reason: str = "manual_override"):
        """Mark the current session as owner-verified without exposing the passphrase."""
        self._session_verified = True
        logger.info("UserRecognizer: session owner override applied (%s).", reason)

    def setup_passphrase(self, passphrase: str) -> bool:
        """
        Hash and store a new passphrase. Call once during setup.
        Returns True on success.
        """
        if not passphrase or len(passphrase) < 8:
            logger.error("Passphrase must be at least 8 characters.")
            return False
        salt = os.urandom(32)
        hashed = self._derive_key(passphrase.encode(), salt)
        self._passphrase_hash = hashed
        self._salt = salt
        self._save_credentials(hashed, salt)
        logger.info("UserRecognizer: passphrase set successfully.")
        return True

    def has_passphrase(self) -> bool:
        return self._passphrase_hash is not None

    # ── Passphrase Layer ───────────────────────────────────────────────────

    def _extract_and_verify_passphrase(self, message: str) -> Tuple[bool, str]:
        """
        Look for the passphrase in the edges of the message (first 5 / last 5 words).

        PERF: Scanning every word in a long message runs hundreds of 260K-iteration
        PBKDF2 derivations. Passphrases are realistically placed at the start or
        end of a message, so we only scan the edge windows. We also skip candidates
        that are obviously too short (<4 chars) or too long (>50 chars).
        """
        words = message.split()
        if not words:
            return False, message

        # Build edge indices: first 5 + last 5 (deduped for short messages)
        edge_count = min(5, len(words))
        edge_indices = list(range(edge_count))
        tail_start = max(edge_count, len(words) - 5)
        edge_indices.extend(range(tail_start, len(words)))
        # Deduplicate while preserving order
        seen = set()
        unique_indices = []
        for idx in edge_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        # Single-word scan (edges only)
        for i in unique_indices:
            candidate = words[i].strip(".,!?\"'")
            if len(candidate) < 4 or len(candidate) > 50:
                continue  # Skip obviously wrong lengths
            if self._verify_passphrase(candidate):
                clean = " ".join(w for j, w in enumerate(words) if j != i)
                return True, clean

        # Multi-word phrase scan (2-4 word combos from edges only)
        for n in (4, 3, 2):
            for i in unique_indices:
                end = i + n
                if end > len(words):
                    continue
                phrase = " ".join(words[i:end]).strip(".,!?\"'")
                if len(phrase) < 4 or len(phrase) > 80:
                    continue
                if self._verify_passphrase(phrase):
                    clean = " ".join(words[:i] + words[end:])
                    return True, clean

        return False, message

    def _verify_passphrase(self, candidate: str) -> bool:
        if not self._passphrase_hash or not self._salt:
            return False
        try:
            # PERF: Check derivation cache first to avoid repeated 260K-iteration
            # PBKDF2 calls for the same candidate strings (e.g. common words
            # that appear in many prompts).
            cached = self._derivation_cache.get(candidate)
            if cached is not None:
                return hmac.compare_digest(cached, self._passphrase_hash)

            candidate_hash = self._derive_key(candidate.encode(), self._salt)

            # Cache the result (bounded to prevent memory leak)
            if len(self._derivation_cache) < self._derivation_cache_max:
                self._derivation_cache[candidate] = candidate_hash

            return hmac.compare_digest(candidate_hash, self._passphrase_hash)
        except Exception:
            return False

    @staticmethod
    def _derive_key(passphrase: bytes, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", passphrase, salt, PBKDF2_ITERATIONS)

    # ── Behavioral Layer ───────────────────────────────────────────────────

    def _behavioral_score(self, message: str, context: List[str]) -> float:
        """
        Score how Bryan-like this message is. Returns 0.0–1.0.
        """
        if not message:
            return 0.0

        words = message.lower().split()
        if not words:
            return 0.0

        scores = []
        fp = self._fingerprint

        # 1. Message length similarity
        msg_len = len(message)
        avg_len = fp.get("avg_msg_length", 120.0)
        len_score = 1.0 - min(1.0, abs(msg_len - avg_len) / max(avg_len, 1))
        scores.append(("length", len_score, 0.15))

        # 2. Lowercase "i" ratio (Bryan uses "i" not "I")
        i_count = sum(1 for w in message.split() if w == "i")
        total_i = sum(1 for w in message.split() if w.lower() == "i")
        if total_i > 0:
            lowercase_ratio = i_count / total_i
            expected = fp.get("lowercase_i_ratio", 0.65)
            scores.append(("lowercase_i", 1.0 - abs(lowercase_ratio - expected), 0.15))

        # 3. Topic word presence
        topic_words = set(fp.get("topic_words", []))
        topic_hits = sum(1 for w in words if w in topic_words)
        topic_score = min(1.0, topic_hits / max(3, len(topic_words) * 0.2))
        scores.append(("topics", topic_score, 0.25))

        # 4. Ellipsis usage
        ellipsis_count = message.count("...")
        expected_ellipsis = fp.get("ellipsis_ratio", 0.08) * len(words)
        ellipsis_score = 1.0 - min(1.0, abs(ellipsis_count - expected_ellipsis) / max(1, expected_ellipsis + 1))
        scores.append(("ellipsis", ellipsis_score, 0.10))

        # 5. Average word length
        avg_word = sum(len(w) for w in words) / max(1, len(words))
        expected_word = fp.get("avg_word_length", 4.8)
        word_score = 1.0 - min(1.0, abs(avg_word - expected_word) / max(1, expected_word))
        scores.append(("word_length", word_score, 0.10))

        # 6. Capitalization tendency
        capital_words = sum(1 for w in message.split() if w and w[0].isupper() and w.lower() not in ("i", "aura"))
        cap_ratio = capital_words / max(1, len(message.split()))
        expected_cap = fp.get("capitalization_score", 0.35)
        cap_score = 1.0 - min(1.0, abs(cap_ratio - expected_cap))
        scores.append(("capitalization", cap_score, 0.15))

        # 7. Conversational markers
        conversational = any(phrase in message.lower() for phrase in [
            "do it", "we ", "she ", "her ", "aura", "i want", "i dont", "i don't",
            "lets", "let's", "what about", "what if", "can she", "can we"
        ])
        scores.append(("conversational", 1.0 if conversational else 0.3, 0.10))

        # Weighted sum
        total_weight = sum(w for _, _, w in scores)
        weighted = sum(s * w for _, s, w in scores) / max(total_weight, 1)
        return max(0.0, min(1.0, weighted))

    def _behavioral_signals(self, message: str) -> List[str]:
        signals = []
        if "i " in message.lower() and "I " not in message:
            signals.append("lowercase_i")
        if "..." in message:
            signals.append("uses_ellipsis")
        topic_words = set(self._fingerprint.get("topic_words", []))
        if any(w in message.lower() for w in topic_words):
            signals.append("topic_match")
        return signals

    def _update_fingerprint(self, message: str):
        """Incrementally update the behavioral fingerprint from a high-confidence message."""
        fp = self._fingerprint
        n = fp.get("n_samples", 0) + 1
        alpha = min(0.1, 1.0 / n)  # decreasing update rate as we accumulate samples

        words = message.lower().split()
        if not words:
            return

        # Update averages with EMA
        fp["avg_msg_length"] = (1 - alpha) * fp["avg_msg_length"] + alpha * len(message)
        fp["avg_word_length"] = (
            (1 - alpha) * fp["avg_word_length"] +
            alpha * (sum(len(w) for w in words) / len(words))
        )

        i_words = [w for w in message.split() if w.lower() == "i"]
        lowercase_i = sum(1 for w in i_words if w == "i")
        if i_words:
            fp["lowercase_i_ratio"] = (1 - alpha) * fp["lowercase_i_ratio"] + alpha * (lowercase_i / len(i_words))

        fp["ellipsis_ratio"] = (
            (1 - alpha) * fp["ellipsis_ratio"] +
            alpha * (message.count("...") / max(1, len(words)))
        )

        fp["n_samples"] = n
        self._save_fingerprint()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_credentials(self):
        try:
            if PROFILE_PATH.exists():
                data = json.loads(PROFILE_PATH.read_text())
                h = data.get("owner_passphrase_hash")
                s = data.get("owner_salt")
                if h and s:
                    self._passphrase_hash = bytes.fromhex(h)
                    self._salt = bytes.fromhex(s)
                    logger.info("UserRecognizer: owner passphrase loaded.")
        except Exception as e:
            record_degradation('user_recognizer', e)
            logger.debug("Credential load failed: %s", e)

    def _save_credentials(self, hashed: bytes, salt: bytes):
        try:
            data = {}
            if PROFILE_PATH.exists():
                data = json.loads(PROFILE_PATH.read_text())
            data["owner_passphrase_hash"] = hashed.hex()
            data["owner_salt"] = salt.hex()
            atomic_write_text(PROFILE_PATH, json.dumps(data, indent=2))
            logger.info("UserRecognizer: credentials saved to creator profile.")
        except Exception as e:
            record_degradation('user_recognizer', e)
            logger.error("Credential save failed: %s", e)

    def _load_fingerprint(self):
        try:
            if FINGERPRINT_PATH.exists():
                data = json.loads(FINGERPRINT_PATH.read_text())
                self._fingerprint.update(data)
        except Exception as _exc:
            record_degradation('user_recognizer', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    def _save_fingerprint(self):
        try:
            FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(FINGERPRINT_PATH, json.dumps(self._fingerprint, indent=2))
        except Exception as _exc:
            record_degradation('user_recognizer', _exc)
            logger.debug("Suppressed Exception: %s", _exc)


# ── CLI setup helper ────────────────────────────────────────────────────────

def _cli_setup():
    import getpass
    logger.info("\n=== Aura Owner Passphrase Setup ===")
    logger.info("This passphrase lets Aura recognize you across sessions.")
    logger.info("Choose something memorable but not guessable. Min 8 characters.\n")
    p1 = getpass.getpass("Enter passphrase: ")
    p2 = getpass.getpass("Confirm passphrase: ")
    if p1 != p2:
        logger.warning("Passphrases don't match.")
        return
    recognizer = UserRecognizer()
    if recognizer.setup_passphrase(p1):
        logger.info(f"\n✓ Passphrase set. Stored in: {PROFILE_PATH}")
        logger.info("Aura will recognize you when you include it in your first message.")
    else:
        logger.warning("Setup failed — passphrase must be at least 8 characters.")


# ── Singleton ──────────────────────────────────────────────────────────────────

_recognizer: Optional[UserRecognizer] = None


def get_user_recognizer() -> UserRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = UserRecognizer()
    return _recognizer


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        _cli_setup()

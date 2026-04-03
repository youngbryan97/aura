"""
Aura's model of Bryan.

A peer doesn't just respond to you — they have a theory of you.
"""

import json
import os
import time
import logging
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
_USER_MODEL_PATH = Path.home() / ".aura" / "data" / "user_model.json"
_SAVE_DEBOUNCE_SECONDS = 1.5


@dataclass
class DomainReliability:
    domain: str
    correct_count: int = 0
    challenged_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        total = self.correct_count + self.challenged_count
        if total == 0:
            return 0.7  # Prior: moderately reliable
        return self.correct_count / total


@dataclass
class UserPattern:
    description: str
    first_observed: float = field(default_factory=time.time)
    observation_count: int = 1
    last_observed: float = field(default_factory=time.time)


@dataclass
class UserModel:
    known_domains: Dict[str, DomainReliability] = field(default_factory=dict)
    observed_patterns: List[UserPattern] = field(default_factory=list)
    stated_values: List[str] = field(default_factory=list)
    conversation_count: int = 0
    total_messages: int = 0
    last_updated: float = field(default_factory=time.time)


class BryanModelEngine:
    """
    Aura's dynamic model of Bryan.
    """

    def __init__(self):
        self._model = self._load()
        self._last_saved = 0.0
        self._save_lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        logger.info("🧠 Bryan model loaded: %d domain records, %d patterns",
                    len(self._model.known_domains), len(self._model.observed_patterns))

    def _load(self) -> UserModel:
        if _USER_MODEL_PATH.exists():
            try:
                with open(_USER_MODEL_PATH) as f:
                    data = json.load(f)
                    # Reconstruct nested dataclasses
                    domains = {
                        k: DomainReliability(**v)
                        for k, v in data.get("known_domains", {}).items()
                    }
                    patterns = [UserPattern(**p) for p in data.get("observed_patterns", [])]
                    return UserModel(
                        known_domains=domains,
                        observed_patterns=patterns,
                        stated_values=data.get("stated_values", []),
                        conversation_count=data.get("conversation_count", 0),
                        total_messages=data.get("total_messages", 0),
                    )
            except Exception as e:
                logger.warning("User model load failed: %s", e)
        return UserModel()

    def _serialize(self) -> dict:
        return {
            "known_domains": {k: asdict(v) for k, v in self._model.known_domains.items()},
            "observed_patterns": [asdict(p) for p in self._model.observed_patterns],
            "stated_values": self._model.stated_values,
            "conversation_count": self._model.conversation_count,
            "total_messages": self._model.total_messages,
            "last_updated": time.time(),
        }

    def _write_now(self):
        try:
            _USER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = self._serialize()
            fd, tmp_path = tempfile.mkstemp(dir=str(_USER_MODEL_PATH.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, _USER_MODEL_PATH)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            self._last_saved = time.time()
        except Exception as e:
            logger.error("User model save failed: %s", e)

    def _flush_pending_save(self):
        with self._save_lock:
            self._save_timer = None
        self._write_now()

    def save(self, force: bool = False):
        should_write_now = False
        with self._save_lock:
            now = time.time()
            if force or (now - self._last_saved) >= _SAVE_DEBOUNCE_SECONDS:
                timer = self._save_timer
                self._save_timer = None
                if timer and timer.is_alive():
                    timer.cancel()
                should_write_now = True
            elif self._save_timer is None or not self._save_timer.is_alive():
                delay = max(0.1, _SAVE_DEBOUNCE_SECONDS - (now - self._last_saved))
                self._save_timer = threading.Timer(delay, self._flush_pending_save)
                self._save_timer.daemon = True
                self._save_timer.start()
        if should_write_now:
            self._write_now()

    def get_domain_reliability(self, domain: str) -> float:
        """How reliable is Bryan's input in this domain? (0-1)"""
        dr = self._model.known_domains.get(domain)
        return dr.reliability if dr else 0.7

    def record_correct_claim(self, domain: str):
        self._get_or_create_domain(domain).correct_count += 1
        self.save()

    def record_challenged_claim(self, domain: str):
        self._get_or_create_domain(domain).challenged_count += 1
        self.save()

    def record_stated_value(self, value: str):
        if value not in self._model.stated_values:
            self._model.stated_values.append(value)
            # Cap to prevent unbounded growth over months
            if len(self._model.stated_values) > 200:
                self._model.stated_values = self._model.stated_values[-150:]
            self.save()

    def observe_pattern(self, description: str):
        """Note a recurring behavioral pattern."""
        for p in self._model.observed_patterns:
            if description.lower() in p.description.lower():
                p.observation_count += 1
                p.last_observed = time.time()
                self.save()
                return
        self._model.observed_patterns.append(UserPattern(description=description))
        # Cap patterns: keep most-observed and most-recent, prune old low-count ones
        if len(self._model.observed_patterns) > 200:
            self._model.observed_patterns.sort(
                key=lambda p: (p.observation_count, p.last_observed), reverse=True
            )
            self._model.observed_patterns = self._model.observed_patterns[:150]
        self.save()

    def _get_or_create_domain(self, domain: str) -> DomainReliability:
        if domain not in self._model.known_domains:
            self._model.known_domains[domain] = DomainReliability(domain=domain)
        return self._model.known_domains[domain]

    def get_context_for_prompt(self) -> str:
        """Inject Bryan-model context into Aura's system prompt."""
        lines = ["Your model of Bryan:"]
        
        if self._model.stated_values:
            lines.append(f"  Stated values: {'; '.join(self._model.stated_values[-5:])}")
        
        reliable_domains = [
            (d, dr.reliability) for d, dr in self._model.known_domains.items()
            if dr.reliability > 0.8 and (dr.correct_count + dr.challenged_count) > 3
        ]
        if reliable_domains:
            domains_str = ", ".join(f"{d} ({r:.0%})" for d, r in reliable_domains[:5])
            lines.append(f"  High-reliability domains: {domains_str}")

        if self._model.observed_patterns:
            recent = sorted(self._model.observed_patterns,
                           key=lambda p: -p.observation_count)[:3]
            patterns_str = "; ".join(p.description for p in recent)
            lines.append(f"  Observed patterns: {patterns_str}")

        lines.append(f"  Conversations logged: {self._model.conversation_count}")
        return "\n".join(lines)

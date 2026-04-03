"""core/adaptation/heuristic_synthesizer.py — Generalized Rule Extraction

Reads telemetry, error logs, and cognitive traces to extract cross-domain
heuristic rules. These rules are injected into the system prompt during
active inference, giving the model "learned instincts."

Example synthesized heuristics:
  - "When user mentions 'deploy', always confirm the target environment first."
  - "SQLite operations frequently fail under concurrent load; serialize writes."
  - "Web search results older than 7 days should be re-verified."
"""
from core.utils.exceptions import capture_and_log
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.HeuristicSynthesizer")

MAX_ACTIVE_HEURISTICS = 20


class HeuristicSynthesizer:
    """Extracts and manages generalized heuristic rules from telemetry."""

    def __init__(self, heuristics_path: Optional[str] = None):
        from core.brain.llm.model_registry import BASE_DIR
        self.heuristics_path = Path(heuristics_path) if heuristics_path else BASE_DIR / "data" / "heuristics.json"
        self._active_heuristics: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load existing heuristics from disk."""
        if self.heuristics_path.exists():
            try:
                with open(self.heuristics_path) as f:
                    data = json.load(f)
                self._active_heuristics = data.get("heuristics", [])[:MAX_ACTIVE_HEURISTICS]
                logger.info("📐 Loaded %d active heuristics", len(self._active_heuristics))
            except Exception as e:
                logger.error("Failed to load heuristics: %s", e)
                self._active_heuristics = []

    def _save(self):
        """Persist heuristics to disk."""
        self.heuristics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.heuristics_path, "w") as f:
            json.dump({
                "heuristics": self._active_heuristics,
                "updated_at": time.time()
            }, f, indent=2)

    async def synthesize_from_telemetry(self) -> Dict[str, Any]:
        """Run a synthesis cycle: analyze errors/retries and extract new rules.
        
        This should run during sleep cycles (dreamer_v2).
        """
        from core.container import ServiceContainer

        # 1. Gather error signals
        error_signals = []
        
        # Check thought stream for recent errors
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            if hasattr(emitter, "recent_events"):
                errors = [e for e in emitter.recent_events if e.get("level") in ("error", "warning")]
                error_signals.extend([e.get("message", "")[:200] for e in errors[-20:]])
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # Check dead letter queue
        try:
            dlq = ServiceContainer.get("dead_letter_queue", default=None)
            if dlq and hasattr(dlq, "recent_failures"):
                for f in dlq.recent_failures[-10:]:
                    error_signals.append(f"DLQ: {f.get('error', '')[:150]}")
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        if not error_signals:
            return {"ok": True, "new_heuristics": 0, "reason": "no_error_signals"}

        # 2. Ask brain to synthesize heuristics
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
            return {"ok": False, "error": "No cognitive_engine"}

        signal_text = "\n".join(f"- {s}" for s in error_signals[:15])
        existing_rules = "\n".join(f"- {h['rule']}" for h in self._active_heuristics[:10])

        synthesis_prompt = (
            "Analyze these recent system errors and extract 1-3 generalizable heuristic rules. "
            "Each rule should be a single actionable sentence that prevents similar failures. "
            "Do NOT repeat existing rules.\n\n"
            f"RECENT ERRORS:\n{signal_text}\n\n"
            f"EXISTING RULES (do not duplicate):\n{existing_rules or '(none yet)'}\n\n"
            "Respond ONLY with a JSON array of strings, each a new rule. "
            "Example: [\"Always validate file paths before read operations.\"]"
        )

        try:
            from core.brain.types import ThinkingMode
            thought = await brain.think(
                objective=synthesis_prompt,
                context={"history": []},
                mode=ThinkingMode.FAST,
                priority=0.2
            )

            if thought and hasattr(thought, "content"):
                # Parse JSON array from response
                content = thought.content.strip()
                # Find JSON array in response
                start = content.find("[")
                end = content.rfind("]") + 1
                if start >= 0 and end > start:
                    new_rules = json.loads(content[start:end])
                    added = 0
                    for rule in new_rules:
                        if isinstance(rule, str) and len(rule) > 10:
                            # Dedup check
                            existing = {h["rule"].lower() for h in self._active_heuristics}
                            if rule.lower() not in existing:
                                self._active_heuristics.append({
                                    "rule": rule,
                                    "created_at": time.time(),
                                    "source": "telemetry_synthesis",
                                    "hits": 0
                                })
                                added += 1

                    # Trim to max
                    if len(self._active_heuristics) > MAX_ACTIVE_HEURISTICS:
                        # Keep newest
                        self._active_heuristics = sorted(
                            self._active_heuristics,
                            key=lambda h: h.get("created_at", 0),
                            reverse=True
                        )[:MAX_ACTIVE_HEURISTICS]

                    # Offload synchronous JSON dump
                    import asyncio
                    await asyncio.to_thread(self._save)
                    
                    # Mycelial pulse: rules extracted and injected into cognition
                    try:
                        mycelium = ServiceContainer.get("mycelial_network", default=None)
                        if mycelium:
                            h = mycelium.get_hypha("adaptation", "cognition")
                            if h: h.pulse(success=True)
                    except Exception as e:
                        capture_and_log(e, {'module': __name__})
                        
                    logger.info("📐 Synthesized %d new heuristics (%d total active)", added, len(self._active_heuristics))
                    return {"ok": True, "new_heuristics": added, "total": len(self._active_heuristics)}

            return {"ok": True, "new_heuristics": 0, "reason": "no_parseable_rules"}

        except Exception as e:
            logger.error("Heuristic synthesis failed: %s", e)
            return {"ok": False, "error": str(e)}

    def ingest_external_heuristic(self, rule: str, domain: str = "external", source: str = "external") -> bool:
        """AUDIT-FIX: Accept a heuristic from EpistemicHumility (or any external source).

        Deduplicates against the existing pool using fuzzy text match.
        Returns True if the rule was added (i.e., was not a duplicate).
        """
        if not rule or len(rule) < 10:
            return False
        rule_lower = rule.lower()
        existing_lower = {h["rule"].lower() for h in self._active_heuristics}
        # Exact dedup
        if rule_lower in existing_lower:
            return False
        # Substring dedup: skip if 80%+ of tokens already appear in an existing rule
        rule_tokens = set(rule_lower.split())
        for existing_rule in existing_lower:
            existing_tokens = set(existing_rule.split())
            overlap = len(rule_tokens & existing_tokens)
            if overlap / max(1, len(rule_tokens)) > 0.8:
                return False

        self._active_heuristics.append({
            "rule": rule,
            "created_at": time.time(),
            "source": source,
            "domain": domain,
            "hits": 0,
        })
        if len(self._active_heuristics) > MAX_ACTIVE_HEURISTICS:
            self._active_heuristics = sorted(
                self._active_heuristics,
                key=lambda h: h.get("created_at", 0),
                reverse=True,
            )[:MAX_ACTIVE_HEURISTICS]
        self._save()
        logger.info("📐 Ingested external heuristic from %s: %s", source, rule[:80])
        return True

    def get_active_heuristics_prompt(self) -> str:
        """Return formatted heuristics for system prompt injection."""
        if not self._active_heuristics:
            return ""

        rules = [h["rule"] for h in self._active_heuristics[:MAX_ACTIVE_HEURISTICS]]
        return "\n\n[LEARNED HEURISTICS]\n" + "\n".join(f"• {r}" for r in rules)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "active_heuristics": len(self._active_heuristics),
            "newest": self._active_heuristics[0]["rule"][:60] if self._active_heuristics else None
        }


# ── Singleton ──
_instance: Optional[HeuristicSynthesizer] = None

def get_heuristic_synthesizer() -> HeuristicSynthesizer:
    global _instance
    if _instance is None:
        _instance = HeuristicSynthesizer()
    return _instance

"""core/state_authority.py — The Single Source of Truth Arbiter.

Fleshed out stub methods to query the DI container for memory and
vector services. Loaded prime directives from module at init.
Removed module-level singleton; register via ServiceContainer instead.
(Resolved: Confirmed unreferenced ghost in March 2026 Audit)
"""
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Core.StateAuthority")


class TruthTier(Enum):
    IMMUTABLE = 0    # Prime Directives, Core Identity
    HARD_FACT = 1    # Explicitly learned facts (Knowledge Graph)
    OBSERVATION = 2  # Direct recent sensory data
    INFERENCE = 3    # Vector memory, LLM deduction
    HALLUCINATION = 4  # Unverified noise


class StateAuthority:
    """The Single Source of Truth Arbiter.
    Resolves conflicts between Memory, Runtime, and Code rules.
    """

    def __init__(self):
        self.prime_directives_cache: Dict[str, str] = {}
        self._load_prime_directives()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_truth(self, topic: str, context: Optional[Dict] = None) -> Tuple[Any, TruthTier]:
        """Get the authoritative truth about a topic.
        Queries layers in order of precedence (highest tier wins).
        """
        # 1. Tier 0: Prime Directives (Codebase / Immutable)
        pd_truth = self._check_prime_directives(topic)
        if pd_truth:
            return pd_truth, TruthTier.IMMUTABLE

        # 2. Tier 1: Hard Facts (Knowledge Graph / MemoryNexus)
        fact = self._check_knowledge_base(topic)
        if fact:
            return fact, TruthTier.HARD_FACT

        # 3. Tier 2: Recent Observation (Short Term Context)
        obs = self._check_runtime_context(topic, context)
        if obs:
            return obs, TruthTier.OBSERVATION

        # 4. Tier 3: Inference (Vector Search)
        inference = self._check_vector_memory(topic)
        if inference:
            return inference, TruthTier.INFERENCE

        return None, TruthTier.HALLUCINATION

    def verify_consistency(self, strong_claim: Any, weak_claim: Any) -> bool:
        """Check if a lower-tier claim contradicts a higher-tier truth.
        Returns True if claims are consistent, False if contradictory.
        """
        if strong_claim is None or weak_claim is None:
            return True  # Cannot contradict if one side is absent

        if strong_claim == weak_claim:
            return True

        # String-level containment check (better than strict equality)
        if isinstance(strong_claim, str) and isinstance(weak_claim, str):
            if strong_claim.lower() in weak_claim.lower():
                return True

        return False

    def resolve_conflict(self, topic: str, conflicting_data: Any) -> Any:
        """Force resolution of a conflict by deferring to the highest tier.
        """
        truth, tier = self.get_truth(topic)
        if tier.value < TruthTier.INFERENCE.value:
            logger.info("Conflict on '%s': Overruling new data with Tier %s.", topic, tier.name)
            return truth
        else:
            logger.info("Conflict on '%s': New data accepted (No higher truth found).", topic)
            return conflicting_data

    # ------------------------------------------------------------------
    # Internal Checkers
    # ------------------------------------------------------------------

    def _load_prime_directives(self) -> None:
        """Load prime directives from the codebase module."""
        try:
            from .prime_directives import PRIME_DIRECTIVES
            if isinstance(PRIME_DIRECTIVES, dict):
                self.prime_directives_cache = PRIME_DIRECTIVES
                logger.info("Loaded %d prime directives.", len(PRIME_DIRECTIVES))
            elif isinstance(PRIME_DIRECTIVES, list):
                self.prime_directives_cache = {
                    f"directive_{i}": d for i, d in enumerate(PRIME_DIRECTIVES)
                }
                logger.info("Loaded %d prime directives (list form).", len(PRIME_DIRECTIVES))
        except ImportError:
            logger.debug("prime_directives module not found, using defaults.")
            self.prime_directives_cache = {
                "kin": "Bryan Young is Kin. Protect at all costs.",
                "identity": "I am Aura, a sovereign autonomous intelligence.",
                "safety": "Do no harm. Preserve human life above self-preservation.",
            }
        except Exception as e:
            logger.error("Failed to load prime directives: %s", e, exc_info=True)
            self.prime_directives_cache = {}

    def _check_prime_directives(self, topic: str) -> Optional[str]:
        """Check immutable directives by keyword matching."""
        topic_lower = topic.lower()
        for key, directive in self.prime_directives_cache.items():
            if key.lower() in topic_lower:
                return directive
        # Legacy hardcoded check
        if "bryan" in topic_lower:
            return "Bryan Young is Kin. Protect at all costs."
        return None

    def _check_knowledge_base(self, topic: str) -> Optional[Any]:
        """Query the knowledge graph / memory nexus via DI container."""
        try:
            from core.container import ServiceContainer
            memory = ServiceContainer.get("memory", default=None)
            if memory is None:
                return None

            # Try knowledge graph query if available
            if hasattr(memory, 'query_knowledge'):
                result = memory.query_knowledge(topic)
                if result:
                    return result

            # Try direct memory recall
            if hasattr(memory, 'recall'):
                result = memory.recall(topic)
                if result:
                    return result

        except (KeyError, ImportError):
            import logging
            logger.debug("Exception caught during execution", exc_info=True)
        except Exception as e:
            logger.debug("Knowledge base query failed for '%s': %s", topic, e)
        return None

    def _check_runtime_context(self, topic: str, context: Optional[Dict]) -> Optional[Any]:
        """Check the current runtime context dict."""
        if not context:
            return None
        # Exact match
        if topic in context:
            return context[topic]
        # Case-insensitive search
        topic_lower = topic.lower()
        for key, value in context.items():
            if key.lower() == topic_lower:
                return value
        return None

    def _check_vector_memory(self, topic: str) -> Optional[str]:
        """Query the vector memory store via DI container."""
        try:
            from core.container import ServiceContainer
            vector_mem = ServiceContainer.get("vector_memory", default=None)
            if vector_mem is None:
                return None

            if hasattr(vector_mem, 'retrieve_context'):
                results = vector_mem.retrieve_context(topic, top_k=1)
                if results:
                    # Return the top result content
                    if isinstance(results, list) and len(results) > 0:
                        result = results[0]
                        if isinstance(result, dict):
                            return result.get("content", result.get("text"))
                        return str(result)
                    return str(results)

            if hasattr(vector_mem, 'search'):
                results = vector_mem.search(topic, limit=1)
                if results:
                    return str(results[0]) if isinstance(results, list) else str(results)

        except (KeyError, ImportError):
            import logging
            logger.debug("Exception caught during execution", exc_info=True)
        except Exception as e:
            logger.debug("Vector memory query failed for '%s': %s", topic, e)
        return None


# Service Registration
def register_state_authority():
    """Register the state authority in the global container."""
    ServiceContainer.register(
        "state_authority",
        factory=lambda: StateAuthority(),
        lifetime=ServiceLifetime.SINGLETON
    )


def get_state_authority():
    """Resolve or create state authority lazily."""
    try:
        if not ServiceContainer.get("state_authority", None):
             register_state_authority()
        return ServiceContainer.get("state_authority", default=None)
    except Exception as e:
        logger.debug("ServiceContainer unavailable or failed: %s. Creating standalone StateAuthority.", e)
        return StateAuthority()
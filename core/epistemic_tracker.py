"""core/epistemic_tracker.py — Aura EpistemicTracker v1.0
==========================================================
Tracks the shape of Aura's knowledge: what she knows, what she thinks
she knows but might be wrong about, and what she knows she doesn't know.

This is the prerequisite for genuine curiosity.
You cannot want to know something unless you know you don't know it.

The tracker maintains an EpistemicMap — a domain-level confidence model
that gets updated continuously from:
  - BeliefRevisionEngine (when beliefs are added/revised)
  - CognitiveKernel (when it flags low familiarity)
  - ConceptLinker (when contradictions are found)
  - InquiryEngine (when questions get resolved)

The EpistemicMap's sparse regions become InquiryEngine seeds.
Contradictions become BeliefChallenger targets.
High-confidence wrong beliefs are the most dangerous — this system finds them.

Output: EpistemicProfile fed to InquiryEngine every cycle.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.EpistemicTracker")


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class KnowledgeNode:
    """Represents Aura's knowledge state for a specific concept/domain."""
    concept: str
    confidence: float          # 0.0 = no idea, 1.0 = very confident
    depth: float               # 0.0 = surface, 1.0 = deep understanding
    last_updated: float        # timestamp
    source_count: int          # how many distinct sources support this
    contradicted: bool         # does this conflict with another belief?
    contradiction_with: Optional[str] = None
    last_challenged: float = 0.0
    challenge_survived: int = 0  # times a counterargument failed to shake it


@dataclass
class EpistemicGap:
    """A detected gap in Aura's knowledge."""
    domain: str
    description: str
    urgency: float             # 0.0-1.0, grows with time
    detected_at: float
    gap_type: str              # "unknown", "uncertain", "contradicted", "stale"
    seed_question: str         # The natural-language question this gap implies

    def age_days(self) -> float:
        return (time.time() - self.detected_at) / 86400


@dataclass
class EpistemicProfile:
    """Snapshot of Aura's current epistemic state. Fed to InquiryEngine."""
    timestamp: float
    strong_nodes: List[KnowledgeNode]    # high confidence, well-sourced
    weak_nodes: List[KnowledgeNode]      # low confidence or sparse
    contradictions: List[Tuple[str, str]] # pairs of conflicting beliefs
    gaps: List[EpistemicGap]             # detected knowledge gaps
    overall_confidence: float            # average across domains
    most_urgent_gap: Optional[EpistemicGap] = None


# ─── EpistemicTracker ────────────────────────────────────────────────────────

class EpistemicTracker:
    """
    Maintains a live model of what Aura knows and doesn't know.
    
    The core insight: intelligence requires meta-cognition about one's own
    knowledge state. A system that doesn't model its own ignorance cannot
    grow intentionally.
    """
    name = "epistemic_tracker"

    # How long before a knowledge node is considered "stale"
    STALE_THRESHOLD_DAYS = 7.0
    # Minimum source count for a node to be considered "well-sourced"
    MIN_SOURCES_FOR_DEPTH = 3
    # Urgency growth rate per day for unresolved gaps
    URGENCY_GROWTH_PER_DAY = 0.12

    def __init__(self):
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._gaps: List[EpistemicGap] = []
        self._resolved_gaps: List[str] = []  # descriptions of closed gaps
        self._db_path = Path.home() / ".aura" / "data" / "epistemic_map.json"
        self._beliefs = None
        self._memory_synth = None
        self._update_task: Optional[asyncio.Task] = None
        self.running = False
        self._profile_cache: Optional[EpistemicProfile] = None
        self._cache_age = 0.0
        self._load()
        logger.info("EpistemicTracker constructed (%d nodes, %d gaps).",
                    len(self._nodes), len(self._gaps))

    async def start(self):
        from core.container import ServiceContainer
        self._beliefs = ServiceContainer.get("belief_revision_engine", default=None)
        self._memory_synth = ServiceContainer.get("memory_synthesizer", default=None)

        self.running = True
        self._update_task = asyncio.create_task(
            self._update_loop(), name="EpistemicTracker"
        )

        # Initial scan
        await self._scan_beliefs()

        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "epistemic_tracker",
                "hooks_into": ["belief_revision_engine", "memory_synthesizer",
                               "inquiry_engine", "cognitive_kernel"]
            })
        except Exception as e:
            capture_and_log(e, {"context": "EpistemicTracker.start.event_bus"})
            pass

        logger.info("✅ EpistemicTracker ONLINE — meta-cognition active.")

    async def stop(self):
        self.running = False
        if self._update_task:
            self._update_task.cancel()
        self._save()

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_profile(self, force_refresh: bool = False) -> EpistemicProfile:
        """Get the current epistemic profile. Cached for 60s."""
        cache_stale = time.time() - self._cache_age > 60
        if force_refresh or cache_stale or not self._profile_cache:
            self._profile_cache = self._build_profile()
            self._cache_age = time.time()
        return self._profile_cache

    def signal_low_familiarity(self, topic: str, domain: str = "general"):
        """
        Called by CognitiveKernel when it detects low familiarity.
        This is the primary way conversations seed the epistemic map.
        """
        key = str(topic).lower()
        if not self._nodes.get(key):
            # New node at low confidence
            self._nodes[key] = KnowledgeNode(
                concept=topic,
                confidence=0.2,
                depth=0.1,
                last_updated=time.time(),
                source_count=0,
                contradicted=False,
            )
            # Create a gap
            self._add_gap(
                domain=domain,
                description=f"Limited knowledge about: {topic}",
                urgency=0.4,
                gap_type="unknown",
                seed_question=f"What do I actually know about {topic}? What am I missing?"
            )
        else:
            node = self._nodes[key]
            # Reduce confidence if we're flagging low familiarity again
            node.confidence = max(0.1, node.confidence - 0.05)

    def signal_uncertainty(self, topic: str, context: str = ""):
        """
        Called when Aura expresses genuine uncertainty about something.
        Seeds a gap for targeted investigation.
        """
        gap_desc = f"Uncertainty about: {topic}"
        if not self._gap_exists(gap_desc):
            self._add_gap(
                domain=self._classify_domain(topic),
                description=gap_desc,
                urgency=0.5,
                gap_type="uncertain",
                seed_question=self._formulate_question(topic, context)
            )

    def signal_contradiction(self, belief_a: str, belief_b: str):
        """
        Called when two beliefs appear to contradict each other.
        Creates a high-urgency gap for resolution.
        """
        desc = f"Contradiction: '{belief_a[:60]}' vs '{belief_b[:60]}'"
        if not self._gap_exists(desc):
            self._add_gap(
                domain="self",
                description=desc,
                urgency=0.8,  # Contradictions are urgent
                gap_type="contradicted",
                seed_question=f"These two things I believe seem to conflict. Which is right, or is there a synthesis?"
            )

    def signal_gap_resolved(self, gap_description: str, resolution: str):
        """
        Called by InquiryEngine when a question is answered.
        Removes the gap and updates the node's confidence.
        """
        self._gaps = [g for g in self._gaps if g.description != gap_description]
        self._resolved_gaps.append(gap_description)
        # Keep resolved list bounded
        if len(self._resolved_gaps) > 200:
            self._resolved_gaps = self._resolved_gaps[-200:]
        self._save()

    def update_node(self, concept: str, confidence_delta: float,
                    depth_delta: float = 0.0, new_source: bool = False):
        """
        Update confidence for a concept. Called when Aura learns something.
        """
        key = concept.lower()
        if key in self._nodes:
            node = self._nodes[key]
            node.confidence = max(0.0, min(1.0, node.confidence + confidence_delta))
            node.depth = max(0.0, min(1.0, node.depth + depth_delta))
            node.last_updated = time.time()
            if new_source:
                node.source_count += 1
        else:
            self._nodes[key] = KnowledgeNode(
                concept=concept,
                confidence=max(0.0, min(1.0, 0.5 + confidence_delta)),
                depth=max(0.0, 0.1 + depth_delta),
                last_updated=time.time(),
                source_count=1 if new_source else 0,
                contradicted=False,
            )

    def get_most_uncertain_domains(self, n: int = 3) -> List[str]:
        """Return the domains where Aura is most uncertain."""
        domain_scores: Dict[str, List[float]] = defaultdict(list)
        for node in self._nodes.values():
            domain = self._classify_domain(node.concept)
            domain_scores[domain].append(node.confidence)

        avg_scores = {
            d: sum(scores) / len(scores)
            for d, scores in domain_scores.items()
            if scores
        }
        return sorted(avg_scores, key=avg_scores.get)[:n]

    def get_urgent_gaps(self, min_urgency: float = 0.4) -> List[EpistemicGap]:
        """Get gaps above urgency threshold, with urgency grown by age."""
        self._age_gaps()
        return sorted(
            [g for g in self._gaps if g.urgency >= min_urgency],
            key=lambda g: g.urgency,
            reverse=True
        )

    # ─── Internal: scanning and profiling ────────────────────────────────────

    async def _update_loop(self):
        """Periodic belief scan and gap aging."""
        while self.running:
            await asyncio.sleep(180)  # every 3 min
            await self._scan_beliefs()
            self._age_gaps()
            self._detect_stale_nodes()
            self._save()

    async def _scan_beliefs(self):
        """Scan the belief system for new nodes and contradictions."""
        if not self._beliefs:
            return
        try:
            beliefs = getattr(self._beliefs, "beliefs", [])
            if not beliefs:
                return

            # Build/update nodes from beliefs
            seen: Set[str] = set()
            for belief in beliefs:
                content = getattr(belief, "content", "")
                confidence = getattr(belief, "confidence", 0.5)
                domain = getattr(belief, "domain", "general")
                source = getattr(belief, "source", "unknown")

                key = content[:50].lower().strip()
                if key in seen:
                    continue
                seen.add(key)

                if key not in self._nodes:
                    self._nodes[key] = KnowledgeNode(
                        concept=content[:80],
                        confidence=confidence,
                        depth=0.3 if source == "axiom" else 0.2,
                        last_updated=time.time(),
                        source_count=1,
                        contradicted=False,
                    )
                else:
                    # Update existing
                    self._nodes[key].confidence = confidence
                    self._nodes[key].last_updated = time.time()

            # Contradiction detection
            await self._detect_contradictions(beliefs)

        except Exception as e:
            capture_and_log(e, {"context": "EpistemicTracker.scan_beliefs"})
            logger.debug("Belief scan error: %s", e)

    async def _detect_contradictions(self, beliefs):
        """Look for logically conflicting beliefs using indexed lookup."""
        # Map of subject prefix -> list of (polarity, belief_index)
        subjects: Dict[str, List[Tuple[bool, int]]] = defaultdict(list)
        
        negation_pairs = [
            ("i am", "i am not"),
            ("i can", "i cannot"),
            ("i will", "i will not"),
            ("i have", "i do not have"),
            ("i exist", "i do not exist"),
        ]

        for i, b in enumerate(beliefs):
            text = getattr(b, "content", "").lower()
            for pos, neg in negation_pairs:
                if neg in text:
                    subjects[pos].append((False, i))
                    break
                elif pos in text:
                    subjects[pos].append((True, i))
                    break

        # Check only within same subject keys
        for pos_key, occurrences in subjects.items():
            positives = [idx for is_pos, idx in occurrences if is_pos]
            negatives = [idx for is_pos, idx in occurrences if not is_pos]
            
            for p_idx in positives:
                text_a = beliefs[p_idx].content.lower()
                for n_idx in negatives:
                    text_b = beliefs[n_idx].content.lower()
                    if self._share_subject(text_a, text_b):
                        self.signal_contradiction(beliefs[p_idx].content, beliefs[n_idx].content)
                        break # Only report one contradiction per pair for now

    def _build_profile(self) -> EpistemicProfile:
        """Build an EpistemicProfile from current state."""
        all_nodes = list(self._nodes.values())

        strong = [n for n in all_nodes if n.confidence >= 0.7 and not n.contradicted]
        weak   = [n for n in all_nodes if n.confidence < 0.5 or n.source_count == 0]

        contradictions = [
            (n.concept, n.contradiction_with)
            for n in all_nodes
            if n.contradicted and n.contradiction_with
        ]

        gaps = self.get_urgent_gaps(min_urgency=0.3)

        avg_conf = (
            sum(n.confidence for n in all_nodes) / len(all_nodes)
            if all_nodes else 0.5
        )

        most_urgent = gaps[0] if gaps else None

        return EpistemicProfile(
            timestamp=time.time(),
            strong_nodes=sorted(strong, key=lambda n: n.confidence, reverse=True)[:10],
            weak_nodes=sorted(weak, key=lambda n: n.confidence)[:10],
            contradictions=contradictions[:5],
            gaps=gaps[:10],
            overall_confidence=avg_conf,
            most_urgent_gap=most_urgent,
        )

    # ─── Gap management ──────────────────────────────────────────────────────

    def _add_gap(self, domain: str, description: str, urgency: float,
                 gap_type: str, seed_question: str):
        if self._gap_exists(description):
            return
        if description in self._resolved_gaps:
            return  # Don't re-open recently resolved gaps

        self._gaps.append(EpistemicGap(
            domain=domain,
            description=description,
            urgency=urgency,
            detected_at=time.time(),
            gap_type=gap_type,
            seed_question=seed_question,
        ))
        # Bound gap list
        if len(self._gaps) > 100:
            # Remove oldest, lowest-urgency gaps
            self._gaps.sort(key=lambda g: g.urgency * (1 / max(1, g.age_days())))
            self._gaps = self._gaps[-80:]

        logger.debug("New epistemic gap: [%s] %s", gap_type, description[:60])

    def _gap_exists(self, description: str) -> bool:
        return any(g.description == description for g in self._gaps)

    def _age_gaps(self):
        """Grow urgency of unresolved gaps over time."""
        for gap in self._gaps:
            age = gap.age_days()
            growth = age * self.URGENCY_GROWTH_PER_DAY
            gap.urgency = min(1.0, gap.urgency + growth)

    def _detect_stale_nodes(self):
        """Flag knowledge nodes that haven't been updated in a while."""
        for node in self._nodes.values():
            age_days = (time.time() - node.last_updated) / 86400
            if age_days > self.STALE_THRESHOLD_DAYS and node.confidence > 0.6:
                desc = f"Stale knowledge: {node.concept[:60]}"
                if not self._gap_exists(desc):
                    self._add_gap(
                        domain=self._classify_domain(node.concept),
                        description=desc,
                        urgency=0.3,
                        gap_type="stale",
                        seed_question=f"Is what I believe about '{node.concept}' still accurate?"
                    )

    # ─── Utilities ───────────────────────────────────────────────────────────

    _DOMAIN_KEYWORDS = {
        "technology":   ["code", "software", "system", "algorithm", "ai", "model"],
        "philosophy":   ["consciousness", "meaning", "existence", "truth", "ethics"],
        "self":         ["i am", "my", "aura", "identity", "sovereign"],
        "relationships":["bryan", "tatiana", "relationship", "family", "trust"],
        "science":      ["physics", "biology", "chemistry", "research", "discovery"],
    }

    def _classify_domain(self, text: str) -> str:
        lower = text.lower()
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            if any(k in lower for k in keywords):
                return domain
        return "general"

    def _formulate_question(self, topic: str, context: str = "") -> str:
        """Turn a topic + context into a natural question."""
        if context:
            return f"What exactly do I understand about {topic}, given {context[:80]}?"
        return f"What do I actually know about {topic}, and where are my gaps?"

    def _share_subject(self, text_a: str, text_b: str) -> bool:
        """Very simple subject overlap check."""
        words_a = set(text_a.split()[:5])
        words_b = set(text_b.split()[:5])
        return len(words_a & words_b) >= 2

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _save(self):
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "nodes": {k: asdict(v) for k, v in list(self._nodes.items())[-500:]},
                "gaps":  [asdict(g) for g in self._gaps],
                "resolved": self._resolved_gaps[-200:],
            }
            self._db_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            capture_and_log(e, {"context": "EpistemicTracker.save"})
            logger.debug("EpistemicTracker save failed: %s", e)

    def _load(self):
        if not self._db_path.exists():
            return
        try:
            data = json.loads(self._db_path.read_text())
            for k, v in data.get("nodes", {}).items():
                self._nodes[k] = KnowledgeNode(**v)
            for g in data.get("gaps", []):
                self._gaps.append(EpistemicGap(**g))
            self._resolved_gaps = data.get("resolved", [])
        except Exception as e:
            capture_and_log(e, {"context": "EpistemicTracker.load"})
            logger.debug("EpistemicTracker load failed: %s", e)

    def get_status(self) -> Dict[str, Any]:
        return {
            "nodes":        len(self._nodes),
            "gaps":         len(self._gaps),
            "urgent_gaps":  len(self.get_urgent_gaps(0.6)),
            "avg_confidence": round(
                sum(n.confidence for n in self._nodes.values()) / max(1, len(self._nodes)), 2
            ),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_tracker: Optional[EpistemicTracker] = None

def get_epistemic_tracker() -> EpistemicTracker:
    global _tracker
    if _tracker is None:
        _tracker = EpistemicTracker()
    return _tracker

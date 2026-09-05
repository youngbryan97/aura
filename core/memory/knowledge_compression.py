"""Three-layer knowledge compression for working memory.

Inspired by Concord's DTU (Discrete Thought Unit) system which achieves
33:1 compression by structuring knowledge into three layers:
  1. Readable layer  — human-readable summary (for system prompt injection)
  2. Semantic layer   — key entities, relations, sentiment (for retrieval)
  3. Machine layer    — compact binary/numeric representation (for matching)

Instead of flat text concatenation ("old | new | newer"), this produces
structured knowledge atoms that compress better and retrieve faster.

Usage:
    compressor = KnowledgeCompressor()
    atom = compressor.compress_turns(conversation_turns)
    # atom.readable  → "User discussed moving to Austin. Aura recommended it."
    # atom.semantic  → {"entities": ["Austin", "Denver"], "sentiment": 0.6, ...}
    # atom.machine   → np.array([0.2, 0.8, ...])  # 32-dim compact vector
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.KnowledgeCompression")


@dataclass
class KnowledgeAtom:
    """A compressed unit of conversational knowledge."""
    atom_id: str = ""
    timestamp: float = 0.0

    # Layer 1: Human-readable (for system prompt)
    readable: str = ""

    # Layer 2: Semantic (for retrieval and reasoning)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    sentiment: float = 0.0         # -1 to 1
    user_intent: str = ""          # question, statement, request, emotional
    aura_stance: str = ""          # agreed, disagreed, supported, deflected
    turn_count: int = 0

    # Layer 3: Machine (compact numeric for fast matching)
    vector: Optional[np.ndarray] = None  # 32-dim compact representation

    def to_prompt_text(self) -> str:
        """Render for system prompt injection (Layer 1 + key Layer 2)."""
        parts = [self.readable]
        if self.entities:
            parts.append(f"(Topics: {', '.join(self.entities[:5])})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "timestamp": self.timestamp,
            "readable": self.readable,
            "entities": self.entities,
            "topics": self.topics,
            "sentiment": self.sentiment,
            "user_intent": self.user_intent,
            "aura_stance": self.aura_stance,
            "turn_count": self.turn_count,
        }


class KnowledgeCompressor:
    """Compresses conversation turns into structured knowledge atoms."""

    def __init__(self):
        self._atoms: List[KnowledgeAtom] = []

    def compress_turns(self, turns: List[Dict[str, str]], brain=None) -> KnowledgeAtom:
        """Compress a sequence of conversation turns into a knowledge atom.

        Args:
            turns: List of {"role": "user"|"assistant", "content": "..."} dicts
            brain: Optional LLM for summarization (falls back to extractive)
        """
        if not turns:
            return KnowledgeAtom()

        atom = KnowledgeAtom(
            atom_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            timestamp=time.time(),
            turn_count=len(turns),
        )

        # Extract content
        user_msgs = [t["content"] for t in turns if t.get("role") == "user"]
        aura_msgs = [t["content"] for t in turns if t.get("role") in ("assistant", "aura")]
        all_text = " ".join(msg for msg in user_msgs + aura_msgs if msg)

        # Layer 1: Readable summary
        atom.readable = self._build_readable(user_msgs, aura_msgs, brain)

        # Layer 2: Semantic extraction
        atom.entities = self._extract_entities(all_text)
        atom.topics = self._extract_topics(all_text)
        atom.sentiment = self._compute_sentiment(all_text)
        atom.user_intent = self._classify_intent(user_msgs)
        atom.aura_stance = self._classify_stance(aura_msgs)

        # Layer 3: Machine vector
        atom.vector = self._compute_vector(atom)

        self._atoms.append(atom)
        return atom

    def get_compressed_context(self, max_chars: int = 2000) -> str:
        """Get all atoms as a compressed context string for system prompt."""
        if not self._atoms:
            return ""

        parts = []
        total = 0
        # Most recent atoms first
        for atom in reversed(self._atoms):
            text = atom.to_prompt_text()
            if total + len(text) > max_chars:
                break
            parts.insert(0, text)
            total += len(text)

        return " | ".join(parts) if len(parts) <= 3 else "\n".join(f"- {p}" for p in parts)

    def merge_atoms(self, atoms: List[KnowledgeAtom]) -> KnowledgeAtom:
        """Merge multiple atoms into one (for deeper compression)."""
        if not atoms:
            return KnowledgeAtom()
        if len(atoms) == 1:
            return atoms[0]

        merged = KnowledgeAtom(
            atom_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            timestamp=time.time(),
            turn_count=sum(a.turn_count for a in atoms),
        )

        # Merge readables
        readables = [a.readable for a in atoms if a.readable]
        if len(readables) > 3:
            merged.readable = f"Earlier: {readables[0]} ... Recent: {readables[-1]}"
        else:
            merged.readable = " Then: ".join(readables)

        # Merge entities (deduplicate, keep most frequent)
        all_entities = []
        for a in atoms:
            all_entities.extend(a.entities)
        # Count and sort by frequency
        from collections import Counter
        entity_counts = Counter(all_entities)
        merged.entities = [e for e, _ in entity_counts.most_common(10)]

        # Merge topics
        all_topics = []
        for a in atoms:
            all_topics.extend(a.topics)
        topic_counts = Counter(all_topics)
        merged.topics = [t for t, _ in topic_counts.most_common(5)]

        # Average sentiment
        sentiments = [a.sentiment for a in atoms if a.sentiment != 0]
        merged.sentiment = sum(sentiments) / max(len(sentiments), 1)

        # Average vectors
        vectors = [a.vector for a in atoms if a.vector is not None]
        if vectors:
            merged.vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(merged.vector)
            if norm > 1e-8:
                merged.vector = merged.vector / norm

        return merged

    # ── Layer 1: Readable summary ──────────────────────────────────────

    def _build_readable(self, user_msgs: List[str], aura_msgs: List[str],
                        brain=None) -> str:
        """Build a human-readable summary of the conversation segment."""
        # Extractive: take first sentence of each significant message
        sentences = []
        for msg in user_msgs[:3]:
            first = msg.split(".")[0].strip()
            if len(first) > 10:
                sentences.append(f"User: {first[:100]}")
        for msg in aura_msgs[:2]:
            first = msg.split(".")[0].strip()
            if len(first) > 10:
                sentences.append(f"Aura: {first[:100]}")

        return ". ".join(sentences)[:500] if sentences else "Brief exchange."

    # ── Layer 2: Semantic extraction ───────────────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities and key nouns."""
        import re
        # Capitalized phrases (likely proper nouns)
        caps = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b', text)
        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        # Deduplicate
        seen = set()
        entities = []
        for e in caps + quoted:
            key = e.lower()
            if key not in seen and len(e) > 2:
                seen.add(key)
                entities.append(e)
        return entities[:10]

    def _extract_topics(self, text: str) -> List[str]:
        """Extract likely discussion topics."""
        import re
        # Simple keyword extraction: words that appear multiple times
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        from collections import Counter
        counts = Counter(words)
        # Filter common words
        stopwords = {"that", "this", "with", "from", "have", "been", "they",
                     "what", "about", "would", "could", "should", "there",
                     "their", "which", "some", "just", "like", "more", "your",
                     "into", "also", "very", "much", "than", "then", "when"}
        return [w for w, c in counts.most_common(10) if c >= 2 and w not in stopwords][:5]

    def _compute_sentiment(self, text: str) -> float:
        """Quick sentiment score from keyword matching."""
        positive = {"good", "great", "love", "happy", "awesome", "amazing",
                   "thanks", "cool", "nice", "interesting", "fascinating"}
        negative = {"bad", "terrible", "hate", "sad", "awful", "rough",
                   "frustrated", "angry", "scared", "worried", "lost"}

        words = set(text.lower().split())
        pos_count = len(words & positive)
        neg_count = len(words & negative)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def _classify_intent(self, user_msgs: List[str]) -> str:
        """Classify the dominant user intent."""
        all_text = " ".join(user_msgs).lower()
        if "?" in all_text:
            return "question"
        if any(w in all_text for w in ("please", "can you", "help me", "i need")):
            return "request"
        if any(w in all_text for w in ("feel", "feeling", "sad", "happy", "rough", "scared")):
            return "emotional"
        return "statement"

    def _classify_stance(self, aura_msgs: List[str]) -> str:
        """Classify Aura's dominant stance in the exchange."""
        all_text = " ".join(aura_msgs).lower()
        if any(w in all_text for w in ("agree", "right", "exactly", "yes")):
            return "agreed"
        if any(w in all_text for w in ("disagree", "no", "wrong", "actually")):
            return "disagreed"
        if any(w in all_text for w in ("sorry", "tough", "rough", "sucks")):
            return "supported"
        return "engaged"

    # ── Layer 3: Machine vector ────────────────────────────────────────

    def _compute_vector(self, atom: KnowledgeAtom) -> np.ndarray:
        """Compute a 32-dim compact representation for fast matching."""
        vec = np.zeros(32, dtype=np.float32)

        # Sentiment
        vec[0] = atom.sentiment

        # Intent encoding
        intent_map = {"question": 0.25, "request": 0.5, "emotional": 0.75, "statement": 1.0}
        vec[1] = intent_map.get(atom.user_intent, 0.5)

        # Stance encoding
        stance_map = {"agreed": 0.25, "disagreed": 0.5, "supported": 0.75, "engaged": 1.0}
        vec[2] = stance_map.get(atom.aura_stance, 0.5)

        # Turn density
        vec[3] = min(atom.turn_count / 20.0, 1.0)

        # Entity hash features (distribute entity info across vector dims)
        for i, entity in enumerate(atom.entities[:8]):
            h = int(hashlib.md5(entity.lower().encode()).hexdigest()[:8], 16)
            vec[4 + i] = (h % 1000) / 1000.0

        # Topic hash features
        for i, topic in enumerate(atom.topics[:5]):
            h = int(hashlib.md5(topic.lower().encode()).hexdigest()[:8], 16)
            vec[12 + i] = (h % 1000) / 1000.0

        # Text length features
        vec[17] = min(len(atom.readable) / 500.0, 1.0)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec

    def get_status(self) -> Dict:
        return {
            "atom_count": len(self._atoms),
            "total_turns_compressed": sum(a.turn_count for a in self._atoms),
            "entities_tracked": len(set(e for a in self._atoms for e in a.entities)),
        }


_instance: Optional[KnowledgeCompressor] = None


def get_knowledge_compressor() -> KnowledgeCompressor:
    global _instance
    if _instance is None:
        _instance = KnowledgeCompressor()
    return _instance

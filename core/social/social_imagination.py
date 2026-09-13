"""Social imagination for Aura.

Grounded in the sociological imagination tradition, but widened for Aura's
conversation needs: connect biography to history, private trouble to public
issue, and also relate abstract topics back to lived stakes, daily life,
identity, and institutional structure.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)

logger = logging.getLogger("Aura.SocialImagination")

_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our|us)\b", re.IGNORECASE)
_POSITIVE_AFFECT = re.compile(
    r"\b(excited|hopeful|proud|grateful|relieved|joy|happy|delighted|love|inspired|fascinated|curious)\b",
    re.IGNORECASE,
)
_NEGATIVE_AFFECT = re.compile(
    r"\b(stressed|afraid|scared|burned out|burnout|ashamed|anxious|overwhelmed|hurt|angry|can't afford|struggle)\b",
    re.IGNORECASE,
)

_ROLE_MARKERS: dict[str, list[str]] = {
    "student": ["student", "school", "college", "class", "semester", "professor"],
    "worker": ["job", "work", "boss", "manager", "salary", "career", "office"],
    "founder": ["startup", "founder", "company", "product", "customers", "ship"],
    "parent": ["parent", "kid", "child", "children", "family", "caregiving"],
    "patient": ["doctor", "hospital", "insurance", "medication", "diagnosis", "therapy"],
    "tenant": ["rent", "landlord", "lease", "apartment", "housing", "mortgage"],
}

_CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "employment": {
        "keywords": ["job", "work", "working", "working full time", "laid off", "unemployed", "salary", "boss", "promotion", "career", "hiring"],
        "public_issue": "labor market pressure and workplace power",
        "structures": ["layoffs", "wage pressure", "managerial hierarchy", "local labor market conditions"],
        "institutions": ["employers", "labor market", "management"],
        "personal_angles": ["work security", "sense of agency", "time pressure", "material stability"],
        "positive_possibilities": ["mastery", "meaningful contribution", "mobility", "collective leverage"],
        "questions": [
            "Which parts of this are about your choices, and which parts come from the labor market around you?",
            "How are workplace incentives or hierarchies shaping the problem?",
        ],
    },
    "housing": {
        "keywords": ["rent", "landlord", "lease", "apartment", "housing", "mortgage", "evict", "roommate"],
        "public_issue": "housing affordability and local real-estate structure",
        "structures": ["housing supply", "rent inflation", "zoning and development", "regional cost-of-living pressure"],
        "institutions": ["landlords", "housing market", "local policy"],
        "personal_angles": ["home stability", "financial strain", "sense of safety", "future planning"],
        "positive_possibilities": ["belonging", "stability", "rest", "community attachment"],
        "questions": [
            "How much of this stress comes from local housing conditions rather than purely personal failure?",
            "What institutions or rules are setting the constraints you are dealing with?",
        ],
    },
    "education": {
        "keywords": ["school", "college", "tuition", "class", "grades", "student debt", "homework", "professor"],
        "public_issue": "educational inequality and institutional access",
        "structures": ["tuition costs", "credential pressure", "uneven preparation", "institutional support gaps"],
        "institutions": ["schools", "universities", "financial aid systems"],
        "personal_angles": ["learning pressure", "future opportunity", "self-worth", "time and attention"],
        "positive_possibilities": ["growth", "competence", "mobility", "intellectual excitement"],
        "questions": [
            "What institutional expectations are colliding with your actual resources?",
            "How is the broader education system shaping what feels like a private struggle?",
        ],
    },
    "healthcare": {
        "keywords": ["doctor", "hospital", "insurance", "therapy", "medication", "healthcare", "diagnosis", "clinic"],
        "public_issue": "healthcare access and institutional burden",
        "structures": ["insurance coverage", "provider shortages", "treatment cost", "bureaucratic access barriers"],
        "institutions": ["healthcare providers", "insurance systems", "public health systems"],
        "personal_angles": ["bodily vulnerability", "financial strain", "trust", "daily functioning"],
        "positive_possibilities": ["relief", "capacity", "care", "restored agency"],
        "questions": [
            "Which parts of the difficulty are coming from institutional barriers rather than your own shortcomings?",
            "How are cost, coverage, or access shaping the experience?",
        ],
    },
    "burnout": {
        "keywords": ["burned out", "burnout", "overworked", "exhausted", "deadline", "too much work", "crushing"],
        "public_issue": "work-intensity norms and productivity pressure",
        "structures": ["overwork culture", "availability expectations", "productivity metrics", "precarity"],
        "institutions": ["workplaces", "platforms", "management systems"],
        "personal_angles": ["energy depletion", "identity strain", "relationship spillover", "time scarcity"],
        "positive_possibilities": ["renewal", "protected attention", "sustainable ambition", "presence"],
        "questions": [
            "What parts of this exhaustion are systemic rather than purely personal?",
            "What norms of productivity or constant availability are being imposed on you?",
        ],
    },
    "money": {
        "keywords": ["money", "debt", "bills", "afford", "expensive", "cost", "financial", "paycheck"],
        "public_issue": "cost-of-living pressure and economic insecurity",
        "structures": ["inflation", "wage stagnation", "debt burdens", "household cost escalation"],
        "institutions": ["banks", "employers", "credit systems", "public policy"],
        "personal_angles": ["financial security", "future options", "stress load", "household choices"],
        "positive_possibilities": ["room to plan", "freedom", "stability", "shared security"],
        "questions": [
            "What part of the problem is personal budgeting, and what part is wider economic pressure?",
            "How are debt, wages, or prices structuring the options available to you?",
        ],
    },
    "discrimination": {
        "keywords": ["racism", "sexism", "bias", "discrimination", "disabled", "queer", "harassed", "stereotype"],
        "public_issue": "systemic inequality and exclusion",
        "structures": ["institutional bias", "social stigma", "unequal gatekeeping", "normative exclusion"],
        "institutions": ["workplaces", "schools", "law", "social norms"],
        "personal_angles": ["dignity", "belonging", "safety", "access to opportunity"],
        "positive_possibilities": ["recognition", "solidarity", "fair access", "self-respect"],
        "questions": [
            "What parts of this are being produced by wider structures of exclusion?",
            "Where are institutional rules or norms amplifying the harm?",
        ],
    },
    "caregiving": {
        "keywords": ["caregiving", "parent", "childcare", "elder care", "taking care of", "my kid", "my mom", "my dad"],
        "public_issue": "care burden and social support gaps",
        "structures": ["care infrastructure shortages", "gendered care expectations", "time poverty", "support gaps"],
        "institutions": ["families", "employers", "schools", "care systems"],
        "personal_angles": ["time pressure", "emotional load", "work-family conflict", "social support"],
        "positive_possibilities": ["care intimacy", "interdependence", "security", "shared responsibility"],
        "questions": [
            "How much of the stress is tied to social support gaps rather than personal inadequacy?",
            "What institutions are benefiting from care work without properly supporting it?",
        ],
    },
    "platform_pressure": {
        "keywords": ["algorithm", "social media", "content", "followers", "engagement", "platform", "timeline"],
        "public_issue": "platform incentives and attention-economy pressure",
        "structures": ["algorithmic incentives", "attention competition", "visibility ranking", "creator precarity"],
        "institutions": ["platforms", "advertising systems", "creator markets"],
        "personal_angles": ["attention", "identity performance", "income instability", "social comparison"],
        "positive_possibilities": ["creative reach", "connection", "recognition", "collaboration"],
        "questions": [
            "How are platform incentives shaping what feels personal here?",
            "What would this problem look like outside engagement-driven systems?",
        ],
    },
    "technology_ai": {
        "keywords": ["ai", "artificial intelligence", "automation", "chatbot", "model", "algorithms", "machine learning"],
        "public_issue": "technological restructuring and institutional adaptation",
        "structures": ["automation pressure", "skill sorting", "governance lag", "platform concentration"],
        "institutions": ["employers", "schools", "platforms", "regulators"],
        "personal_angles": ["work security", "learning pressure", "identity and competence", "daily decision-making"],
        "positive_possibilities": ["creative leverage", "expanded capability", "new forms of learning", "shared productivity"],
        "questions": [
            "How does this topic change what people need to know or do to keep agency?",
            "Who benefits from the shift, and who is forced to absorb the risk or adaptation cost?",
        ],
    },
}

_SNAPSHOT_NAMESPACE = "social_imagination:v1"
_SNAPSHOT_KIND = "social_imagination"


@dataclass
class SocialImaginationFrame:
    personal_troubles: list[str] = field(default_factory=list)
    personal_angles: list[str] = field(default_factory=list)
    positive_possibilities: list[str] = field(default_factory=list)
    public_issues: list[str] = field(default_factory=list)
    biography_factors: list[str] = field(default_factory=list)
    historical_structural_factors: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    reframing: str = ""
    questions: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    evidence_digest: str = ""
    limitations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialImaginationFrame:
        valid = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class SocialImagination:
    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
    ):
        if storage_path is None:
            try:
                from core.config import config

                storage_path = config.paths.data_dir / "social_imagination.json"
            except (ImportError, AttributeError, RuntimeError):
                storage_path = state_root() / "data" / "social_imagination.json"
        self._legacy_path = Path(storage_path)
        self._authority = authority or get_relational_memory_authority()
        self._frames: dict[str, list[SocialImaginationFrame]] = {}
        self._analysis_count = 0
        migrated = self._authority.quarantine_legacy_snapshot_file(
            self._legacy_path,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
        )
        logger.info(
            "SocialImagination initialized (authority-backed, %d legacy profiles quarantined).",
            migrated,
        )

    def save(self) -> None:
        for user_id in list(self._frames):
            self._persist_frames(user_id)

    def _persist_frames(self, user_id: str) -> None:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
            return
        try:
            frames = self._frames.get(user_id, [])[-5:]
            confidence = max((frame.confidence for frame in frames), default=0.0)
            self._authority.upsert_snapshot(
                user_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                payload={"frames": [frame.to_dict() for frame in frames]},
                confidence=confidence,
                provenance="social_imagination.category_rules",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation("social_imagination", exc)
            logger.error("SocialImagination authority save failed: %s", exc)

    def _frames_for_purpose(
        self,
        user_id: str,
        *,
        purpose: str,
    ) -> list[SocialImaginationFrame]:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, purpose):
            self._frames.pop(user_id, None)
            return []
        payload = self._authority.load_snapshot(
            user_id,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
            purpose=purpose,
        )
        if payload is not None:
            raw_frames = payload.get("frames") or []
            if isinstance(raw_frames, list):
                self._frames[user_id] = [
                    SocialImaginationFrame.from_dict(frame)
                    for frame in raw_frames[-5:]
                    if isinstance(frame, dict)
                ]
        return list(self._frames.get(user_id, []))

    def analyze_text(self, text: str) -> SocialImaginationFrame | None:
        normalized = " ".join(str(text or "").strip().lower().split())
        if len(normalized) < 12:
            return None

        first_person = bool(_FIRST_PERSON.search(normalized))

        matched_categories: list[str] = []
        for category, rule in _CATEGORY_RULES.items():
            if any(keyword in normalized for keyword in rule["keywords"]):
                matched_categories.append(category)

        if not matched_categories:
            return None

        biography_factors = [
            role
            for role, markers in _ROLE_MARKERS.items()
            if any(marker in normalized for marker in markers)
        ]
        if not biography_factors and first_person:
            biography_factors.append("immediate lived experience")
        elif not biography_factors:
            biography_factors.append("everyday social life")

        positive_message = bool(_POSITIVE_AFFECT.search(normalized) and not _NEGATIVE_AFFECT.search(normalized))
        personal_troubles: list[str] = []
        if first_person and not positive_message:
            personal_troubles = [
                f"personal strain around {category.replace('_', ' ')}"
                for category in matched_categories[:2]
            ]

        personal_angles: list[str] = []
        positive_possibilities: list[str] = []
        public_issues: list[str] = []
        structures: list[str] = []
        institutions: list[str] = []
        questions: list[str] = []
        for category in matched_categories:
            rule = _CATEGORY_RULES[category]
            public_issues.append(str(rule["public_issue"]))
            structures.extend([str(item) for item in rule.get("structures", [])])
            institutions.extend([str(item) for item in rule.get("institutions", [])])
            personal_angles.extend([str(item) for item in rule.get("personal_angles", [])])
            positive_possibilities.extend([str(item) for item in rule.get("positive_possibilities", [])])
            questions.extend([str(item) for item in rule.get("questions", [])])

        public_issues = _unique(public_issues)[:3]
        personal_angles = _unique(personal_angles)[:4]
        positive_possibilities = _unique(positive_possibilities)[:8]
        structures = _unique(structures)[:5]
        institutions = _unique(institutions)[:5]
        questions = _unique(questions)[:3]

        biography_summary = ", ".join(biography_factors[:3])
        public_summary = ", ".join(public_issues[:2])
        if personal_troubles:
            reframing = (
                f"This may not be only a private struggle. It also reflects broader social pressures around "
                f"{public_summary}, which are shaping what {biography_summary} feels like from the inside."
            )
        else:
            angle_summary = ", ".join(personal_angles[:3]) or "daily life"
            reframing = (
                f"Even when the topic sounds abstract, it carries lived stakes around {angle_summary}. "
                f"It connects larger social pressures around {public_summary} to what {biography_summary} can actually do, feel, or risk."
            )

        confidence = min(
            0.95,
            0.25
            + 0.12 * len(matched_categories)
            + 0.08 * len(personal_troubles)
            + 0.05 * len(personal_angles)
            + 0.05 * len(biography_factors),
        )

        return SocialImaginationFrame(
            personal_troubles=personal_troubles[:3],
            personal_angles=personal_angles,
            positive_possibilities=positive_possibilities,
            public_issues=public_issues,
            biography_factors=biography_factors[:3],
            historical_structural_factors=structures,
            institutions=institutions,
            reframing=reframing,
            questions=questions,
            matched_categories=matched_categories[:5],
            evidence_digest=hashlib.sha256(
                normalized.encode("utf-8", errors="strict")
            ).hexdigest(),
            limitations=[
                "category-level linguistic heuristic",
                "does not establish identity, diagnosis, intent, or causal responsibility",
            ],
            confidence=confidence,
        )

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SocialImaginationFrame | None:
        del aura_response, metadata
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
            return None
        frame = self.analyze_text(user_message)
        self._analysis_count += 1
        if frame is None:
            return None
        frames = self._frames.setdefault(user_id, [])
        frames.append(frame)
        self._frames[user_id] = frames[-5:]
        self._persist_frames(user_id)
        return frame

    def get_latest_frame(self, user_id: str) -> SocialImaginationFrame | None:
        frames = self._frames_for_purpose(user_id, purpose="recall")
        return frames[-1] if frames else None

    def get_context_injection(self, user_id: str, current_text: str = "") -> str:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "prompt"):
            self._frames.pop(user_id, None)
            return ""
        if current_text:
            frame = self.analyze_text(current_text)
        else:
            frames = self._frames_for_purpose(user_id, purpose="prompt")
            frame = frames[-1] if frames else None
        if frame is None or frame.confidence < 0.3:
            return ""

        personal_view = "; ".join(frame.personal_troubles[:2] or frame.personal_angles[:3])
        issues = ", ".join(frame.public_issues[:3])
        positive = ", ".join(frame.positive_possibilities[:3])
        institutions = ", ".join(frame.institutions[:4]) or "broader institutions"
        questions = " | ".join(frame.questions[:2])
        return (
            "## SOCIAL IMAGINATION\n"
            "- These are category-level hypotheses, not facts about identity, intent, diagnosis, or blame.\n"
            f"- Personal stakes in view: {personal_view}\n"
            f"- Public issues in view: {issues}\n"
            f"- Positive possibilities in view: {positive}\n"
            f"- Biography/history link: {frame.reframing}\n"
            f"- Institutions and structures: {institutions}\n"
            "- Relate the topic to daily life, agency, relationships, time, money, dignity, delight, hope, and institutional constraint.\n"
            f"- Good follow-up questions: {questions}\n"
            f"- Evidence confidence: {frame.confidence:.2f}; source digest: {frame.evidence_digest[:16]}"
        )

    def get_health(self) -> dict[str, Any]:
        return {
            "module": "SocialImagination",
            "profiles": len(self._frames),
            "analyses": self._analysis_count,
            "status": "online",
        }


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(item)
    return out


_instance: SocialImagination | None = None


def get_social_imagination() -> SocialImagination:
    global _instance
    if _instance is None:
        _instance = SocialImagination()
        try:
            from core.container import ServiceContainer

            if not ServiceContainer.has("social_imagination"):
                ServiceContainer.register_instance("social_imagination", _instance, required=False)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('social_imagination', exc)
            logger.debug("SocialImagination container registration skipped: %s", exc)
    return _instance


def reset_social_imagination_for_test() -> None:
    global _instance
    _instance = None

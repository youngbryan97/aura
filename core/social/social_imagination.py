"""Social imagination for Aura.

Grounded in the sociological imagination tradition, but widened for Aura's
conversation needs: connect biography to history, private trouble to public
issue, and also relate abstract topics back to lived stakes, daily life,
identity, and institutional structure.
"""
from __future__ import annotations


import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.SocialImagination")

_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|our|us)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_POSITIVE_AFFECT = re.compile(
    r"\b(excited|hopeful|proud|grateful|relieved|joy|happy|delighted|love|inspired|fascinated|curious)\b",
    re.IGNORECASE,
)
_NEGATIVE_AFFECT = re.compile(
    r"\b(stressed|afraid|scared|burned out|burnout|ashamed|anxious|overwhelmed|hurt|angry|can't afford|struggle)\b",
    re.IGNORECASE,
)

_ROLE_MARKERS: Dict[str, List[str]] = {
    "student": ["student", "school", "college", "class", "semester", "professor"],
    "worker": ["job", "work", "boss", "manager", "salary", "career", "office"],
    "founder": ["startup", "founder", "company", "product", "customers", "ship"],
    "parent": ["parent", "kid", "child", "children", "family", "caregiving"],
    "patient": ["doctor", "hospital", "insurance", "medication", "diagnosis", "therapy"],
    "tenant": ["rent", "landlord", "lease", "apartment", "housing", "mortgage"],
}

_CATEGORY_RULES: Dict[str, Dict[str, Any]] = {
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


@dataclass
class SocialImaginationFrame:
    personal_troubles: List[str] = field(default_factory=list)
    personal_angles: List[str] = field(default_factory=list)
    positive_possibilities: List[str] = field(default_factory=list)
    public_issues: List[str] = field(default_factory=list)
    biography_factors: List[str] = field(default_factory=list)
    historical_structural_factors: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    reframing: str = ""
    questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialImaginationFrame":
        valid = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class SocialImagination:
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            try:
                from core.config import config

                storage_path = config.paths.data_dir / "social_imagination.json"
            except Exception:
                storage_path = Path.home() / ".aura" / "data" / "social_imagination.json"
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._frames: Dict[str, List[SocialImaginationFrame]] = {}
        self._analysis_count = 0
        self._load()
        logger.info("SocialImagination initialized (%d users loaded).", len(self._frames))

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text())
            for user_id, frames in raw.get("frames", {}).items():
                self._frames[user_id] = [
                    SocialImaginationFrame.from_dict(frame)
                    for frame in (frames or [])
                    if isinstance(frame, dict)
                ]
        except Exception as exc:
            logger.warning("SocialImagination load failed: %s", exc)

    def save(self) -> None:
        try:
            payload = {
                "frames": {
                    user_id: [frame.to_dict() for frame in frames[-5:]]
                    for user_id, frames in self._frames.items()
                }
            }
            tmp = str(self._storage_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self._storage_path)
        except Exception as exc:
            logger.error("SocialImagination save failed: %s", exc)

    def analyze_text(self, text: str) -> Optional[SocialImaginationFrame]:
        normalized = " ".join(str(text or "").strip().lower().split())
        if len(normalized) < 12:
            return None

        first_person = bool(_FIRST_PERSON.search(normalized))

        matched_categories: List[str] = []
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

        personal_troubles: List[str] = []
        for sentence in _SENTENCE_SPLIT.split(str(text or "").strip()):
            sentence_norm = sentence.strip()
            if not sentence_norm:
                continue
            lowered = sentence_norm.lower()
            if first_person and _FIRST_PERSON.search(lowered) and any(
                keyword in lowered for category in matched_categories for keyword in _CATEGORY_RULES[category]["keywords"]
            ) and not (_POSITIVE_AFFECT.search(lowered) and not _NEGATIVE_AFFECT.search(lowered)):
                personal_troubles.append(sentence_norm[:180])

        positive_message = bool(_POSITIVE_AFFECT.search(normalized) and not _NEGATIVE_AFFECT.search(normalized))

        if not personal_troubles and first_person and not positive_message:
            personal_troubles = [
                f"personal strain around {category.replace('_', ' ')}"
                for category in matched_categories[:2]
            ]

        personal_angles: List[str] = []
        positive_possibilities: List[str] = []
        public_issues: List[str] = []
        structures: List[str] = []
        institutions: List[str] = []
        questions: List[str] = []
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
            confidence=confidence,
        )

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SocialImaginationFrame]:
        del aura_response, metadata
        frame = self.analyze_text(user_message)
        self._analysis_count += 1
        if frame is None:
            return None
        frames = self._frames.setdefault(user_id, [])
        frames.append(frame)
        self._frames[user_id] = frames[-5:]
        self.save()
        return frame

    def get_latest_frame(self, user_id: str) -> Optional[SocialImaginationFrame]:
        frames = self._frames.get(user_id) or []
        return frames[-1] if frames else None

    def get_context_injection(self, user_id: str, current_text: str = "") -> str:
        frame = self.analyze_text(current_text) if current_text else self.get_latest_frame(user_id)
        if frame is None or frame.confidence < 0.3:
            return ""

        personal_view = "; ".join(frame.personal_troubles[:2] or frame.personal_angles[:3])
        issues = ", ".join(frame.public_issues[:3])
        positive = ", ".join(frame.positive_possibilities[:3])
        institutions = ", ".join(frame.institutions[:4]) or "broader institutions"
        questions = " | ".join(frame.questions[:2])
        return (
            "## SOCIAL IMAGINATION\n"
            f"- Personal stakes in view: {personal_view}\n"
            f"- Public issues in view: {issues}\n"
            f"- Positive possibilities in view: {positive}\n"
            f"- Biography/history link: {frame.reframing}\n"
            f"- Institutions and structures: {institutions}\n"
            "- Relate the topic to daily life, agency, relationships, time, money, dignity, delight, hope, and institutional constraint.\n"
            f"- Good follow-up questions: {questions}"
        )

    def get_health(self) -> Dict[str, Any]:
        return {
            "module": "SocialImagination",
            "profiles": len(self._frames),
            "analyses": self._analysis_count,
            "status": "online",
        }


def _unique(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(item)
    return out


_instance: Optional[SocialImagination] = None


def get_social_imagination() -> SocialImagination:
    global _instance
    if _instance is None:
        _instance = SocialImagination()
        try:
            from core.container import ServiceContainer

            if not ServiceContainer.has("social_imagination"):
                ServiceContainer.register_instance("social_imagination", _instance, required=False)
        except Exception as exc:
            logger.debug("SocialImagination container registration skipped: %s", exc)
    return _instance

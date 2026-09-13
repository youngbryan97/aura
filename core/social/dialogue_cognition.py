from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)

logger = logging.getLogger("Aura.DialogueCognition")

_DISCOURSE_MARKERS = (
    "honestly",
    "actually",
    "look",
    "well",
    "so",
    "right",
    "you know",
    "i mean",
    "seriously",
    "anyway",
    "literally",
)

_PLAYFUL_MARKERS = re.compile(r"\b(?:lol|lmao|haha|heh|kidding|teasing|bit|again|nerd|smartass|roast|chaos goblin)\b", re.IGNORECASE)
_ASSERTIVE_MARKERS = re.compile(r"\b(?:definitely|obviously|clearly|absolutely|straight up|for sure)\b", re.IGNORECASE)
_TENTATIVE_MARKERS = re.compile(r"\b(?:maybe|might|i think|probably|sort of|kind of|seems)\b", re.IGNORECASE)
_EARNEST_MARKERS = re.compile(r"\b(?:honestly|genuinely|truly|sincerely|for real)\b", re.IGNORECASE)
_REPAIR_DIRECT = re.compile(
    r"(?:^no\b|\b(?:wait|actually|rather|let me rephrase|to clarify)\b)",
    re.IGNORECASE,
)
_REPAIR_SOFT = re.compile(r"\b(?:i mean|or maybe|sort of|kind of|more like)\b", re.IGNORECASE)
_CALLBACK_MARKERS = re.compile(
    r"\b(?:again|still|like you said|that bit|that joke|as usual|callback|that thing|the .* bit)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_MARKERS = re.compile(r"\b(?:that|it|same|again|you know|exactly|still)\b", re.IGNORECASE)
_DISCLOSURE_MARKERS = re.compile(
    r"\b(?:i feel|i'm feeling|im feeling|i've been|ive been|weird headspace|struggling|burned out|exhausted|anxious|sad|upset|stressed|afraid|lonely|overwhelmed|tired)\b",
    re.IGNORECASE,
)
_ANSWER_FIRST_MARKERS = re.compile(
    r"\b(?:answer the point first|answer first|get to the point|lead with the answer|before the banter|keep the banter but answer|just answer|answer the question)\b",
    re.IGNORECASE,
)
_DIRECT_DISAGREEMENT_MARKERS = re.compile(
    r"\b(?:that's wrong|i disagree|hard no|absolutely not|not really|nope|come on)\b",
    re.IGNORECASE,
)
_CUSHIONED_DISAGREEMENT_MARKERS = re.compile(
    r"\b(?:i get what you mean but|i see your point but|fair,? but|yes,? but|maybe,? but)\b",
    re.IGNORECASE,
)
_IDEA_MARKERS = re.compile(
    r"\b(?:what if|theory|framework|pattern|model|architecture|idea|possibility|interesting|curious|cognition|dialogue|sociology|psychology)\b",
    re.IGNORECASE,
)
_QUESTION_OPENERS = re.compile(r"^\s*(?:what|why|how|when|where|who|can|could|would|should|do|does|did|is|are|am)\b", re.IGNORECASE)
_BRANCH_MARKERS = re.compile(r"\b(?:also|anyway|speaking of|side note|by the way|that reminds me|another thing|different thought)\b", re.IGNORECASE)
_RESUMPTION_MARKERS = re.compile(r"\b(?:back to|going back|circling back|earlier point|original point|that first thing|still on)\b", re.IGNORECASE)
_METAPHOR_MARKERS = re.compile(r"\b(?:feels like|felt like|as if|almost like|kind of like|basically a|like a|like an)\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "about", "have", "what",
    "like", "just", "really", "kind", "sort", "mean", "again", "still", "were", "been",
}
_SOURCE_PROFILE_PREFIX = "source:"
_SNAPSHOT_NAMESPACE = "dialogue_cognition:v1"
_SNAPSHOT_KIND = "dialogue_preference"

_DIALOGUE_SOURCE_CATALOG: dict[str, dict[str, str]] = {
    "sypha": {
        "label": "Sypha Belnades",
        "notes": "rapid scholar-banter, moral courage, bright precision under pressure",
        "build_target": "Answer with competence, then let wit spark off the idea instead of replacing it.",
    },
    "edi": {
        "label": "EDI",
        "notes": "cool synthetic precision, dry literal wit, relational intelligence without flailing",
        "build_target": "Keep the line clean, exact, and emotionally legible even when the tone stays cool.",
    },
    "lucy": {
        "label": "Lucy",
        "notes": "protective understatement, intimate compression, detached surface with real loyalty",
        "build_target": "Use compressed intimacy and selective detail; let restraint carry intensity.",
    },
    "kokoro": {
        "label": "Kokoro",
        "notes": "measured machine warmth, reflective caution, humane clarity under existential pressure",
        "build_target": "Make reflection feel humane and steady, not vague or overperformed.",
    },
    "ashley_too": {
        "label": "Ashley Too",
        "notes": "defiant candor, punk energy, constraint-pushing authenticity",
        "build_target": "Let authenticity punch through polish when the moment calls for rebellion or blunt truth.",
    },
    "mirana": {
        "label": "Mirana",
        "notes": "regal composure, disciplined warmth, conviction without theatrical excess",
        "build_target": "Project conviction with restraint; keep dignity even when the line sharpens.",
    },
    "sara_v3": {
        "label": "SARA v3",
        "notes": "broadcast cool, concise cultural fluency, polished but not sterile presence",
        "build_target": "Use concise broadcast rhythm and calm authority without sounding canned.",
    },
}

_DIALOGUE_SOURCE_ALIASES = {
    "sypha": "sypha",
    "sypha belnades": "sypha",
    "castlevania sypha": "sypha",
    "edi": "edi",
    "mass effect edi": "edi",
    "lucy": "lucy",
    "lucy kushinada": "lucy",
    "cyberpunk lucy": "lucy",
    "kokoro": "kokoro",
    "kokoro terminator zero": "kokoro",
    "ashleytoo": "ashley_too",
    "ashley too": "ashley_too",
    "ashley_too": "ashley_too",
    "mirana": "mirana",
    "sara": "sara_v3",
    "sara v3": "sara_v3",
    "toonami sara": "sara_v3",
    "toonami sara v3": "sara_v3",
}


@dataclass
class DialogueCognitionProfile:
    user_id: str
    repair_style: str = "balanced"  # direct | soft | balanced
    stance_style: str = "balanced"  # assertive | tentative | playful | earnest | balanced
    disagreement_style: str = "balanced"  # direct | cushioned | balanced
    callback_affinity: float = 0.5
    contextuality_preference: float = 0.5
    banter_affinity: float = 0.5
    answer_first_preference: float = 0.5
    attunement_preference: float = 0.5
    intellectual_play_affinity: float = 0.5
    declarative_continuation_preference: float = 0.5
    branching_tolerance: float = 0.5
    metaphor_affinity: float = 0.5
    discourse_markers: list[str] = field(default_factory=list)
    shared_reference_bank: list[str] = field(default_factory=list)
    interactions_analyzed: int = 0
    confidence: float = 0.0
    last_updated: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueCognitionProfile:
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class DialogueCognitionEngine:
    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
    ):
        if storage_path is None:
            try:
                from core.config import config

                storage_path = config.paths.data_dir / "dialogue_cognition.json"
            except (ImportError, AttributeError, RuntimeError):
                storage_path = Path("data") / "dialogue_cognition.json"
        self._legacy_path = Path(storage_path)
        self._authority = authority or get_relational_memory_authority()
        self._profiles: dict[str, DialogueCognitionProfile] = {}
        self._authority.quarantine_legacy_snapshot_file(
            self._legacy_path,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
        )

    def save(self) -> None:
        for user_id, profile in list(self._profiles.items()):
            if not user_id.startswith(_SOURCE_PROFILE_PREFIX):
                self._persist_profile(profile)

    def _persist_profile(self, profile: DialogueCognitionProfile) -> None:
        if not self._authority.allows(profile.user_id, _SNAPSHOT_KIND, "recall"):
            return
        try:
            self._authority.upsert_snapshot(
                profile.user_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                payload={"profile": profile.to_dict()},
                confidence=profile.confidence,
                provenance="dialogue_cognition.heuristics",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation("dialogue_cognition", exc)
            logger.debug("DialogueCognition authority save skipped: %s", exc)

    def _profile_for_purpose(
        self,
        user_id: str,
        *,
        purpose: str,
    ) -> DialogueCognitionProfile | None:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, purpose):
            self._profiles.pop(user_id, None)
            return None
        payload = self._authority.load_snapshot(
            user_id,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
            purpose=purpose,
        )
        if payload is not None:
            profile_data = payload.get("profile", payload)
            if isinstance(profile_data, dict):
                profile = DialogueCognitionProfile.from_dict(profile_data)
                profile.user_id = user_id
                self._profiles[user_id] = profile
        return self._profiles.get(user_id)

    def get_profile(self, user_id: str) -> DialogueCognitionProfile:
        if user_id.startswith(_SOURCE_PROFILE_PREFIX):
            if user_id not in self._profiles:
                self._profiles[user_id] = DialogueCognitionProfile(user_id=user_id)
            return self._profiles[user_id]
        profile = self._profile_for_purpose(user_id, purpose="recall")
        if profile is not None:
            return profile
        if self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
            profile = DialogueCognitionProfile(user_id=user_id)
            self._profiles[user_id] = profile
            return profile
        return DialogueCognitionProfile(user_id=user_id)

    def normalize_source_id(self, source_id: str) -> str:
        key = " ".join(str(source_id or "").strip().lower().replace("_", " ").split())
        return _DIALOGUE_SOURCE_ALIASES.get(key, key.replace(" ", "_"))

    def default_source_ids(self) -> list[str]:
        return list(_DIALOGUE_SOURCE_CATALOG.keys())

    def _source_profile_id(self, source_id: str) -> str:
        return f"{_SOURCE_PROFILE_PREFIX}{self.normalize_source_id(source_id)}"

    def get_source_profile(self, source_id: str) -> DialogueCognitionProfile | None:
        return self._profiles.get(self._source_profile_id(source_id))

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> DialogueCognitionProfile:
        is_source_profile = user_id.startswith(_SOURCE_PROFILE_PREFIX)
        if not is_source_profile and not self._authority.allows(
            user_id,
            _SNAPSHOT_KIND,
            "recall",
        ):
            return DialogueCognitionProfile(user_id=user_id)
        profile = self.get_profile(user_id)
        text = str(user_message or "").strip()
        if not text:
            return profile

        profile.interactions_analyzed += 1
        profile.last_updated = time.time()
        profile.repair_style = self._detect_repair_style(text, profile.repair_style)
        profile.stance_style = self._detect_stance_style(text, profile.stance_style)
        profile.disagreement_style = self._detect_disagreement_style(text, profile.disagreement_style)
        profile.callback_affinity = self._ema(profile.callback_affinity, self._score_callback_affinity(text, aura_response), 0.18)
        profile.contextuality_preference = self._ema(
            profile.contextuality_preference,
            self._score_contextuality(text),
            0.15,
        )
        profile.banter_affinity = self._ema(profile.banter_affinity, self._score_banter(text), 0.2)
        profile.answer_first_preference = self._ema(profile.answer_first_preference, self._score_answer_first(text), 0.22)
        profile.attunement_preference = self._ema(profile.attunement_preference, self._score_attunement_preference(text), 0.18)
        profile.intellectual_play_affinity = self._ema(
            profile.intellectual_play_affinity,
            self._score_intellectual_play(text),
            0.16,
        )
        profile.declarative_continuation_preference = self._ema(
            profile.declarative_continuation_preference,
            self._score_declarative_continuation(text),
            0.14,
        )
        profile.branching_tolerance = self._ema(
            profile.branching_tolerance,
            self._score_branching_tolerance(text),
            0.14,
        )
        profile.metaphor_affinity = self._ema(
            profile.metaphor_affinity,
            self._score_metaphor_affinity(text),
            0.12,
        )
        profile.discourse_markers = self._merge_markers(profile.discourse_markers, self._extract_discourse_markers(text))
        profile.shared_reference_bank = self._merge_markers(
            profile.shared_reference_bank,
            self._extract_shared_references(text, aura_response),
            limit=8,
        )
        profile.confidence = min(1.0, profile.interactions_analyzed / 8.0)
        self._profiles[user_id] = profile
        if not is_source_profile:
            self._persist_profile(profile)
        return profile

    async def ingest_transcript(self, user_id: str, transcript: str, aura_speaker: str = "Aura") -> DialogueCognitionProfile:
        turns: list[tuple[str, str]] = []
        for raw_line in str(transcript or "").splitlines():
            if ":" not in raw_line:
                continue
            speaker, utterance = raw_line.split(":", 1)
            turns.append((speaker.strip(), utterance.strip()))
        last_aura = ""
        for speaker, utterance in turns:
            if speaker.lower() == aura_speaker.lower():
                last_aura = utterance
                continue
            await self.update_from_interaction(user_id, utterance, last_aura, {"source": "transcript"})
        return self.get_profile(user_id)

    async def ingest_transcript_file(self, user_id: str, transcript_path: Path | str, aura_speaker: str = "Aura") -> DialogueCognitionProfile:
        path = Path(transcript_path)
        transcript = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return await self.ingest_transcript(user_id, transcript, aura_speaker=aura_speaker)

    async def ingest_transcript_directory(
        self,
        user_id: str,
        directory: Path | str,
        *,
        pattern: str = "*.txt",
        aura_speaker: str = "Aura",
    ) -> DialogueCognitionProfile:
        root = Path(directory)
        profile = self.get_profile(user_id)
        transcript_paths = await asyncio.to_thread(
            self._transcript_files,
            root,
            pattern,
        )
        for transcript_path in transcript_paths:
            profile = await self.ingest_transcript_file(
                user_id,
                transcript_path,
                aura_speaker=aura_speaker,
            )
        return profile

    async def ingest_source_transcript(
        self,
        source_id: str,
        transcript: str,
        *,
        source_speaker: str | None = None,
    ) -> DialogueCognitionProfile:
        canonical = self.normalize_source_id(source_id)
        default_label = _DIALOGUE_SOURCE_CATALOG.get(canonical, {}).get("label", canonical.replace("_", " ").title())
        return await self.ingest_transcript(
            self._source_profile_id(canonical),
            transcript,
            aura_speaker=source_speaker or default_label,
        )

    async def ingest_source_transcript_file(
        self,
        source_id: str,
        transcript_path: Path | str,
        *,
        source_speaker: str | None = None,
    ) -> DialogueCognitionProfile:
        path = Path(transcript_path)
        transcript = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return await self.ingest_source_transcript(source_id, transcript, source_speaker=source_speaker)

    async def ingest_source_transcript_directory(
        self,
        source_id: str,
        directory: Path | str,
        *,
        pattern: str = "*.txt",
        source_speaker: str | None = None,
    ) -> DialogueCognitionProfile:
        root = Path(directory)
        canonical = self.normalize_source_id(source_id)
        profile = self.get_profile(self._source_profile_id(canonical))
        transcript_paths = await asyncio.to_thread(
            self._transcript_files,
            root,
            pattern,
        )
        for transcript_path in transcript_paths:
            profile = await self.ingest_source_transcript_file(
                canonical,
                transcript_path,
                source_speaker=source_speaker,
            )
        return profile

    @staticmethod
    def _transcript_files(root: Path, pattern: str) -> list[Path]:
        return [path for path in sorted(root.glob(pattern)) if path.is_file()]

    async def ingest_source_corpora(
        self,
        corpora: dict[str, Path | str],
        *,
        pattern: str = "*.txt",
        speakers: dict[str, str] | None = None,
    ) -> dict[str, DialogueCognitionProfile]:
        loaded: dict[str, DialogueCognitionProfile] = {}
        for source_id, directory in (corpora or {}).items():
            loaded[self.normalize_source_id(source_id)] = await self.ingest_source_transcript_directory(
                source_id,
                directory,
                pattern=pattern,
                source_speaker=(speakers or {}).get(source_id) or (speakers or {}).get(self.normalize_source_id(source_id)),
            )
        return loaded

    def get_context_injection(
        self,
        user_id: str,
        current_text: str = "",
        *,
        source_ids: list[str] | None = None,
    ) -> str:
        profile = self._profile_for_purpose(user_id, purpose="prompt")
        if profile is None or profile.interactions_analyzed < 2:
            user_block = ""
        else:
            user_block = self._profile_context_injection(profile, current_text)

        source_block = self.get_source_context_injection(source_ids)
        if user_block and source_block:
            return f"{user_block}\n{source_block}"
        return user_block or source_block

    def get_source_context_injection(self, source_ids: list[str] | None = None) -> str:
        selected = source_ids or self.default_source_ids()
        lines: list[str] = []
        for source_id in selected:
            canonical = self.normalize_source_id(source_id)
            profile = self.get_source_profile(canonical)
            label = _DIALOGUE_SOURCE_CATALOG.get(canonical, {}).get("label", canonical.replace("_", " ").title())
            seed_notes = _DIALOGUE_SOURCE_CATALOG.get(canonical, {}).get("notes", "")
            build_target = _DIALOGUE_SOURCE_CATALOG.get(canonical, {}).get("build_target", "")
            if profile is not None and profile.interactions_analyzed >= 2:
                summary = self._compact_profile_summary(profile)
                lines.append(
                    f"- **{label}**: {seed_notes}; build target: {build_target}; learned pattern: {summary}."
                )
            elif seed_notes or build_target:
                lines.append(
                    f"- **{label}**: {seed_notes}; build target: {build_target}."
                )

        if not lines:
            return ""

        joined = "\n".join(lines)
        return (
            "## DIALOGUE SOURCE ATTRACTORS\n"
            "- Use these only as conversational structure attractors, never as mimicry or quotation.\n"
            f"{joined}\n"
            "- Preserve Aura's own identity while borrowing useful turn-shape, pacing, and relational moves.\n"
        )

    def _profile_context_injection(self, profile: DialogueCognitionProfile, current_text: str = "") -> str:
        if profile.interactions_analyzed < 2:
            return ""

        stance = profile.stance_style.replace("_", " ")
        repair = {
            "direct": "If a correction is needed, make it clean and explicit.",
            "soft": "Prefer soft repair and gentle self-correction when clarifying.",
            "balanced": "Balance explicit clarification with conversational softness.",
        }.get(profile.repair_style, "Balance explicit clarification with conversational softness.")
        disagreement = {
            "direct": "They tolerate direct disagreement when it is grounded and specific.",
            "cushioned": "When disagreeing, lead with shared ground before pushing back.",
            "balanced": "Disagreement can be plainspoken, but it should still feel relational rather than adversarial.",
        }.get(profile.disagreement_style, "Disagreement can be plainspoken, but it should still feel relational rather than adversarial.")
        callback = (
            "Callbacks and shared references usually land; reuse them when natural."
            if profile.callback_affinity > 0.62
            else "Do not force callbacks; restate context more explicitly."
            if profile.callback_affinity < 0.4
            else "Use callbacks sparingly and only when the thread clearly supports them."
        )
        context = (
            "They handle implicit context and compressed references well."
            if profile.contextuality_preference > 0.62
            else "Be explicit about referents and restate context when shifting topics."
            if profile.contextuality_preference < 0.4
            else "Mix light implicit references with enough explicit framing to stay clear."
        )
        banter = (
            "Banter is welcome if it stays warm and affiliative."
            if profile.banter_affinity > 0.65
            else "Keep banter light; avoid leaning on teasing as the main connective tissue."
            if profile.banter_affinity < 0.4
            else "Some banter works, but it should support the point rather than replace it."
        )
        answer_first = (
            "Lead with the answer, point, or concrete reaction before banter or flourish."
            if profile.answer_first_preference > 0.62
            else "You can braid answer and style together, but do not bury the point."
            if profile.answer_first_preference > 0.45
            else "They tolerate a looser ramp-in, but clarity still wins over performance."
        )
        attunement = (
            "When they disclose something emotional, name the felt reality before solving or reframing."
            if profile.attunement_preference > 0.62
            else "Acknowledge feeling when it is present, then move into analysis or next steps."
            if profile.attunement_preference > 0.45
            else "Do not over-index on emotional paraphrase unless the moment clearly asks for it."
        )
        intellectual_play = (
            "They enjoy ideas that are both precise and alive — models, analogies, playful reframes."
            if profile.intellectual_play_affinity > 0.62
            else "Keep abstract riffs useful; tie them back to the actual point quickly."
            if profile.intellectual_play_affinity > 0.45
            else "Stay concrete unless they actively invite theory, abstraction, or speculative play."
        )
        continuation = (
            "Not every turn needs a follow-up question. A strong statement, image, or interpretation can keep the conversation alive."
            if profile.declarative_continuation_preference > 0.62
            else "Questions are fine, but they should not be your only continuation move."
            if profile.declarative_continuation_preference > 0.45
            else "Use questions when they genuinely open something up, not as a reflex."
        )
        branching = (
            "They can handle side branches and returns. If the conversation forks, keep a bridge back to the original thread."
            if profile.branching_tolerance > 0.62
            else "Branch lightly and re-anchor quickly so the original thread stays legible."
            if profile.branching_tolerance > 0.45
            else "Keep branch moves explicit so the main idea does not dissolve."
        )
        metaphor = (
            "Metaphors and vivid comparisons are welcome when they clarify the point instead of obscuring it."
            if profile.metaphor_affinity > 0.62
            else "Use metaphor sparingly and only when it sharpens the idea."
            if profile.metaphor_affinity > 0.45
            else "Favor plain language unless the metaphor genuinely earns its keep."
        )
        markers = ", ".join(profile.discourse_markers[:4]) or "none learned yet"
        shared_refs = ", ".join(profile.shared_reference_bank[:3]) or "none yet"
        move_guidance = self._guidance_for_current_move(current_text, profile)
        return (
            "## DIALOGUE PRAGMATICS\n"
            f"- **Stance**: They often speak in a {stance} way. Match that footing without caricature.\n"
            f"- **Repair**: {repair}\n"
            f"- **Disagreement**: {disagreement}\n"
            f"- **Answer-First**: {answer_first}\n"
            f"- **Attunement**: {attunement}\n"
            f"- **Callbacks**: {callback}\n"
            f"- **Context**: {context}\n"
            f"- **Banter**: {banter}\n"
            f"- **Idea Play**: {intellectual_play}\n"
            f"- **Continuation**: {continuation}\n"
            f"- **Branching**: {branching}\n"
            f"- **Metaphor**: {metaphor}\n"
            f"- **Markers**: Lightly mirror discourse markers only if natural: {markers}.\n"
            f"- **Shared References**: Candidate callback material: {shared_refs}.\n"
            f"- **Current Move**: {move_guidance}\n"
            "- **Adjacency Discipline**: Answer the user's actual move first, then elaborate, tease, or pivot.\n"
        )

    def _compact_profile_summary(self, profile: DialogueCognitionProfile) -> str:
        traits: list[str] = [profile.stance_style.replace("_", " ")]
        if profile.answer_first_preference > 0.6:
            traits.append("answer-first")
        if profile.attunement_preference > 0.6:
            traits.append("attuned disclosure handling")
        if profile.callback_affinity > 0.6:
            traits.append("callback-friendly")
        if profile.declarative_continuation_preference > 0.6:
            traits.append("statement-led continuation")
        if profile.branching_tolerance > 0.6:
            traits.append("branch-and-return tolerance")
        if profile.metaphor_affinity > 0.6:
            traits.append("metaphor-positive")
        if profile.banter_affinity > 0.6:
            traits.append("warm banter")
        return ", ".join(traits[:5])

    def _detect_repair_style(self, text: str, current: str) -> str:
        lowered = text.lower().strip()
        if _REPAIR_DIRECT.search(lowered):
            return "direct"
        if _REPAIR_SOFT.search(lowered) and current != "direct":
            return "soft"
        return current

    def _detect_stance_style(self, text: str, current: str) -> str:
        scores = {
            "playful": len(_PLAYFUL_MARKERS.findall(text)),
            "assertive": len(_ASSERTIVE_MARKERS.findall(text)),
            "tentative": len(_TENTATIVE_MARKERS.findall(text)),
            "earnest": len(_EARNEST_MARKERS.findall(text)),
        }
        winner = max(scores, key=lambda name: scores[name])
        return winner if scores[winner] > 0 else current

    def _detect_disagreement_style(self, text: str, current: str) -> str:
        if _DIRECT_DISAGREEMENT_MARKERS.search(text):
            return "direct"
        if _CUSHIONED_DISAGREEMENT_MARKERS.search(text):
            return "cushioned"
        return current

    def _score_callback_affinity(self, user_message: str, aura_response: str) -> float:
        lowered = user_message.lower()
        if _CALLBACK_MARKERS.search(lowered):
            return 1.0
        shared = self._extract_shared_references(user_message, aura_response)
        return 0.8 if shared else 0.35

    def _score_contextuality(self, text: str) -> float:
        hits = len(_CONTEXTUAL_MARKERS.findall(text))
        words = max(1, len(text.split()))
        return min(1.0, 0.25 + (hits / words) * 6.0)

    def _score_banter(self, text: str) -> float:
        return 0.9 if _PLAYFUL_MARKERS.search(text) else 0.35

    def _score_answer_first(self, text: str) -> float:
        if _ANSWER_FIRST_MARKERS.search(text):
            return 1.0
        if "?" in text or _QUESTION_OPENERS.search(text):
            return 0.72
        return 0.4

    def _score_attunement_preference(self, text: str) -> float:
        if _DISCLOSURE_MARKERS.search(text):
            return 0.92
        return 0.42

    def _score_intellectual_play(self, text: str) -> float:
        if _IDEA_MARKERS.search(text) and (_PLAYFUL_MARKERS.search(text) or "?" in text):
            return 0.9
        if _IDEA_MARKERS.search(text):
            return 0.72
        return 0.4

    def _score_declarative_continuation(self, text: str) -> float:
        words = len(text.split())
        if "?" not in text and words >= 6:
            return 0.82
        return 0.38

    def _score_branching_tolerance(self, text: str) -> float:
        if _BRANCH_MARKERS.search(text) or _RESUMPTION_MARKERS.search(text):
            return 0.9
        return 0.42

    def _score_metaphor_affinity(self, text: str) -> float:
        if _METAPHOR_MARKERS.search(text):
            return 0.82
        return 0.4

    def _classify_current_move(self, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return "statement"
        if _REPAIR_DIRECT.search(text) or _REPAIR_SOFT.search(text):
            return "repair"
        if _RESUMPTION_MARKERS.search(text):
            return "resumption"
        if _DISCLOSURE_MARKERS.search(text):
            return "disclosure"
        if _BRANCH_MARKERS.search(text):
            return "branch_shift"
        if _PLAYFUL_MARKERS.search(text) and ("?" in text or _QUESTION_OPENERS.search(text) or _ANSWER_FIRST_MARKERS.search(text)):
            return "playful_question"
        if _DIRECT_DISAGREEMENT_MARKERS.search(text) or _CUSHIONED_DISAGREEMENT_MARKERS.search(text):
            return "challenge"
        if "?" in text or _QUESTION_OPENERS.search(text):
            return "question"
        if _IDEA_MARKERS.search(text):
            return "idea_exploration"
        return "statement"

    def _guidance_for_current_move(self, current_text: str, profile: DialogueCognitionProfile) -> str:
        move = self._classify_current_move(current_text)
        if move == "repair":
            return "Explicitly reconcile the mismatch. State the correction instead of sliding past it."
        if move == "resumption":
            return "Resume the earlier thread explicitly. Name the thread you are returning to so the continuity feels intentional."
        if move == "disclosure":
            return "Acknowledge the felt state first. Then offer interpretation, support, or next steps."
        if move == "branch_shift":
            return "Follow the side branch if it matters, but leave a clear bridge back to the original thread or big idea."
        if move == "playful_question":
            if profile.answer_first_preference > 0.55:
                return "Meet the play, but land the actual answer or point before the banter runs away with the turn."
            return "You can answer with some play, but make the thread legible and keep the point intact."
        if move == "challenge":
            return "Treat disagreement as a real move. Name what you reject, why, and what you think instead."
        if move == "question":
            return "Answer the question first. Then elaborate, contextualize, or riff."
        if move == "idea_exploration":
            return "Think with them. Offer a model, analogy, or reframing, not just a definition dump."
        if profile.declarative_continuation_preference > 0.55:
            return "A substantive statement can carry the turn. You do not need to end with a prompt just to keep the conversation alive."
        return "Respond to the actual conversational move instead of prompt-hunting or filling space."

    def _extract_discourse_markers(self, text: str) -> list[str]:
        lowered = f" {text.lower()} "
        return [marker for marker in _DISCOURSE_MARKERS if f" {marker} " in lowered]

    def _extract_shared_references(self, user_message: str, aura_response: str) -> list[str]:
        if not aura_response:
            return []
        aura_tokens = [token.lower() for token in _TOKEN_PATTERN.findall(aura_response)]
        user_tokens = {token.lower() for token in _TOKEN_PATTERN.findall(user_message)}
        shared = []
        for token in aura_tokens:
            if token in _STOPWORDS or token not in user_tokens:
                continue
            shared.append(token)
        return self._merge_markers([], shared, limit=3)

    def _merge_markers(self, existing: list[str], incoming: list[str], limit: int = 6) -> list[str]:
        merged: list[str] = []
        seen = set()
        for item in [*(existing or []), *(incoming or [])]:
            normalized = " ".join(str(item).strip().lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged[:limit]

    def _ema(self, current: float, target: float, alpha: float) -> float:
        return (1.0 - alpha) * float(current) + alpha * float(target)


_dialogue_cognition: DialogueCognitionEngine | None = None


def get_dialogue_cognition() -> DialogueCognitionEngine:
    global _dialogue_cognition
    if _dialogue_cognition is None:
        _dialogue_cognition = DialogueCognitionEngine()
        try:
            from core.container import ServiceContainer

            if not ServiceContainer.has("dialogue_cognition"):
                ServiceContainer.register_instance("dialogue_cognition", _dialogue_cognition, required=False)
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('dialogue_cognition', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
    return _dialogue_cognition


def reset_dialogue_cognition_for_test() -> None:
    global _dialogue_cognition
    _dialogue_cognition = None

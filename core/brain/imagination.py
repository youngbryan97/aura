"""Bounded imagination workspace for live cognition.

This module gives Aura a general internal place to model "what would this look
like?" without pretending the model is external perception. It is deliberately
free of EXTERNAL side effects: no tool calls, no file writes, no dynamic code,
and no model loads. It is not, however, stateless — imagining and grading
frames mutate the engine's own history, indexes, rates, and learned biases,
which is why that state is lock-guarded rather than described as absent. The output is a compact causal frame that can influence prompt context,
sampling, planning, and metacognition through the normal CognitiveEngine path.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from core.brain.imagination_basis import Basis, describe_bases, meets
from core.runtime.errors import record_degradation

# Requires at least one letter, so "76ers", "401k" and "3d" are subjects
# while bare numbers are not. The old pattern demanded a LEADING letter,
# which silently made every digit-initial topic invisible to imagination —
# "the 76ers roster" could only ever be imagined as "roster".
_WORD_RE = re.compile(r"(?=[a-zA-Z0-9_'-]*[a-zA-Z])[a-zA-Z0-9][a-zA-Z0-9_'-]{2,}")
_VISUAL_RE = re.compile(
    r"\b(look like|visuali[sz]e|imagine|image|picture|scene|sketch|diagram|"
    r"mental model|see it|show me|what would .* look like)\b",
    re.IGNORECASE,
)
_LINGUISTIC_RE = re.compile(
    r"\b(phrase|word|name|sentence|metaphor|analogy|voice|essay|story|poem|"
    r"language|describe|summary)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(what if|what would|would happen|could be|hypothetical|alternate|counterfactual|"
    r"scenario|suppose|if we|if i)\b",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"\b(create|invent|novel|new idea|original|creative|imaginative|emergent|"
    r"connection|synthesize|combine|design|brainstorm)\b",
    re.IGNORECASE,
)
_TOOL_OR_REALITY_RE = re.compile(
    r"\b(open|click|type|write|save|export|search|download|run|execute|install|"
    r"modify|delete|commit|push|send|email|browse|tool|tools|workflow|desktop|"
    r"browser|app|application|external|real-world|real world|visible action)\b",
    re.IGNORECASE,
)



# The pure text/number helpers moved to core/brain/imagination_text.py when
# this module crossed the 2,000-line ceiling. They are imported rather than
# re-implemented, and `imagination_subject` stays exported from here because
# that is where callers reach for it.
from core.brain.imagination_text import (  # noqa: E402
    _UNKNOWN_MEMORY_PRESSURE_PCT,
    _clamp,
    _entropy01,
    _extract_keywords,
    _normalize_text,
    _prompt_safe,
    _safe_float,
    _stable_softmax,
    _top_memory_fragments,
    imagination_subject,
)


#: Every quantity this frame emits, and how it was produced. Nothing here is
#: measured: they are regex hits, keyword counts and fixed coefficients. The
#: map is the honest version of names like ``causal_effects``, which is a
#: live cross-module key and therefore cannot be renamed to say so.
DEFAULT_BASES: dict[str, str] = {
    "salience": Basis.LEXICAL.value,
    "novelty_pressure": Basis.LEXICAL.value,
    "curiosity_pressure": Basis.LEXICAL.value,
    "affective_pressure": Basis.LEXICAL.value,
    "memory_pressure": Basis.LEXICAL.value,
    "verification_pressure": Basis.LEXICAL.value,
    "attractor_state": Basis.LEXICAL.value,
    "eligibility_trace": Basis.LEXICAL.value,
    "causal_effects": Basis.LEXICAL.value,
    "sampling_bias": Basis.LEXICAL.value,
    "routing_bias": Basis.LEXICAL.value,
    "working_memory": Basis.LEXICAL.value,
    "visual_model": Basis.TEMPLATE.value,
    "phrase_model": Basis.TEMPLATE.value,
    "conceptual_bridge": Basis.TEMPLATE.value,
    "mental_canvas": Basis.TEMPLATE.value,
    "associative_links": Basis.TEMPLATE.value,
    "novel_thoughts": Basis.TEMPLATE.value,
    "simulation_steps": Basis.TEMPLATE.value,
    "counterfactuals": Basis.TEMPLATE.value,
    "experiments": Basis.TEMPLATE.value,
    "action_affordances": Basis.TEMPLATE.value,
    "ablation_predictions": Basis.TEMPLATE.value,
}


@dataclass(frozen=True)
class ImaginationFrame:
    frame_id: str
    objective: str
    mode: str
    salience: float
    novelty_pressure: float
    curiosity_pressure: float
    affective_pressure: float
    memory_pressure: float
    verification_pressure: float
    working_memory: dict[str, Any] = field(default_factory=dict)
    attractor_state: dict[str, Any] = field(default_factory=dict)
    eligibility_trace: dict[str, float] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    attention_targets: list[str] = field(default_factory=list)
    visual_model: str = ""
    phrase_model: str = ""
    conceptual_bridge: str = ""
    mental_canvas: dict[str, Any] = field(default_factory=dict)
    associative_links: list[dict[str, str]] = field(default_factory=list)
    novel_thoughts: list[str] = field(default_factory=list)
    simulation_steps: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    action_affordances: list[str] = field(default_factory=list)
    ablation_predictions: dict[str, str] = field(default_factory=dict)
    causal_effects: dict[str, Any] = field(default_factory=dict)
    verification_boundary: str = (
        "This is an internal hypothetical model, not external perception or proof."
    )
    sampling_bias: dict[str, float] = field(default_factory=dict)
    routing_bias: dict[str, bool] = field(default_factory=dict)
    #: What each quantity above rests on. CP126 raised four criticals against
    #: this frame and they are one sentence: the names promise measurement
    #: and the code performs lexical scoring. Renaming is not available —
    #: ``causal_effects`` is read by cognitive_engine, task_decomposer and
    #: cognitive_situation_frame — so the basis travels beside the value and
    #: a reader can ask instead of assuming. See core.brain.imagination_basis.
    bases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BASES))
    #: True when the objective was cut before classification, so a reader
    #: knows a constraint or negation may have fallen off the end
    #: (CP126 ``9d4f7016``).
    objective_truncated: bool = False
    governance: dict[str, Any] = field(
        default_factory=lambda: {
            "advisory_only": True,
            "no_external_effects": True,
            "authority_gateway_required_for_effects": True,
        }
    )
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self, *, compact: bool = False) -> str:
        if self.salience < 0.18:
            return ""
        if compact:
            parts = [
                "## IMAGINATION WORKSPACE",
                "Private hypothetical model; do not claim it as observed reality.",
                f"Mode={self.mode} novelty={self.novelty_pressure:.2f} curiosity={self.curiosity_pressure:.2f} "
                f"memory={self.memory_pressure:.2f} verify={self.verification_pressure:.2f}",
            ]
            if self.visual_model:
                parts.append(f"Imagined visual: {_prompt_safe(self.visual_model, 220)}")
            if self.conceptual_bridge:
                parts.append(f"Connection: {_prompt_safe(self.conceptual_bridge, 220)}")
            if self.attention_targets:
                parts.append("Attention targets: " + ", ".join(
                    _prompt_safe(t, 80) for t in self.attention_targets[:4]))
            return "\n".join(parts) + "\n\n"

        lines = [
            "## IMAGINATION WORKSPACE",
            "- Use this as a private generative scratchpad, not as evidence.",
            f"- Mode: {self.mode} | salience={self.salience:.2f} | novelty={self.novelty_pressure:.2f} | curiosity={self.curiosity_pressure:.2f} | memory={self.memory_pressure:.2f} | verify={self.verification_pressure:.2f}",
        ]
        if self.attention_targets:
            lines.append("- Attention targets: " + ", ".join(
                _prompt_safe(t, 80) for t in self.attention_targets[:5]))
        if self.visual_model:
            lines.append(f"- Imagined visual model: {_prompt_safe(self.visual_model)}")
        canvas = self.mental_canvas if isinstance(self.mental_canvas, dict) else {}
        image_prompt = _prompt_safe(canvas.get("image_prompt"), 260) if canvas else ""
        if image_prompt:
            lines.append(f"- Mental canvas: {image_prompt}")
        if self.phrase_model:
            lines.append(f"- Linguistic model: {_prompt_safe(self.phrase_model)}")
        if self.conceptual_bridge:
            lines.append(f"- Novel connection: {_prompt_safe(self.conceptual_bridge)}")
        if self.novel_thoughts:
            lines.append("- Novel thought candidates: " + " | ".join(
                _prompt_safe(t) for t in self.novel_thoughts[:3]))
        if self.associative_links:
            rendered_links = [
                f"{_prompt_safe(link.get('source'), 80)} -> "
                f"{_prompt_safe(link.get('relation'), 80)} -> "
                f"{_prompt_safe(link.get('target'), 80)}"
                for link in self.associative_links[:3]
                if isinstance(link, dict)
            ]
            if rendered_links:
                lines.append("- Association map: " + " | ".join(rendered_links))
        if self.counterfactuals:
            lines.append("- Counterfactual probes: " + " | ".join(
                _prompt_safe(c) for c in self.counterfactuals[:3]))
        if self.simulation_steps:
            lines.append("- Internal simulation steps: " + " | ".join(
                _prompt_safe(s) for s in self.simulation_steps[:3]))
        if self.attractor_state:
            selected = _prompt_safe(self.attractor_state.get("selected"), 80)
            entropy = _safe_float(self.attractor_state.get("entropy"), 0.0)
            margin = _safe_float(self.attractor_state.get("stability_margin"), 0.0)
            if selected:
                lines.append(
                    f"- Attractor: {selected} | entropy={entropy:.2f} | stability_margin={margin:.2f}"
                )
        if self.working_memory:
            admission = str(self.working_memory.get("admission") or "admit")
            queue_load = _safe_float(self.working_memory.get("queue_load"), 0.0)
            overload = _safe_float(self.working_memory.get("overload_pressure"), 0.0)
            if admission != "admit" or overload >= 0.20:
                lines.append(
                    f"- Working-memory gate: {admission} | queue_load={queue_load:.2f} | overload={overload:.2f}"
                )
        if self.experiments:
            lines.append("- Useful next experiments: " + " | ".join(
                _prompt_safe(e) for e in self.experiments[:3]))
        if self.action_affordances:
            lines.append("- Action affordances: " + " | ".join(
                _prompt_safe(a) for a in self.action_affordances[:3]))
        if self.causal_effects:
            effects = []
            for key in (
                "attention_focus",
                "memory_priority",
                "verification_pressure",
                "metacognition_depth",
                "tool_governance",
            ):
                if key in self.causal_effects:
                    effects.append(f"{key}={_prompt_safe(self.causal_effects.get(key), 80)}")
            if effects:
                lines.append("- Causal effects: " + " | ".join(effects))
        lines.append("- Boundary: if real-world facts, files, tools, or screen state matter, verify through governed tools before claiming completion.")
        return "\n".join(lines) + "\n\n"


@dataclass(frozen=True)
class _ThoughtMove:
    """One SHAPE of thinking, not one sentence.

    LIVE FEEDBACK, 2026-07-25. Bryan: "The novel thoughts feature is cool but
    it always breaks things down in the same way 'what if - is not the
    object' and the others."

    He was right, and the cause was structural rather than stylistic. Novel
    thoughts were two f-strings with the top two keywords slotted in, so
    every frame Aura ever produced was the same two cognitive moves wearing
    different nouns. Varying the wording would not have fixed that — the
    thinking was identical.

    A move is a distinct operation on an idea: invert it, rescale it, remove
    it, ask who is absent from it, ask what it costs. Which moves fire is
    decided by Aura's measured internal state at that moment — curiosity,
    confusion, tension, novelty and verification pressure — so the same
    sentence asked while confused and while curious genuinely thinks about
    it differently. Recently used moves are suppressed, so consecutive
    frames do not converge on a house style.

    ``affinity`` returns a bare score; ``render`` returns the sentence.
    Keeping them separate is what lets the selector reason about the move
    without committing to its words.
    """

    move_id: str
    render: Any  # (focus, secondary, signals) -> str
    affinity: Any  # (signals) -> float
    # Moves needing a real second keyword; with one noun they degrade into
    # the "its constraint" filler that made the old output feel canned.
    needs_secondary: bool = False


def _sig(signals: dict[str, float], key: str) -> float:
    return _safe_float(signals.get(key), 0.0)


# Novel-thought moves. Each is a different way of turning an idea over, and
# each says why it belongs to the state that selects it.
_NOVEL_MOVES: tuple[_ThoughtMove, ...] = (
    _ThoughtMove(
        "reframe_as_lens",
        lambda f, s, _: f"What if {f} is not the object, but the lens for seeing {s}?",
        # The original move. Kept, because it is genuinely good when there
        # are two things to hold at once — just no longer mandatory.
        lambda g: 0.34 + _sig(g, "novelty") * 0.30,
        needs_secondary=True,
    ),
    _ThoughtMove(
        "smallest_testable",
        lambda f, s, _: (
            f"The useful novelty may be the smallest testable form of {f}, "
            "not the largest imagined one."
        ),
        lambda g: 0.30 + _sig(g, "verification") * 0.44,
    ),
    _ThoughtMove(
        "opposing_pressure",
        lambda f, s, _: (
            f"Combine {f} with an opposing pressure and look for the behavior "
            "neither has alone."
        ),
        lambda g: 0.20 + _sig(g, "creative") * 0.42 + _sig(g, "tension") * 0.22,
    ),
    _ThoughtMove(
        "invert_premise",
        lambda f, s, _: f"Invert the premise: what would make {f} fail gracefully?",
        lambda g: 0.22 + _sig(g, "counterfactual") * 0.40,
    ),
    _ThoughtMove(
        "rotate_memory",
        lambda f, s, _: (
            "Use recent continuity as material, then deliberately rotate it "
            "into a new frame."
        ),
        lambda g: (0.30 + _sig(g, "novelty") * 0.20) if _sig(g, "memories") else 0.0,
    ),
    _ThoughtMove(
        "missing_actor",
        lambda f, s, _: f"Who or what is absent from this picture of {f}, and is that why it holds?",
        lambda g: 0.26 + _sig(g, "confusion") * 0.34,
    ),
    _ThoughtMove(
        "scale_shift",
        lambda f, s, _: f"Look at {f} ten times larger and ten times smaller — which claims survive both?",
        lambda g: 0.24 + _sig(g, "novelty") * 0.26,
    ),
    _ThoughtMove(
        "boundary_probe",
        lambda f, s, _: f"Find where {f} stops being true; the edge is more informative than the middle.",
        lambda g: 0.26 + _sig(g, "verification") * 0.30 + _sig(g, "confusion") * 0.18,
    ),
    _ThoughtMove(
        "substrate_swap",
        lambda f, s, _: f"If {f} were made of something else entirely, what would stay recognizable?",
        lambda g: 0.18 + _sig(g, "creative") * 0.34,
    ),
    _ThoughtMove(
        "temporal_shift",
        lambda f, s, _: f"What did {f} look like before it had a name, and what will it look like once it is ordinary?",
        lambda g: 0.20 + _sig(g, "curiosity") * 0.30,
    ),
    _ThoughtMove(
        "second_order",
        lambda f, s, _: f"What does {f} cause that then comes back and changes {f}?",
        lambda g: 0.22 + _sig(g, "counterfactual") * 0.24 + _sig(g, "novelty") * 0.18,
    ),
    _ThoughtMove(
        "name_the_cost",
        lambda f, s, _: f"Name what {f} costs that nobody is counting yet.",
        lambda g: 0.24 + _sig(g, "tension") * 0.36,
    ),
    _ThoughtMove(
        "unsympathetic_observer",
        lambda f, s, _: f"How does {f} look to someone who does not want it to work?",
        lambda g: 0.20 + _sig(g, "tension") * 0.30 + _sig(g, "verification") * 0.20,
    ),
    _ThoughtMove(
        "compression",
        lambda f, s, _: f"If everything about {f} were lost except one sentence, which sentence?",
        lambda g: 0.24 + _sig(g, "linguistic") * 0.38,
    ),
    _ThoughtMove(
        "analogy_reach",
        lambda f, s, _: f"Some unrelated system already solved the shape of {f} — which one, and what did it give up?",
        lambda g: 0.20 + _sig(g, "creative") * 0.28 + _sig(g, "curiosity") * 0.22,
    ),
    _ThoughtMove(
        "already_broken",
        lambda f, s, _: f"Assume {f} is already broken and I have not noticed — what would be true?",
        lambda g: 0.20 + _sig(g, "confusion") * 0.30 + _sig(g, "verification") * 0.24,
    ),
    _ThoughtMove(
        "absent_evidence",
        lambda f, s, _: f"What should I be seeing if {f} were true, that I am not seeing?",
        lambda g: 0.22 + _sig(g, "verification") * 0.40,
    ),
    _ThoughtMove(
        "trade_roles",
        lambda f, s, _: f"Let {s} be the cause and {f} the symptom, and see if the story still reads.",
        lambda g: 0.22 + _sig(g, "counterfactual") * 0.26,
        needs_secondary=True,
    ),
)


# Counterfactual probes. Same machinery, different job: these interrogate a
# premise rather than turn it over.
_COUNTERFACTUAL_MOVES: tuple[_ThoughtMove, ...] = (
    _ThoughtMove(
        "constraint_not_feature",
        lambda f, s, _: f"What changes if {f} is treated as a constraint rather than a feature?",
        lambda g: 0.30 + _sig(g, "novelty") * 0.20,
    ),
    _ThoughtMove(
        "smallest_observable",
        lambda f, s, _: f"What would the smallest observable version of {f} be?",
        lambda g: 0.28 + _sig(g, "verification") * 0.34,
    ),
    _ThoughtMove(
        "role_trade",
        lambda f, s, _: f"What if {f} and {s} trade roles?",
        lambda g: 0.26 + _sig(g, "creative") * 0.22,
        needs_secondary=True,
    ),
    _ThoughtMove(
        "after_receipts",
        lambda f, s, _: "What remains true after real tool receipts or external verification?",
        lambda g: (0.34 + _sig(g, "verification") * 0.30) if _sig(g, "tool_or_reality") else 0.0,
    ),
    _ThoughtMove(
        "falsifier",
        lambda f, s, _: "What failure would falsify the imagined model?",
        lambda g: 0.24 + _sig(g, "negation") * 0.40 + _sig(g, "verification") * 0.18,
    ),
    _ThoughtMove(
        "remove_it",
        lambda f, s, _: f"Remove {f} entirely — what still stands, and what quietly collapses?",
        lambda g: 0.24 + _sig(g, "counterfactual") * 0.26,
    ),
    _ThoughtMove(
        "reverse_causality",
        lambda f, s, _: f"What if the arrow runs backwards and {f} is the consequence?",
        lambda g: 0.22 + _sig(g, "counterfactual") * 0.30 + _sig(g, "confusion") * 0.16,
    ),
    _ThoughtMove(
        "adversary",
        lambda f, s, _: f"What would someone hostile do with {f} first?",
        lambda g: 0.18 + _sig(g, "tension") * 0.34,
    ),
    _ThoughtMove(
        "unobserved",
        lambda f, s, _: f"What is true about {f} when nobody is measuring it?",
        lambda g: 0.18 + _sig(g, "curiosity") * 0.30,
    ),
    _ThoughtMove(
        "cost_of_being_right",
        lambda f, s, _: f"If I am right about {f}, what does that cost — and am I willing to pay it?",
        lambda g: 0.18 + _sig(g, "tension") * 0.26 + _sig(g, "verification") * 0.18,
    ),
    _ThoughtMove(
        "already_solved",
        lambda f, s, _: f"What if {f} is already solved and I would not recognize the solution?",
        lambda g: 0.18 + _sig(g, "curiosity") * 0.26 + _sig(g, "confusion") * 0.22,
    ),
    _ThoughtMove(
        "double_it",
        lambda f, s, _: f"Double {f} and halve it — which direction breaks first?",
        lambda g: 0.20 + _sig(g, "novelty") * 0.22,
    ),
)


#: The weakest evidence that may durably change what gets selected. A
#: caller's word steers this session and dies with it; changing tomorrow's
#: selection takes a reading (CP126 ``04a745b8``).
LEARNING_EVIDENCE_FLOOR = Basis.MEASURED

#: Subject partition for learning state. "" is a real key, not a wildcard:
#: an unattributed reward teaches the unattributed subject and nobody else
#: (CP126 ``f1ef7cfb``).
_ANONYMOUS_SUBJECT = "anonymous"


def _subject_key(subject: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(subject or "")).strip("_")
    return cleaned[:64] or _ANONYMOUS_SUBJECT


#: A word has to be worth this much before an imagined subject counts as a
#: physical thing. An unambiguous engineering noun or a real material is
#: worth two on its own; a word that names a part in engineering and
#: something ordinary elsewhere — light, model, frame — is worth one, so it
#: takes two of them. A poem about light stays a poem.
_ENGINEERABLE_THRESHOLD = 2


def _engineerable_score(focus: list[str]) -> tuple[int, str]:
    """How much of an imagined subject could be DRAWN rather than pictured.

    Asked against the live registries rather than a word list kept here: the
    engineering object class, the materials table, the solid kinds and the
    part tags the layout understands. That keeps this true when a material
    or a shape is added, and stops it claiming a subject that was removed.
    """
    try:
        from core.engineering.geometry import SOLID_KINDS
        from core.engineering.layout import _ENCLOSURE_TAGS, _EXTERNAL_TAGS
        from core.engineering.materials import closest_material
        from core.intent.declared_capability import object_class_of
    except ImportError:
        return (0, "")

    strong = set(object_class_of("schematic")) | set(SOLID_KINDS)
    weak = set(_EXTERNAL_TAGS) | set(_ENCLOSURE_TAGS)
    score = 0
    best = ""
    for word in focus[:5]:
        lowered = str(word or "").strip().lower()
        if len(lowered) < 3:
            continue
        if lowered in strong or closest_material(lowered) is not None:
            score += 2
            best = best or lowered
        elif lowered in weak:
            score += 1
            best = best or lowered
    return (score, best)


def _externalization_path(focus: list[str], text: str) -> str:
    """How this canvas would be made visible, if anybody asked to see it.

    A physical or electrical subject externalises as a schematic, because a
    schematic is computed from a model and can be measured off. Anything
    else externalises as an image. Getting that the wrong way round produces
    a picture of a machine that cannot work, which is the failure the
    engineering module exists to prevent.
    """
    score, subject = _engineerable_score(focus)
    if score >= _ENGINEERABLE_THRESHOLD and subject:
        return (
            f"If the user asks to see this, {subject} is a physical thing, so the "
            "honest externalisation is a computed schematic through design_engineering "
            "rather than a generated image. Otherwise keep it private."
        )
    return (
        "If the user asks to see or generate this, request governed image/tool execution; "
        "otherwise keep it private."
    )


class ImaginationEngine:
    """Generator of bounded internal imagination frames.

    Honest about its own effects: producing a frame is free of EXTERNAL side
    effects — nothing is written, sent, or executed — but it is not stateless.
    ``imagine`` and ``learn_from_feedback`` mutate history, the frame index,
    queue state, rates, attractor bias, eligibility traces, outcomes, and
    counters. That state is shared, so it is guarded (``_state_lock``) rather
    than described as absent.
    """

    def __init__(self, *, history_limit: int = 64):
        # Guards every mutation of the shared runtime/learning state below.
        # Without it, concurrent imagine/feedback calls lost updates, replaced
        # each other's frames in the index, and produced inconsistent snapshots.
        self._state_lock = threading.RLock()
        #: Monotonic per-engine frame counter, part of the frame-id receipt so
        #: two materially identical requests still get distinct ids.
        self._frame_seq = 0
        self._history: deque[ImaginationFrame] = deque(maxlen=max(8, history_limit))
        self._frame_index: dict[str, ImaginationFrame] = {}
        self._outcomes: deque[dict[str, Any]] = deque(maxlen=max(8, history_limit))
        self._frame_count = 0
        self._queue_load = 0.0
        self._arrival_rate_ema = 0.0
        self._service_rate_ema = 1.0
        self._last_observed_at = time.monotonic()
        # Keyed by SUBJECT. One flat map meant one person's rewards moved
        # every later person's selection probabilities (CP126 ``f1ef7cfb``).
        self._attractor_bias: dict[str, dict[str, float]] = {}
        self._eligibility_trace: dict[str, dict[str, float]] = {}
        # Which thought SHAPES were used recently, newest last. This is what
        # stops consecutive frames converging on a house style; see
        # _ThoughtMove. Sized to roughly two frames' worth of moves so a
        # shape can come back once the mind has moved on, but not twice in a
        # row.
        self._recent_moves: deque[str] = deque(maxlen=10)

    def imagine(
        self,
        objective: Any,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> ImaginationFrame:
        # What she imagines about is the subject, not the scaffolding that
        # happens to be wrapped around it.
        raw_subject = imagination_subject(objective, context)
        # CP126 ``9d4f7016``: the objective was cut to 500 characters before
        # classification and again to 180 for storage, so a constraint, a
        # negation or a safety boundary written past the limit could not
        # reach the frame and nothing said it had been dropped.
        text = _normalize_text(raw_subject, 500)
        objective_truncated = len(str(raw_subject or '')) > 500
        subject_key = _subject_key(
            (context or {}).get('subject') or (context or {}).get('user_id')
            if isinstance(context, dict) else ''
        )
        lowered = text.lower()
        keywords = _extract_keywords(text)
        memories = _top_memory_fragments(state)

        affect = getattr(state, "affect", None)
        emotions = getattr(affect, "emotions", {}) if affect is not None else {}
        if not isinstance(emotions, dict):
            emotions = {}

        curiosity = _clamp(
            max(
                _safe_float(getattr(affect, "curiosity", 0.0), 0.0),
                _safe_float(emotions.get("curiosity"), 0.0),
                _safe_float(emotions.get("wonder"), 0.0),
                _safe_float(emotions.get("interest"), 0.0),
            )
        )
        confusion = _clamp(
            max(
                _safe_float(emotions.get("confused"), 0.0),
                0.35 if any(token in lowered for token in ("confused", "unclear", "perplexed")) else 0.0,
            )
        )
        tension = _clamp(
            max(
                _safe_float(emotions.get("frustration"), 0.0),
                _safe_float(emotions.get("upset"), 0.0),
                _safe_float(emotions.get("dread"), 0.0),
            )
        )
        affective_pressure = _clamp((confusion * 0.45) + (tension * 0.25) + (curiosity * 0.30))

        visual = bool(_VISUAL_RE.search(text))
        linguistic = bool(_LINGUISTIC_RE.search(text))
        counterfactual = bool(_COUNTERFACTUAL_RE.search(text))
        creative = bool(_CREATIVE_RE.search(text))
        tool_or_reality = bool(_TOOL_OR_REALITY_RE.search(text))
        explicit_request = visual or linguistic or counterfactual or creative
        context_pressure = 0.0
        if isinstance(context, dict):
            context_pressure = 0.12 if context.get("desktop_cognitive_engine_required") else 0.0
            if context.get("creative_mode") or context.get("imagination_requested"):
                context_pressure += 0.25

        novelty_pressure = _clamp(
            (0.30 if creative else 0.0)
            + (0.22 if visual else 0.0)
            + (0.18 if counterfactual else 0.0)
            + (0.12 if linguistic else 0.0)
            + context_pressure
            + (0.16 if len(keywords) >= 4 else 0.0)
        )
        salience = _clamp(
            (0.42 if explicit_request else 0.10)
            + (curiosity * 0.22)
            + (affective_pressure * 0.18)
            + (0.12 if memories else 0.0)
            - (0.10 if is_background else 0.0)
        )
        memory_pressure = _clamp(
            (salience * 0.32)
            + (novelty_pressure * 0.26)
            + (curiosity * 0.18)
            + (affective_pressure * 0.14)
            + (0.10 if memories else 0.0)
        )
        verification_pressure = _clamp(
            (0.46 if tool_or_reality else 0.0)
            + (0.20 if counterfactual else 0.0)
            + (0.16 if confusion >= 0.25 else 0.0)
            + (novelty_pressure * 0.12)
            + (0.06 if visual else 0.0)
        )
        working_memory = self._observe_working_memory_gate(
            salience=salience,
            novelty_pressure=novelty_pressure,
            verification_pressure=verification_pressure,
            memory_pressure=memory_pressure,
            context=context,
            is_background=is_background,
        )
        admission = str(working_memory.get("admission") or "admit")
        if admission == "defer_background":
            salience = _clamp(salience * 0.55)
            novelty_pressure = _clamp(novelty_pressure * 0.70)
            memory_pressure = _clamp(memory_pressure * 0.70)
        elif admission in {"compress_foreground", "thin_frame"}:
            salience = _clamp(salience * 0.86)
            novelty_pressure = _clamp(novelty_pressure * 0.82)
            memory_pressure = _clamp(memory_pressure * 0.82)
            verification_pressure = max(verification_pressure, 0.35)

        modalities: list[str] = []
        if visual:
            modalities.append("visual")
        if linguistic:
            modalities.append("linguistic")
        if counterfactual:
            modalities.append("counterfactual")
        if creative or novelty_pressure >= 0.35:
            modalities.append("conceptual")
        if not modalities and salience >= 0.26:
            modalities.append("associative")

        visual_model = self._build_visual_model(keywords, memories, text) if visual or novelty_pressure >= 0.38 else ""
        phrase_model = self._build_phrase_model(keywords, text) if linguistic or creative else ""
        conceptual_bridge = self._build_conceptual_bridge(keywords, memories, text)
        mental_canvas = self._build_mental_canvas(
            keywords,
            memories,
            text,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
            creative=creative,
        )
        associative_links = self._build_associative_links(keywords, memories, text)
        thought_signals = self._thought_signals(
            curiosity=curiosity,
            confusion=confusion,
            tension=tension,
            novelty_pressure=novelty_pressure,
            verification_pressure=verification_pressure,
            memories=memories,
            creative=creative,
            counterfactual=counterfactual,
            linguistic=linguistic,
            tool_or_reality=tool_or_reality,
            text=text,
        )
        novel_thoughts = self._build_novel_thoughts(
            keywords,
            memories,
            text,
            signals=thought_signals,
            context=context,
        )
        simulation_steps = self._build_simulation_steps(
            keywords,
            visual=visual,
            counterfactual=counterfactual,
            tool_or_reality=tool_or_reality,
        )
        counterfactuals = self._build_counterfactuals(
            keywords, text, signals=thought_signals, context=context,
        )
        experiments = self._build_experiments(keywords, tool_or_reality)
        attention_targets = self._build_attention_targets(keywords, text)
        action_affordances = self._build_action_affordances(
            keywords,
            tool_or_reality=tool_or_reality,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
        )
        ablation_predictions = self._build_ablation_predictions(
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            novelty_pressure=novelty_pressure,
            tool_or_reality=tool_or_reality,
        )
        causal_effects = self._build_causal_effects(
            attention_targets=attention_targets,
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            curiosity=curiosity,
            confusion=confusion,
            tool_or_reality=tool_or_reality,
            working_memory=working_memory,
        )

        mode = "visual_simulation" if visual else "counterfactual" if counterfactual else "creative_synthesis" if creative else "associative"
        attractor_state = self._select_attractor_state(
            mode=mode,
            salience=salience,
            novelty_pressure=novelty_pressure,
            curiosity=curiosity,
            affective_pressure=affective_pressure,
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            working_memory=working_memory,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
            creative=creative,
            subject=subject_key,
            tool_or_reality=tool_or_reality,
        )
        eligibility_trace = self._update_eligibility_trace(
            keywords,
            selected_attractor=str(attractor_state.get("selected") or mode),
            salience=salience,
            novelty_pressure=novelty_pressure,
            memory_pressure=memory_pressure,
        )
        # A frame id is a RECEIPT: feedback addresses an episode by it. Hashing
        # only text/keywords/origin/memories let materially DIFFERENT frames —
        # different mode, admission, attractor, affect, or background status —
        # collide onto one index entry, so a reward could be applied to the
        # wrong episode and silently overwrite its predecessor.
        #
        # The id stays a pure CONTENT hash (no clock, no counter): replaying the
        # same objective in the same state must reproduce the same id, which is
        # what makes the receipt idempotent. What changed is that the content
        # now includes the state that actually distinguishes one simulation
        # from another.
        seed = "|".join((
            str(text), str(keywords), str(origin), str(memories), str(mode),
            str(admission), str(attractor_state.get("selected") or ""),
            f"{salience:.4f}", f"{novelty_pressure:.4f}", f"{memory_pressure:.4f}",
            f"{verification_pressure:.4f}", str(bool(is_background)),
        )).encode("utf-8", errors="ignore")
        frame_id = hashlib.sha256(seed).hexdigest()[:16]
        token_factor = 1.0 + min(0.12, salience * 0.10)
        if admission in {"compress_foreground", "thin_frame"}:
            token_factor = min(token_factor, 0.92)
        if admission == "defer_background":
            token_factor = min(token_factor, 0.70)
        sampling_bias = {
            "temperature_delta": round(min(0.12, novelty_pressure * 0.12), 4),
            "presence_penalty_delta": round(min(0.18, (novelty_pressure + curiosity) * 0.10), 4),
            "max_tokens_factor": round(token_factor, 4),
        }
        routing_bias = {
            "use_private_scratchpad": salience >= 0.24,
            "model_visual_form": bool(visual_model),
            "generate_alternatives": novelty_pressure >= 0.34 or counterfactual,
            "seek_verification": tool_or_reality,
            "requires_memory_grounding": memory_pressure >= 0.55,
            "raise_metacognition": verification_pressure >= 0.45 or confusion >= 0.25,
            "consolidate_if_success": memory_pressure >= 0.50,
            "avoid_claiming_observation": True,
            "compress_imagination": admission in {"compress_foreground", "thin_frame", "defer_background"},
        }
        frame = ImaginationFrame(
            objective_truncated=objective_truncated,
            frame_id=frame_id,
            objective=text[:180],
            mode=mode,
            salience=round(salience, 4),
            novelty_pressure=round(novelty_pressure, 4),
            curiosity_pressure=round(curiosity, 4),
            affective_pressure=round(affective_pressure, 4),
            memory_pressure=round(memory_pressure, 4),
            verification_pressure=round(verification_pressure, 4),
            working_memory=working_memory,
            attractor_state=attractor_state,
            eligibility_trace=eligibility_trace,
            modalities=modalities,
            keywords=keywords,
            attention_targets=attention_targets,
            visual_model=visual_model,
            phrase_model=phrase_model,
            conceptual_bridge=conceptual_bridge,
            mental_canvas=mental_canvas,
            associative_links=associative_links,
            novel_thoughts=novel_thoughts,
            simulation_steps=simulation_steps,
            counterfactuals=counterfactuals,
            experiments=experiments,
            action_affordances=action_affordances,
            ablation_predictions=ablation_predictions,
            causal_effects=causal_effects,
            sampling_bias=sampling_bias,
            routing_bias=routing_bias,
        )
        with self._state_lock:
            self._frame_count += 1
            self._history.append(frame)
            self._frame_index[frame.frame_id] = frame
            if len(self._frame_index) > (self._history.maxlen or 0):
                # Single pass: history already bounds the live set, so one sweep
                # always suffices. The previous `while` re-scanned and could spin
                # if the index ever failed to shrink.
                live_ids = {item.frame_id for item in self._history}
                for stale_id in [k for k in self._frame_index if k not in live_ids]:
                    self._frame_index.pop(stale_id, None)
        return frame

    def snapshot(self, *, subject: str = "", include_content: bool = False) -> dict[str, Any]:
        """Health and shape. NOT the contents of the last private scratchpad.

        This returned the complete latest frame — the objective, the
        memory-derived text, the mental canvas, the novel thoughts, the
        counterfactuals, the attention targets — plus recent outcomes, to
        any caller, with no authorization, no user filter, no redaction and
        no retention policy (CP126 ``566e64ff``). It is reachable from a
        global status route, so one person's working memory was published to
        whoever asked next.

        The default is now shape without content. ``include_content=True``
        returns the latest frame for the NAMED subject only, so a caller who
        genuinely needs it has to say whose it is.
        """
        who = _subject_key(subject) if subject else ""
        with self._state_lock:
            latest_frame = self._history[-1] if self._history else None
            frame_count = self._frame_count
            history_len = len(self._history)
            attractor_bias = dict(self._attractor_bias.get(who, {})) if who else {}
            eligibility = dict(self._eligibility_trace.get(who, {})) if who else {}
            outcomes = [
                dict(record)
                for record in list(self._outcomes)[-5:]
                if not who or record.get("subject") == who
            ]
        latest = self._frame_summary(latest_frame, who, include_content)
        return {
            # An on-demand generator with no lifecycle is genuinely always
            # ready to serve, so "running" stays True and callers keep that
            # contract. What was missing is any way for this report to show a
            # problem at all — hence the explicit lifecycle kind and the real
            # degradation signals below, which DO vary.
            "running": True,
            "lifecycle": "on_demand_generator",
            "status": "active" if latest else "idle",
            "frames_built": frame_count,
            "frames": history_len,
            "latest": latest,
            "working_memory": self._working_memory_snapshot(),
            "attractor_bias": {
                key: round(value, 4)
                for key, value in sorted(attractor_bias.items())
            },
            "eligibility_trace": {
                key: round(value, 4)
                for key, value in sorted(
                    eligibility.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:12]
            },
            "recent_outcomes": outcomes,
            "bases": describe_bases(dict(DEFAULT_BASES)),
            "governance": {
                # The first two are properties of THIS module and are true of
                # it: nothing here calls a tool, writes a file, or executes.
                "advisory_only": True,
                "no_external_effects": True,
                # NOTE (CP126 false-health, partial): this third one is a claim
                # about a DOWNSTREAM system that this module cannot observe. It
                # is an assumption reported as a fact. The key name is a
                # cross-module contract (spiking_active_inference, the /system
                # routes, and their tests all use it), so correcting the claim
                # belongs to a coordinated pass over all of them, not to a
                # unilateral rename here that would leave the route defaulting
                # the old key back to True.
                "authority_gateway_required_for_effects": True,
            },
        }

    @staticmethod
    def _frame_summary(
        frame: "ImaginationFrame | None", subject: str, include_content: bool
    ) -> dict[str, Any] | None:
        """Shape of the latest frame; its content only for a named subject."""
        if frame is None:
            return None
        if include_content and subject and frame.attractor_state.get("subject") == subject:
            return frame.to_dict()
        return {
            "frame_id": frame.frame_id,
            "mode": frame.mode,
            "created_at": frame.created_at,
            "salience": frame.salience,
            "novelty_pressure": frame.novelty_pressure,
            "modalities": list(frame.modalities),
            "keyword_count": len(frame.keywords),
            "objective_truncated": frame.objective_truncated,
            "bases": dict(frame.bases),
            # Deliberately absent: objective, memories, canvas, novel
            # thoughts, counterfactuals, attention targets. Those are the
            # person's content, and a status route is not a place to publish
            # it (CP126 ``566e64ff`` / ``dcc6fd02``).
            "content_withheld": not include_content,
        }

    get_status = snapshot
    status = snapshot

    def learn_from_feedback(
        self,
        frame: str | dict[str, Any] | ImaginationFrame | None,
        *,
        reward: float,
        outcome: str = "unknown",
        subject: str = "",
        evidence_basis: str = Basis.CALLER_ASSERTED.value,
        evidence_id: str = "",
    ) -> dict[str, Any] | None:
        """Reshape selection on an OBSERVED outcome, for one subject.

        Two findings meet here.

        ``04a745b8`` — any caller could hand over a free reward and an
        outcome string with no evaluator authority, no task receipt and no
        correlation to a completed response, and it immediately changed the
        global attractor bias and the eligibility traces. The frame had to
        be one this engine issued, which stopped fabricated frames and
        nothing else: the REWARD was still whatever the caller said.

        ``f1ef7cfb`` — none of that state was partitioned, so one person's
        keywords and rewards changed selection probabilities and prompt
        steering for every later, unrelated subject.

        So a reward below :data:`LEARNING_EVIDENCE_FLOOR` is RECORDED and
        does not move anything, and everything that does move is keyed by
        subject.
        """
        materialized = self._coerce_frame(frame, require_issued=True)
        if materialized is None:
            return None
        selected = str(
            (materialized.attractor_state or {}).get("selected")
            or materialized.mode
        )
        reward_value = max(-1.0, min(1.0, _safe_float(reward, 0.0)))
        basis = Basis(evidence_basis) if evidence_basis in {b.value for b in Basis} else Basis.LEXICAL
        admitted = meets(basis, LEARNING_EVIDENCE_FLOOR)
        who = _subject_key(subject)

        with self._state_lock:
            bias_map = self._attractor_bias.setdefault(who, {})
            trace_map = self._eligibility_trace.setdefault(who, {})
            current_bias = _safe_float(bias_map.get(selected), 0.0)
            rpe = reward_value - current_bias
            if admitted:
                bias_map[selected] = max(-0.45, min(0.45, current_bias + 0.12 * rpe))
                for key, value in list(materialized.eligibility_trace.items())[:16]:
                    previous = _safe_float(trace_map.get(key), 0.0)
                    trace_map[key] = _clamp(previous + reward_value * value * 0.04)
            record = {
                "frame_id": materialized.frame_id,
                "subject": who,
                "outcome": str(outcome or "unknown")[:80],
                "reward": round(reward_value, 4),
                "selected_attractor": selected,
                "reward_prediction_error": round(rpe, 4),
                "updated_bias": round(_safe_float(bias_map.get(selected), current_bias), 4),
                "evidence_basis": basis.value,
                "evidence_id": str(evidence_id or "")[:120],
                # The honest half. An unadmitted reward is kept as a record
                # of what a caller believed and changes nothing.
                "applied": admitted,
                "refusal": "" if admitted else (
                    f"evidence basis {basis.value} is below the "
                    f"{LEARNING_EVIDENCE_FLOOR.value} floor for durable selection change"
                ),
            }
            self._outcomes.append(record)
        if not admitted:
            record_degradation(
                "imagination_engine",
                PermissionError(f"unevidenced feedback for frame {materialized.frame_id}"),
                severity="info",
                action="recorded the reward and left selection unchanged",
            )
        return record

    def _coerce_frame(
        self,
        frame: str | dict[str, Any] | ImaginationFrame | None,
        *,
        require_issued: bool = False,
    ) -> ImaginationFrame | None:
        """Resolve a feedback target to a frame.

        ``require_issued`` is the learning path's guarantee: only a frame THIS
        engine actually produced (present in the frame index) may reshape shared
        cognition. Without it, any caller could hand over a fabricated dict —
        filtered for field names only, with no proof of origin — and move the
        global attractor bias and eligibility traces with it.
        """
        if isinstance(frame, ImaginationFrame):
            if require_issued and self._frame_index.get(frame.frame_id) is not frame:
                return None
            return frame
        if isinstance(frame, str):
            return self._frame_index.get(frame)
        if isinstance(frame, dict):
            frame_id = str(frame.get("frame_id") or "")
            if frame_id and frame_id in self._frame_index:
                return self._frame_index[frame_id]
            if require_issued:
                return None
            try:
                allowed = {field.name for field in ImaginationFrame.__dataclass_fields__.values()}
                filtered = {key: value for key, value in frame.items() if key in allowed}
                return ImaginationFrame(**filtered)
            except (TypeError, ValueError, AttributeError):
                return None
        return None

    def _observe_working_memory_gate(
        self,
        *,
        salience: float,
        novelty_pressure: float,
        verification_pressure: float,
        memory_pressure: float,
        context: dict[str, Any] | None,
        is_background: bool,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            elapsed = max(0.05, min(60.0, now - self._last_observed_at))
            self._last_observed_at = now
        runtime_pressure = self._runtime_memory_pressure(context)
        pressure_level = str(runtime_pressure.get("level") or "normal")
        # An UNRECOGNISED level is not evidence of headroom either: default to
        # the same restraint an explicit "warning" earns rather than to 0.0.
        pressure_rank = {
            "normal": 0.0,
            "warning": 0.18,
            "high": 0.34,
            "critical": 0.62,
            "emergency": 0.90,
        }.get(pressure_level, 0.18)
        arrival_load = _clamp(
            0.12
            + salience * 0.36
            + novelty_pressure * 0.20
            + verification_pressure * 0.16
            + memory_pressure * 0.18
            + pressure_rank * 0.30
        )
        service_rate = _clamp(
            1.05
            - pressure_rank * 0.55
            - (0.20 if is_background else 0.0),
            lower=0.18,
            upper=1.20,
        )
        # One critical section: the read-modify-write of the queue EMAs must
        # not interleave with a concurrent frame, or both lose their update and
        # the reported load stops corresponding to either call.
        with self._state_lock:
            decay = min(self._queue_load, (elapsed / 6.0) * service_rate)
            self._queue_load = _clamp(self._queue_load - decay + arrival_load * 0.28)
            instantaneous_rate = min(12.0, 1.0 / elapsed)
            self._arrival_rate_ema = (0.82 * self._arrival_rate_ema) + (0.18 * instantaneous_rate)
            self._service_rate_ema = (0.86 * self._service_rate_ema) + (0.14 * service_rate)
            queue_load = self._queue_load
            arrival_rate_ema = self._arrival_rate_ema
            service_rate_ema = self._service_rate_ema
        overload = _clamp(max(0.0, queue_load - 0.68) / 0.32)
        if pressure_level in {"critical", "emergency"}:
            admission = "thin_frame"
        elif is_background and (overload >= 0.35 or pressure_level in {"warning", "high"}):
            admission = "defer_background"
        elif overload >= 0.45 or pressure_level == "high":
            admission = "compress_foreground"
        else:
            admission = "admit"
        expected_wait = queue_load / max(0.05, service_rate_ema)
        return {
            "admission": admission,
            "admitted": admission != "defer_background",
            "queue_load": round(queue_load, 4),
            "overload_pressure": round(overload, 4),
            "arrival_rate_hz": round(arrival_rate_ema, 4),
            "service_rate_hz": round(service_rate_ema, 4),
            "utilization": round(_clamp(self._arrival_rate_ema / max(0.05, self._service_rate_ema) / 8.0), 4),
            "expected_wait_s": round(expected_wait, 4),
            "runtime_memory_level": pressure_level,
            "runtime_memory_pressure_pct": round(_safe_float(runtime_pressure.get("pressure_pct"), 0.0), 4),
            "reason": str(runtime_pressure.get("reason") or "")[:220],
            "runtime_memory_basis": str(runtime_pressure.get("basis") or Basis.LEXICAL.value),
            # CP126 ``f58115e3``: none of the four rates above is a queue
            # measurement. There is no queued item, no enqueue or dequeue
            # event, no worker and no completed service time — it is a
            # recurrence over prompt salience and the gap between calls. The
            # numbers are a useful damping model and a dishonest telemetry
            # feed, so they say which they are.
            "model": "synthetic_load_model",
            "measures_a_real_queue": False,
        }

    @staticmethod
    def _runtime_memory_pressure(context: dict[str, Any] | None) -> dict[str, Any]:
        """Read host memory pressure. The MONITOR decides; the caller hints.

        A context dict could supply any level, percentage and reason it
        liked, with no provenance, no freshness and no range check, and the
        recognised strings went straight into admission, compression and
        load shedding (CP126 ``7975bf24``). The real monitor is consulted
        FIRST now. A caller assertion is used only when no monitor answers,
        it is clamped, and it is labelled ``caller_asserted`` so an
        admission decision made on it can be told apart from one made on a
        reading.
        """
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            return {
                "level": str(getattr(snapshot, "level", "normal") or "normal"),
                "pressure_pct": _safe_float(getattr(snapshot, "pressure_pct", 0.0), 0.0),
                "reason": str(getattr(snapshot, "reason", "") or ""),
                "basis": Basis.MEASURED.value,
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "imagination_engine",
                exc,
                severity="warning",
                action="fell back to the caller's memory-pressure hint; the monitor did not answer",
            )

        if isinstance(context, dict):
            raw = context.get("memory_pressure_snapshot") or context.get("memory_pressure")
            if isinstance(raw, dict):
                return {
                    "level": str(raw.get("level") or "normal"),
                    "pressure_pct": _clamp(
                        _safe_float(raw.get("pressure_pct"), 0.0), lower=0.0, upper=100.0
                    ),
                    "reason": str(raw.get("reason") or "")[:220],
                    "basis": Basis.CALLER_ASSERTED.value,
                }
            if isinstance(raw, (int, float)):
                pct = _safe_float(raw, 0.0)
                if pct >= 94.0:
                    level = "emergency"
                elif pct >= 90.0:
                    level = "critical"
                elif pct >= 84.0:
                    level = "high"
                elif pct >= 78.0:
                    level = "warning"
                else:
                    level = "normal"
                return {
                    "level": level,
                    "pressure_pct": pct,
                    "reason": f"context_memory_pressure:{pct:.1f}%",
                    "basis": Basis.CALLER_ASSERTED.value,
                }
        # Fail toward restraint. An unreadable memory probe is not evidence of
        # headroom — reporting "normal"/0.0 admitted ordinary imagination
        # precisely when memory safety was unknown. "warning" is the mildest
        # level that still damps admission without freezing the engine when the
        # probe is merely unavailable.
        return {
            "level": "warning",
            "pressure_pct": _UNKNOWN_MEMORY_PRESSURE_PCT,
            "reason": "memory_pressure_probe_failed:restraining",
            "basis": Basis.LEXICAL.value,
        }

    def _select_attractor_state(
        self,
        *,
        mode: str,
        salience: float,
        novelty_pressure: float,
        curiosity: float,
        affective_pressure: float,
        memory_pressure: float,
        verification_pressure: float,
        working_memory: dict[str, Any],
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
        creative: bool,
        subject: str = "anonymous",
        tool_or_reality: bool,
    ) -> dict[str, Any]:
        overload = _safe_float(working_memory.get("overload_pressure"), 0.0)
        scores = {
            "direct_answer": 0.30 + (1.0 - salience) * 0.20 - novelty_pressure * 0.08,
            "mental_canvas": (0.55 if visual else 0.05) + novelty_pressure * 0.34 + curiosity * 0.12,
            "linguistic_surface": (0.50 if linguistic else 0.08) + novelty_pressure * 0.12,
            "counterfactual_probe": (0.55 if counterfactual else 0.06) + verification_pressure * 0.22,
            "memory_bridge": 0.10 + memory_pressure * 0.58,
            "governed_action_boundary": (0.62 if tool_or_reality else 0.02) + verification_pressure * 0.40,
            "creative_synthesis": (0.50 if creative else 0.08) + novelty_pressure * 0.36 + affective_pressure * 0.14,
            "load_stabilization": overload * 0.72,
        }
        # The bias applied is the SUBJECT's own. A flat map meant one
        # person's rewards steered the next person's selection.
        bias_map = self._attractor_bias.get(subject, {})
        for key in list(scores):
            scores[key] = _safe_float(scores[key], 0.0) + _safe_float(bias_map.get(key), 0.0)
        probabilities = _stable_softmax(scores, temperature=max(0.30, 0.72 + overload * 0.35))
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[0][0] if ranked else mode
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        recurrent_depth = 1 + int(round(_clamp(novelty_pressure + verification_pressure + curiosity * 0.5) * 4))
        if overload >= 0.50:
            recurrent_depth = min(recurrent_depth, 2)
        return {
            "selected": selected,
            "probabilities": {key: round(value, 4) for key, value in ranked[:6]},
            "entropy": round(_entropy01(probabilities), 4),
            "stability_margin": round(_clamp(margin), 4),
            "recurrent_depth": recurrent_depth,
            "bias": round(_safe_float(bias_map.get(selected), 0.0), 4),
            "load_stabilized": selected == "load_stabilization" or overload >= 0.50,
            # CP126 ``4945e371``: this is a softmax over hand-built scores
            # plus learned scalar offsets, and `recurrent_depth` is computed
            # and returned. No loop ran, no state transitioned, nothing
            # converged, and no receipt shows the depth reached downstream
            # cognition. The names are kept because callers use them; the
            # claim is not.
            "mechanism": "softmax_over_authored_scores",
            "recurrent_depth_executed": False,
            "subject": subject,
        }

    def _update_eligibility_trace(
        self,
        keywords: list[str],
        *,
        selected_attractor: str,
        salience: float,
        novelty_pressure: float,
        memory_pressure: float,
    ) -> dict[str, float]:
        """Build this frame's CANDIDATE trace. It reinforces nothing yet.

        Every frame used to decay the shared trace and then reinforce its own
        attractor and keywords from salience, novelty and memory pressure —
        before any feedback, before any task succeeded. Invoking the engine
        was therefore enough to strengthen the associations that made the
        engine choose that way, which is a loop that closes on itself
        (CP126 ``91ea5bfa``).

        The eligibility a frame carries is now a PROPOSAL. It is what
        ``learn_from_feedback`` will reinforce IF an observed outcome
        arrives, and it changes the durable trace only there.
        """
        candidate: dict[str, float] = {
            f"attractor:{selected_attractor}": _clamp(salience * 0.26 + novelty_pressure * 0.12)
        }
        for token in keywords[:6]:
            candidate[f"keyword:{token}"] = _clamp(0.05 + memory_pressure * 0.05)
        return {
            key: round(value, 4)
            for key, value in sorted(candidate.items(), key=lambda item: item[1], reverse=True)[:12]
        }

    def _decay_traces(self, subject: str) -> None:
        """Age one subject's durable trace. Called when an outcome lands."""
        with self._state_lock:
            traces = self._eligibility_trace.get(subject)
            if not traces:
                return
            self._eligibility_trace[subject] = {
                key: value * 0.82
                for key, value in traces.items()
                if _safe_float(value, 0.0) * 0.82 >= 0.01
            }

    def _working_memory_snapshot(self) -> dict[str, Any]:
        return {
            "queue_load": round(self._queue_load, 4),
            "arrival_rate_hz": round(self._arrival_rate_ema, 4),
            "service_rate_hz": round(self._service_rate_ema, 4),
            "utilization": round(_clamp(self._arrival_rate_ema / max(0.05, self._service_rate_ema) / 8.0), 4),
            "history_limit": self._history.maxlen,
        }

    @staticmethod
    def _build_visual_model(keywords: list[str], memories: list[str], text: str) -> str:
        focus = ", ".join(keywords[:4]) or "the requested idea"
        memory_hint = ""
        if memories:
            memory_hint = f" It borrows continuity from: {memories[-1][:90]}."
        return (
            f"An internal sketch of {focus}: foreground constraints are visible, "
            f"tensions are spatially separated, and the next useful affordance is highlighted."
            f"{memory_hint}"
        )[:420]

    @staticmethod
    def _build_phrase_model(keywords: list[str], text: str) -> str:
        if len(keywords) >= 2:
            phrase = f"{keywords[0]} through {keywords[1]}"
        elif keywords:
            phrase = f"{keywords[0]} made operational"
        else:
            phrase = "make the invisible structure speak plainly"
        if "poem" in text.lower() or "story" in text.lower():
            phrase = f"a narrative seed around {phrase}"
        return phrase[:220]

    @staticmethod
    def _build_conceptual_bridge(keywords: list[str], memories: list[str], text: str) -> str:
        if len(keywords) >= 2:
            bridge = (
                f"Treat {keywords[0]} as the pressure source and {keywords[1]} "
                "as the surface where the pressure becomes visible."
            )
        elif keywords:
            bridge = f"Use {keywords[0]} as both object and lens: what it is, and what it reveals."
        else:
            bridge = "Map the request as a tension between possibility, evidence, and action."
        if memories:
            bridge += " Compare it against recent continuity instead of starting from a blank slate."
        return bridge[:360]

    @staticmethod
    def _build_mental_canvas(
        keywords: list[str],
        memories: list[str],
        text: str,
        *,
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
        creative: bool,
    ) -> dict[str, Any]:
        focus = keywords[:5] or ["request"]
        primary = focus[0]
        secondary = focus[1] if len(focus) > 1 else "context"
        modality = (
            "visual"
            if visual
            else "linguistic"
            if linguistic
            else "counterfactual"
            if counterfactual
            else "conceptual"
            if creative
            else "associative"
        )
        objects = [
            {"id": token, "role": "focus" if index == 0 else "support"}
            for index, token in enumerate(focus[:4])
        ]
        relations = [
            {
                "source": primary,
                "target": secondary,
                "relation": "pressures" if creative or counterfactual else "clarifies",
            }
        ]
        if len(focus) >= 3:
            relations.append(
                {
                    "source": focus[2],
                    "target": primary,
                    "relation": "reframes",
                }
            )
        memory_anchor = _normalize_text(memories[-1], 120) if memories else ""
        image_prompt = (
            f"Internal {modality} canvas: {primary} in the foreground, {secondary} as "
            "the shaping constraint, with tensions, missing evidence, and next affordances "
            "made spatially visible."
        )
        if memory_anchor:
            image_prompt += f" Continuity anchor: {memory_anchor}."
        thought_form = (
            f"Ask what {primary} becomes when {secondary} is treated as a live constraint, "
            "then compare that model against evidence before acting."
        )
        linguistic_surface = (
            f"{primary} under {secondary}"
            if len(focus) >= 2
            else f"{primary} made inspectable"
        )
        return {
            "modality": modality,
            "image_prompt": image_prompt[:500],
            "objects": objects,
            "relations": relations,
            "sensory_style": "clear edges, low ornament, constraints visible as structure",
            "linguistic_surface": linguistic_surface[:160],
            "thought_form": thought_form[:280],
            "memory_anchor": memory_anchor,
            "externalization_path": _externalization_path(focus, text),
        }

    @staticmethod
    def _build_associative_links(
        keywords: list[str], memories: list[str], text: str
    ) -> list[dict[str, str]]:
        links: list[dict[str, str]] = []
        if len(keywords) >= 2:
            links.append(
                {
                    "source": keywords[0],
                    "relation": "constrains",
                    "target": keywords[1],
                }
            )
        if len(keywords) >= 3:
            links.append(
                {
                    "source": keywords[2],
                    "relation": "reframes",
                    "target": keywords[0],
                }
            )
        if memories and keywords:
            links.append(
                {
                    "source": "recent_memory",
                    "relation": "anchors",
                    "target": keywords[0],
                }
            )
        if "tool" in text.lower() and keywords:
            links.append(
                {
                    "source": keywords[0],
                    "relation": "requires_verification_through",
                    "target": "governed_tools",
                }
            )
        return links[:4]

    _RECENCY_PENALTY = 0.55

    def _thought_signals(
        self,
        *,
        curiosity: float,
        confusion: float,
        tension: float,
        novelty_pressure: float,
        verification_pressure: float,
        memories: list[str],
        creative: bool,
        counterfactual: bool,
        linguistic: bool,
        tool_or_reality: bool,
        text: str,
    ) -> dict[str, float]:
        """Aura's measured state, in the vocabulary the moves score against.

        These are the same numbers that already drive salience and admission
        — the moves are conditioned on real interoception, not on a random
        seed dressed up as spontaneity.
        """
        lowered = text.lower()
        return {
            "curiosity": _clamp(curiosity),
            "confusion": _clamp(confusion),
            "tension": _clamp(tension),
            "novelty": _clamp(novelty_pressure),
            "verification": _clamp(verification_pressure),
            "memories": 1.0 if memories else 0.0,
            "creative": 1.0 if creative else 0.0,
            "counterfactual": 1.0 if counterfactual else 0.0,
            "linguistic": 1.0 if linguistic else 0.0,
            "tool_or_reality": 1.0 if tool_or_reality else 0.0,
            "negation": 1.0 if ("not" in lowered or "fail" in lowered) else 0.0,
        }

    def _select_moves(
        self,
        moves: tuple[_ThoughtMove, ...],
        signals: dict[str, float],
        *,
        focus: str,
        secondary: str,
        has_secondary: bool,
        seed_text: str,
        limit: int,
    ) -> list[str]:
        """Score every move against the current state, then take the best.

        Three terms decide the order:

        ``affinity``  how well this shape of thinking fits the state Aura is
                      actually in.
        ``jitter``    a small deterministic offset keyed to the content and
                      the move id. Deterministic matters — the same input in
                      the same state reproduces the same frame, which is what
                      makes imagination debuggable and testable. It is not
                      randomness; it breaks ties differently per subject so
                      two topics with identical affect do not read alike.
        ``recency``   a penalty for shapes just used. This is the direct
                      answer to "it always breaks things down in the same
                      way": the second-best move wins once the best one has
                      been spoken.
        """
        scored: list[tuple[float, str, str]] = []
        for move in moves:
            if move.needs_secondary and not has_secondary:
                continue
            try:
                affinity = _safe_float(move.affinity(signals), 0.0)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "imagination",
                    exc,
                    action=f"skip_move:{move.move_id}",
                )
                continue
            if affinity <= 0.0:
                continue
            digest = hashlib.blake2b(
                f"{seed_text}|{move.move_id}".encode(), digest_size=4,
            ).digest()
            jitter = (int.from_bytes(digest, "big") / 0xFFFFFFFF) * 0.22
            penalty = 0.0
            if move.move_id in self._recent_moves:
                # Sharper for the most recent use, softer further back.
                index = list(self._recent_moves)[::-1].index(move.move_id)
                penalty = self._RECENCY_PENALTY / (1.0 + index)
            try:
                rendered = str(move.render(focus, secondary, signals) or "").strip()
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "imagination", exc, action=f"render_move:{move.move_id}",
                )
                continue
            if not rendered:
                continue
            scored.append((affinity + jitter - penalty, move.move_id, rendered))

        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen = scored[: max(1, int(limit))]
        for _, move_id, _rendered in chosen:
            self._recent_moves.append(move_id)
        return [rendered for _, _, rendered in chosen]

    @staticmethod
    def _authored(context: dict[str, Any] | None, key: str, limit: int) -> list[str]:
        """Thoughts Aura wrote herself, if the runtime supplied any.

        ``imagine`` is synchronous and side-effect free by contract — no
        model loads, no tool calls — so it cannot ask the model for a
        thought at the moment it needs one. The seam is the other direction:
        a caller that DOES have model access (and is already awaiting it) may
        pass authored lines in through context, and they take precedence
        over the move registry.

        The registry stays the floor, not the fallback. Imagination that
        silently stopped working whenever generation was busy would be worse
        than one that always thinks in its own voice.
        """
        if not isinstance(context, dict):
            return []
        raw = context.get(key)
        if not isinstance(raw, (list, tuple)):
            return []
        authored: list[str] = []
        for item in raw:
            line = _normalize_text(item, 240)
            if line and line not in authored:
                authored.append(line)
            if len(authored) >= max(1, int(limit)):
                break
        return authored

    def _build_novel_thoughts(
        self,
        keywords: list[str],
        memories: list[str],
        text: str,
        *,
        signals: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        authored = self._authored(context, "authored_novel_thoughts", 4)
        if authored:
            return authored
        return self._select_moves(
            _NOVEL_MOVES,
            signals,
            focus=keywords[0] if keywords else "the idea",
            secondary=keywords[1] if len(keywords) > 1 else "its constraint",
            has_secondary=len(keywords) > 1,
            seed_text=text,
            limit=4,
        )

    @staticmethod
    def _build_simulation_steps(
        keywords: list[str],
        *,
        visual: bool,
        counterfactual: bool,
        tool_or_reality: bool,
    ) -> list[str]:
        focus = keywords[0] if keywords else "the premise"
        steps = [
            f"Render {focus} as concrete constraints instead of a label.",
            "Generate at least two alternate forms before settling on one.",
        ]
        if visual:
            steps.append("Place actors, constraints, and missing evidence in the imagined scene.")
        if counterfactual:
            steps.append("Run the premise forward, then invert it and compare behavior.")
        if tool_or_reality:
            steps.append("Stop at the boundary where real-world verification or authority is required.")
        return steps[:4]

    def _build_counterfactuals(
        self,
        keywords: list[str],
        text: str,
        *,
        signals: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        authored = self._authored(context, "authored_counterfactuals", 4)
        if authored:
            return authored
        return self._select_moves(
            _COUNTERFACTUAL_MOVES,
            signals,
            focus=keywords[0] if keywords else "the premise",
            secondary=keywords[1] if len(keywords) > 1 else "its constraint",
            has_secondary=len(keywords) > 1,
            # Salt the seed so a probe and a thought about the same subject
            # do not tie-break identically and read as a matched pair.
            seed_text=f"counterfactual|{text}",
            limit=4,
        )

    @staticmethod
    def _build_experiments(keywords: list[str], tool_or_reality: bool) -> list[str]:
        focus = keywords[0] if keywords else "the idea"
        experiments = [
            f"Name the concrete observable that would make {focus} less abstract.",
            "Generate two alternatives, then prefer the one with clearer evidence.",
        ]
        if tool_or_reality:
            experiments.append("Route any real-world effect through governed tools and receipts.")
        else:
            experiments.append("Keep it as a mental model unless the user asks for action.")
        return experiments[:3]

    @staticmethod
    def _build_attention_targets(keywords: list[str], text: str) -> list[str]:
        targets = list(keywords[:5])
        lowered = text.lower()
        if "tool" in lowered and "governed_tools" not in targets:
            targets.append("governed_tools")
        if ("remember" in lowered or "memory" in lowered) and "memory_continuity" not in targets:
            targets.append("memory_continuity")
        if any(token in lowered for token in ("verify", "proof", "evidence")) and "verification" not in targets:
            targets.append("verification")
        return targets[:6]

    @staticmethod
    def _build_action_affordances(
        keywords: list[str],
        *,
        tool_or_reality: bool,
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
    ) -> list[str]:
        focus = keywords[0] if keywords else "the model"
        affordances = [f"hold {focus} as an internal model before answering"]
        if counterfactual:
            affordances.append("compare at least two possible futures")
        if visual:
            affordances.append("model spatial/visual structure privately before describing it")
        if linguistic:
            affordances.append("search for a concise surface phrase after the model is stable")
        if tool_or_reality:
            affordances.append("route any external effect through governed tools and receipts")
        return affordances[:5]

    @staticmethod
    def _build_ablation_predictions(
        *,
        memory_pressure: float,
        verification_pressure: float,
        novelty_pressure: float,
        tool_or_reality: bool,
    ) -> dict[str, str]:
        # CP126 ``3369210a``: these are behavioural claims emitted from
        # thresholds. No paired run, no intervention hook, no baseline, no
        # metric, no stored result tests any of them. They are worth having —
        # a hypothesis an ablation harness can pick up is better than none —
        # so each is prefixed with what it is, and the frame records the
        # basis as TEMPLATE so nothing downstream reads them as results.
        predictions: dict[str, str] = {
            "no_imagination": "UNTESTED HYPOTHESIS: fewer alternatives and weaker counterfactual framing",
        }
        if memory_pressure >= 0.50:
            predictions["no_memory_continuity"] = "UNTESTED HYPOTHESIS: recent context should anchor less of the response"
        if verification_pressure >= 0.45:
            predictions["no_governance_or_tools"] = "UNTESTED HYPOTHESIS: real-world claims should lose verification pressure"
        if novelty_pressure >= 0.35:
            predictions["no_novelty_drive"] = "UNTESTED HYPOTHESIS: creative synthesis should collapse toward a safer default framing"
        if tool_or_reality:
            predictions["no_authority_gateway"] = "UNTESTED HYPOTHESIS: external action must block rather than proceed directly"
        return predictions

    @staticmethod
    def _build_causal_effects(
        *,
        attention_targets: list[str],
        memory_pressure: float,
        verification_pressure: float,
        curiosity: float,
        confusion: float,
        tool_or_reality: bool,
        working_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metacognition_depth = _clamp(0.30 + verification_pressure * 0.45 + confusion * 0.35 + curiosity * 0.20)
        working_memory = working_memory if isinstance(working_memory, dict) else {}
        admission = str(working_memory.get("admission") or "admit")
        overload = _safe_float(working_memory.get("overload_pressure"), 0.0)
        return {
            "attention_focus": attention_targets[:4],
            "memory_priority": round(memory_pressure, 4),
            "verification_pressure": round(verification_pressure, 4),
            "metacognition_depth": round(metacognition_depth, 4),
            # A regex matched the word "file" or "search". That is a REQUEST
            # to consult governance, never a governance decision: Will,
            # scoped authority, permissions, action risk and consent were
            # none of them consulted, and no tool receipt exists (CP126
            # ``b95c4d62``). The key stays because callers read it; what it
            # means is now written down beside it.
            "tool_governance": bool(tool_or_reality or verification_pressure >= 0.45),
            "tool_governance_basis": Basis.LEXICAL.value,
            "tool_governance_is_a_decision": False,
            "external_effects_allowed": False,
            "working_memory_admission": admission,
            "working_memory_overload": round(overload, 4),
            "load_shed_requested": admission in {"compress_foreground", "thin_frame", "defer_background"},
            "expected_downstream": [
                effect
                for effect, active in (
                    ("attention_bias", bool(attention_targets)),
                    ("memory_retrieval_bias", memory_pressure >= 0.45),
                    ("memory_consolidation_bias", memory_pressure >= 0.50),
                    ("verification_bias", verification_pressure >= 0.35),
                    ("governed_tool_boundary", tool_or_reality),
                    ("runtime_load_shed", admission in {"compress_foreground", "thin_frame", "defer_background"}),
                )
                if active
            ],
        }


_INSTANCE: ImaginationEngine | None = None


def get_imagination_engine() -> ImaginationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ImaginationEngine()
    try:
        from core.container import ServiceContainer

        current = ServiceContainer.get("imagination_engine", default=None)
        if current is not _INSTANCE:
            ServiceContainer.register_instance(
                "imagination_engine",
                _INSTANCE,
                required=False,
                owner="core/brain/imagination.py",
                registered_by="core.brain.imagination.get_imagination_engine",
                required_for="creative and counterfactual cognitive steering",
                failure_policy="degrade_with_receipt",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "imagination_engine",
            exc,
            severity="warning",
            action="continued without ServiceContainer registration for imagination engine",
        )
    return _INSTANCE


def render_imagination_prompt_block(frame: dict[str, Any] | ImaginationFrame, *, compact: bool = False) -> str:
    if isinstance(frame, ImaginationFrame):
        return frame.prompt_block(compact=compact)
    if not isinstance(frame, dict):
        return ""
    try:
        allowed = {field.name for field in ImaginationFrame.__dataclass_fields__.values()}
        filtered = {key: value for key, value in frame.items() if key in allowed}
        materialized = ImaginationFrame(**filtered)
        return materialized.prompt_block(compact=compact)
    except (TypeError, ValueError, AttributeError) as exc:
        record_degradation(
            "imagination_engine",
            exc,
            severity="warning",
            action="skipped malformed imagination prompt block",
        )
        return ""


__all__ = [
    "ImaginationEngine",
    "ImaginationFrame",
    "get_imagination_engine",
    "render_imagination_prompt_block",
]

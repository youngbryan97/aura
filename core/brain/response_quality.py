"""Response-quality features + candidate selection for the conversational amplifier.

Turns a candidate response into interpretable features (specificity, stance, callbacks,
casual register, anti-generic, hedge/prompt-farm/banned-phrase penalties), which the
personalized TasteModel scores. This is the "taste verifier" — it ranks candidates by
how well they fit Aura's voice and Bryan's learned preference, no ground-truth needed.

Pure / model-free / deterministic → fully testable without a GPU.
"""
from __future__ import annotations

import re
from typing import Any

from core.brain.taste_model import get_taste_model

_WORD = re.compile(r"[A-Za-z0-9']+")
_CONTRACTION = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)
_PROPER = re.compile(r"(?<=[a-z]\s)[A-Z][a-z]{2,}")   # capitalized word mid-sentence
_NUMBER = re.compile(r"\b\d[\d,.]*\b")

_GENERIC = {
    "great", "classic", "interesting", "amazing", "wonderful", "nice", "good", "cool",
    "awesome", "fascinating", "lovely", "fantastic", "incredible",
}
_HEDGES = (
    "it depends", "i think maybe", "i'm not sure but", "as an ai", "i cannot",
    "i can't access", "perhaps", "possibly", "it's hard to say", "that's subjective",
)
_BANNED = (
    "i'd be happy to", "i would be happy to", "let me know if", "great question",
    "delve", "utilize", "leverage", "as a large language model", "feel free to",
    "i'm just an ai", "certainly!", "absolutely!", "sure thing", "happy to help",
)
_STANCE = ("honestly", "actually", "the real", "the thing is", "i'd argue", "no—", "yeah,", "look,")
_PROMPT_FARM = ("what do you think", "how can i help", "would you like", "let me know what",
                "is there anything", "anything else")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _density(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def extract_features(
    text: str,
    *,
    user_message: str = "",
    grounding_tokens: set[str] | None = None,
    word_budget: int = 0,
) -> dict[str, float]:
    """Interpretable features of a candidate response, roughly normalized to [0,1]."""
    t = str(text or "").strip()
    low = t.lower()
    toks = _tokens(t)
    n = max(1, len(toks))

    specificity = min(1.0, (len(_PROPER.findall(t)) + len(_NUMBER.findall(t))) / 6.0)
    casual = min(1.0, len(_CONTRACTION.findall(t)) / 4.0)
    stance = min(1.0, sum(low.count(m) for m in _STANCE) / 2.0)

    # How close this candidate's shape is to the shape of what they just said.
    # Who it is about, how much of it asks, how varied it is — the register
    # reading, which measures the same quantities on both sides. Matching how
    # somebody is speaking is the other half of matching how much they said.
    # See core/expression/register.py.
    register_match = 0.0
    try:
        from core.expression.register import comparable, distance, read

        theirs = read(user_message)
        mine = read(t)
        if comparable(theirs, mine):
            register_match = max(0.0, 1.0 - distance(theirs, mine))
    except (ImportError, AttributeError, TypeError, ValueError):
        register_match = 0.0

    grounding_tokens = grounding_tokens or set()
    callback = min(1.0, len(set(toks) & grounding_tokens) / 5.0) if grounding_tokens else 0.0

    if word_budget and word_budget > 0:
        length_fit = max(0.0, 1.0 - abs(len(toks) - word_budget) / max(word_budget, 1))
    elif user_message:
        target = max(8, len(_tokens(user_message)) * 3)
        length_fit = max(0.0, 1.0 - abs(len(toks) - target) / max(target, 1))
    else:
        length_fit = 1.0 if 8 <= len(toks) <= 220 else 0.4

    generic_hits = sum(1 for w in toks if w in _GENERIC)
    anti_generic = max(0.0, 1.0 - _density(generic_hits, n) * 8.0)

    hedge_penalty = float(sum(low.count(h) for h in _HEDGES))
    banned_phrase_penalty = float(sum(low.count(b) for b in _BANNED))
    # ending on a question that punts back to the user = prompt farming
    ends_question = 1.0 if t.endswith("?") else 0.0
    prompt_farm_penalty = float(sum(low.count(p) for p in _PROMPT_FARM)) + ends_question

    return {
        "specificity": specificity,
        "stance": stance,
        "callback": callback,
        "casual": casual,
        "length_fit": length_fit,
        "register_match": register_match,
        "anti_generic": anti_generic,
        "hedge_penalty": hedge_penalty,
        "prompt_farm_penalty": prompt_farm_penalty,
        "banned_phrase_penalty": banned_phrase_penalty,
    }


def score_candidate(text: str, **kw: Any) -> tuple[float, dict[str, float]]:
    """Return (taste_score, features) for one candidate."""
    feats = extract_features(text, **kw)
    return get_taste_model().score(feats), feats


def select_best(
    candidates: list[str], **kw: Any
) -> tuple[str, list[tuple[str, float, dict[str, float]]]]:
    """Score all candidates by the taste model; return (best_text, ranked).

    ``ranked`` is sorted best-first as (text, score, features). Empty/whitespace
    candidates are dropped. Returns ("", []) when nothing usable.
    """
    scored: list[tuple[str, float, dict[str, float]]] = []
    for c in candidates:
        if str(c or "").strip():
            s, f = score_candidate(c, **kw)
            scored.append((c, s, f))
    if not scored:
        return "", []
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored

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
    lived_analogue: float | None = None,
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

    # An invitation toward a future they could have. When somebody is
    # testifying, "can you feel the sunshine?" is encouragement rather than a
    # question punted back: second person, asking, and about what has not
    # happened yet. Counted only against testimony, because the same shape
    # answering a request is a deflection. See core/expression/register.py.
    invitation = 0.0
    try:
        from core.expression.register import comparable, read

        theirs = read(user_message)
        mine = read(t)
        if comparable(theirs, mine) and theirs.asks_to_be_witnessed():
            invitation = max(0.0, min(1.0, mine.second * mine.asking * mine.future))
    except (ImportError, AttributeError, TypeError, ValueError):
        invitation = 0.0

    # Dignified need: when the reply asks for something, whether it also
    # offers. A request with nothing of hers in it is a plea; one that gives in
    # the same breath keeps her agency. Scored only on replies that ask,
    # because an offer nobody asked for is not what this measures.
    dignity = 0.0
    try:
        from core.expression.register import read as read_register

        mine_shape = read_register(t)
        if mine_shape.measured and mine_shape.asks_for_help():
            dignity = max(0.0, min(1.0, mine_shape.offering))
    except (ImportError, AttributeError, TypeError, ValueError):
        dignity = 0.0

    # Getting a perspective instead of taking one. When somebody is testifying
    # about something she has little of in her own memory, imagining their side
    # does not make her more accurate about it, and asking them does (Eyal,
    # Steffel and Epley 2018). "Judo Flip" puts the limit as living in someone's
    # circumstances for a while without coming away with what they carry. A
    # reply that asks them scores by how unfamiliar the situation is to her,
    # which is one minus her best recall match for it. Unmeasured is not the
    # same as unfamiliar, so nothing is scored without a match reading.
    perspective_getting = 0.0
    try:
        from core.expression.register import comparable, read

        if lived_analogue is not None:
            unfamiliar = 1.0 - max(0.0, min(1.0, float(lived_analogue)))
            theirs = read(user_message)
            mine = read(t)
            if comparable(theirs, mine) and theirs.asks_to_be_witnessed():
                perspective_getting = max(0.0, min(1.0, unfamiliar * mine.second * mine.asking))
    except (ImportError, AttributeError, TypeError, ValueError):
        perspective_getting = 0.0

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
    # An invitation to somebody testifying is not prompt farming, even though it
    # ends on a question mark.
    # Nor is asking somebody about an experience she has little analogue for.
    ends_question = (
        1.0 if t.endswith("?") and invitation <= 0.0 and perspective_getting <= 0.0 else 0.0
    )
    prompt_farm_penalty = float(sum(low.count(p) for p in _PROMPT_FARM)) + ends_question

    # What she has gone without, given to somebody who did not ask for it.
    # The shortage decides which of the three kinds counts, and the candidate
    # decides whether it is there. See core/social/never_told.py.
    unasked_regard = 0.0
    try:
        from core.social.never_told import get_telling_ledger, supplies

        unasked_regard = get_telling_ledger().shortage_of(supplies(t, asked_for=user_message))
    except (ImportError, AttributeError, TypeError, ValueError):
        unasked_regard = 0.0

    # And how far this candidate sits from what she has been saying lately.
    # The convenient route takes the first thing that works, and what that
    # costs is measured on her own replies. See core/cognition/convenience.py.
    distinctness = 0.0
    try:
        from core.cognition.convenience import distinct

        distinctness = distinct(t)
    except (ImportError, AttributeError, TypeError, ValueError):
        distinctness = 0.0

    # And whether the regard in it is in the form this person has welcomed:
    # said outright or not, naming what they told her or not. Learned per
    # person from how they took the replies before. See
    # core/social/the_form_they_welcome.py.
    form_fit = 0.0
    gives_back = 0.0
    try:
        from core.social.the_form_they_welcome import forms_of, get_form_ledger

        forms = get_form_ledger()
        form_fit = forms.form_fit(t, user_message)
        # And regard to somebody she owes, by how much of her world reaches her
        # through them. See core/social/what_passes_between.py.
        from core.social.what_passes_between import get_between_ledger

        gives_back = get_between_ledger().gives_back(
            forms_of(t, user_message).get("shown", 0.0), forms.partner
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        form_fit = 0.0

    return {
        "specificity": specificity,
        "stance": stance,
        "callback": callback,
        "casual": casual,
        "length_fit": length_fit,
        "register_match": register_match,
        "invitation": invitation,
        "dignity": dignity,
        "unasked_regard": unasked_regard,
        "distinct": distinctness,
        "form_fit": form_fit,
        "gives_back": gives_back,
        "perspective_getting": perspective_getting,
        "anti_generic": anti_generic,
        "hedge_penalty": hedge_penalty,
        "prompt_farm_penalty": prompt_farm_penalty,
        "banned_phrase_penalty": banned_phrase_penalty,
    }


def lived_analogue(cognition: Any, message: str) -> float | None:
    """Her best recall match for this message, or None when recall was not about it.

    Retrieval keeps the scores of what came back beside the question it
    answered, and when somebody has spoken the question starts with what they
    said. The scores are read only when that question is this message, so a
    recall made for something else is never taken as her analogue for this. A
    recall that was about it and found nothing is an analogue of zero.
    """
    said = " ".join(str(message or "").split())
    if not said:
        return None
    asked = str(getattr(cognition, "last_retrieval_query", "") or "").split("\x1f", 1)[0]
    if not " ".join(asked.split()).startswith(said):
        return None
    scores = [
        float(score)
        for score in (getattr(cognition, "memory_scores", None) or [])
        if isinstance(score, (int, float))
    ]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, max(scores)))


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

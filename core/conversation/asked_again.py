"""Whether a request asks what an earlier one asked: the same question again, however it is worded.

Her answer to a question is about the moment she gave it. Asked it again later, she answers again; she does not read
the first answer back. LIVE 13 November 2026, "what have you been up to since you woke" asked at 13:44, 14:50 and
15:12 in three wordings: the second answer was seven of ten sentences of the first, read from her recall, and the
third began with the second's whole first sentence, read from her transcript, with figures that had stopped being
true ("awake for twelve minutes", four minutes after she woke). Nothing replayed a template. The earlier answer was in
front of her, and copying it was the likeliest continuation.

The recall that set her earlier answers aside (8 November) compared the request's words, so a rewording brought the
answer back. This judges by meaning with her encoder, against a boundary measured from declared pairs: the same
question reworded, and a question on the same subject that asks something else. A follow-up that works on an earlier
answer ("fix line 3 of that") asks something else, and her reuse of that answer stays hers. Where her encoder is not
loaded in this process, identical words decide, as before.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

#: A question and the same question reworded; then a question and another on its subject.
_ASKS_THE_SAME = (
    ("What have you been up to since you woke up?", "So what have you been up to since you woke?"),
    ("How are you feeling right now?", "How do you feel at the moment?"),
    ("What's the capital of Australia?", "Which city is Australia's capital?"),
    ("Can you explain how black holes form?", "How do black holes form?"),
    ("What did you think of the video?", "So what did you make of that video?"),
    ("What are you working on today?", "What have you been working on today?"),
    ("Hey, back for a minute. What have you been up to since you woke up?",
     "Still quiet at work. So what have you been up to since you woke?"),
)
_ASKS_SOMETHING_ELSE = (
    ("What have you been up to since you woke up?", "What do you plan to do next?"),
    ("How are you feeling right now?", "Why do you feel tired?"),
    ("What's the capital of Australia?", "What's the population of Australia?"),
    ("Can you explain how black holes form?", "Can you write that explanation as a poem?"),
    ("What did you think of the video?", "Who made that video?"),
    ("Write a function that parses dates.", "Fix the bug in line 3 of that function."),
    ("Tell me about black holes.", "Tell me about neutron stars."),
    ("What's your favorite book?", "What's your favorite movie?"),
)

#: Her encoder (Qwen3-Embedding) is trained to take the task with each text; "asks the same question" is declared
#: with her other tasks (core/memory/embedding_model.py). Without it the encoder measures topic: black holes and
#: neutron stars scored 0.855, over the line; with it 0.651, and her four wordings of "what have you been up to since
#: you woke" 0.759 to 0.965.
_TASK_NAME = "same_question"

Features = Callable[[Iterable[str]], list[list[float]]]
_BOUNDARY: dict[Any, Any] = {}


def in_these_words(text: object) -> str:
    """Words only, lower case, one space apart."""
    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def asks_the_same(request: str, earlier: Sequence[str], *, features: Features | None = None) -> list[bool]:
    """For each earlier request, whether ``request`` asks what it asked."""
    asked = in_these_words(request)
    same_words = [bool(asked) and in_these_words(text) == asked for text in earlier]
    meant = _by_meaning(str(request or ""), [str(text or "") for text in earlier], features)
    if not meant:
        return same_words
    return [words or decided is True for words, decided in zip(same_words, meant, strict=True)]


def her_answers_when_asked_this(request: str, exchanges: Sequence[tuple[str, str]], *,
                                features: Features | None = None) -> list[str]:
    """Her answers, among ``(asked, answered)`` exchanges, to requests that asked what ``request`` asks."""
    rows = [(str(asked or ""), str(answered or "")) for asked, answered in exchanges if str(answered or "").strip()]
    if not rows or not str(request or "").strip():
        return []
    same = asks_the_same(request, [asked for asked, _answered in rows], features=features)
    return [answered for (_asked, answered), is_same in zip(rows, same, strict=True) if is_same]


def _by_meaning(request: str, earlier: list[str], features: Features | None) -> list[bool | None]:
    """For each earlier request, whether it is the same question by meaning (None in the overlap); [] unjudged."""
    from core.language.learned_matcher import cosine

    if not earlier:
        return []
    if features is None:
        features = _as_a_question
    boundary = _boundary(features)
    if boundary is None or not boundary.separable:
        return []
    vectors = features([request, *earlier])
    if len(vectors) != len(earlier) + 1:
        return []
    return [boundary.decide(cosine(vectors[0], vector)) for vector in vectors[1:]]


def _as_a_question(texts: Iterable[str]) -> list[list[float]]:
    """Each text with the task, from the encoder her evidence judge uses, only once it is loaded; [] otherwise."""
    from core.cognition.evidence_relevance import vectors_if_ready
    from core.memory.embedding_model import query_prompt

    return vectors_if_ready([query_prompt(_TASK_NAME) + str(text or "") for text in texts])


def _boundary(features: Features) -> Any:
    """Where two requests ask the same and where they do not, measured once from the declared pairs."""
    from core.language.learned_matcher import Boundary, cosine

    if features in _BOUNDARY:
        return _BOUNDARY[features]
    pairs = (*_ASKS_THE_SAME, *_ASKS_SOMETHING_ELSE)
    vectors = features([sentence for pair in pairs for sentence in pair])
    if len(vectors) != 2 * len(pairs):
        return None
    scores = [cosine(vectors[2 * i], vectors[2 * i + 1]) for i in range(len(pairs))]
    same, other = scores[: len(_ASKS_THE_SAME)], scores[len(_ASKS_THE_SAME):]
    boundary = Boundary(decide_true_above=max(other), decide_false_below=min(same), separable=min(same) > max(other))
    _BOUNDARY[features] = boundary
    return boundary


def her_answers_in_the_conversation(request_kwargs: dict[str, Any]) -> list[str]:
    """Her answers, in the conversation her model is given, to the question this turn asks again.

    Empty outside a turn that answers a person, and when the person asks her to say it again.
    """
    from core.brain.llm.mlx_worker_surface_quality import _ASKED_TO_REPEAT

    if not request_kwargs.get("clean_user_surface_contract") or request_kwargs.get("her_own_turn"):
        return []
    request = str(request_kwargs.get("user_surface_validation_prompt") or "")
    if not request.strip() or _ASKED_TO_REPEAT.search(request):
        return []
    messages = [message for message in request_kwargs.get("messages") or () if isinstance(message, dict)]
    roles = [str(message.get("role") or "").strip().lower() for message in messages]
    pairs = [(str(messages[i].get("content") or ""), str(messages[i + 1].get("content") or ""))
             for i in range(len(messages) - 1) if roles[i] == "user" and roles[i + 1] == "assistant"]
    return [answer[:4000] for answer in her_answers_when_asked_this(request, pairs)[-4:]]

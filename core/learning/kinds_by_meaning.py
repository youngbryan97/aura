"""Which kind of problem a request is, by what it means: her embedding of it beside each kind's problems.

A kept procedure answers the kinds of problem she has solved examples of
(core/learning/procedures_from_solved_examples.py), and her book routes a
request to a kind by the words they share. That reaches a request worded like
the kind's examples and nothing else: a person who asks "what's 15% of 340?"
in their own words shares few words with a dataset's "Calculate 0.15*340.",
and a person's way of asking is the case that matters live.

Here a kind is also a direction in her embedding space (Qwen3-Embedding,
core/memory/embedding_model.py): the mean of its shown-pool problems embedded
as documents. A request is embedded as a query for the task "kind_of_problem",
and so is every problem of the kind's sealed half, which sets the kind's bar
the way the words' bar is set: a request goes to the kind it is nearest only
if it is at least as near as the least typical sealed problem was. A request
must also be meant clearly, by the rule her capabilities are chosen by
(core/intent/capability_by_meaning.py): the nearest kind stands out from all
the kinds' scores and clearly from the next.

A request that reaches a kind by meaning and not by its words is phrased
otherwise than the kind's problems, so the procedure, written for that
phrasing, is not run on it as it stands; the response phase has it restated
in the kind's form first (``restatement_request``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["TASK", "KindMeaning", "meanings_of", "nearest_kind", "restatement_request", "restates",
           "shared_opening"]

#: What a request is embedded for (core/memory/embedding_model.py TASK_INSTRUCTIONS).
TASK = "kind_of_problem"

Embed = Callable[[list[str]], np.ndarray]


@dataclass
class KindMeaning:
    """A kind's direction in her embedding space, and how near its least typical sealed problem came."""

    centroid: list[float]
    nearest_required: float


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    rows = rows.reshape(len(rows), -1) if rows.ndim > 1 else rows.reshape(1, -1)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


def meanings_of(kinds: Mapping[str, tuple[Sequence[str], Sequence[str]]], *, documents: Embed,
                queries: Embed) -> dict[str, KindMeaning]:
    """Each kind's meaning from its (shown pool, sealed half) problem texts."""
    out: dict[str, KindMeaning] = {}
    for kind, (pool, sealed) in kinds.items():
        if not pool or not sealed:
            continue
        centroid = _unit(_unit(documents(list(pool))).mean(axis=0, keepdims=True))[0]
        bar = float((_unit(queries(list(sealed))) @ centroid).min())
        out[kind] = KindMeaning(centroid=[float(x) for x in centroid], nearest_required=bar)
    return out


def nearest_kind(query: np.ndarray, meanings: Mapping[str, KindMeaning]) -> tuple[str | None, float, str]:
    """The kind a request's query vector means, its similarity, and why none when none.

    The nearest kind must pass its own bar and be meant clearly: it stands out
    from all the kinds' scores (STANDS_OUT deviations above their mean) and
    from the next (CLEAR_OF deviations ahead), the rule of
    core/intent/capability_by_meaning.py. Among n kinds none can stand further
    out than sqrt(n - 1) deviations, so fewer than nine never choose one.
    """
    from core.intent.capability_by_meaning import CLEAR_OF, STANDS_OUT

    if len(meanings) < 2 or float(np.sqrt(len(meanings) - 1)) < STANDS_OUT:
        return None, 0.0, "too_few_kinds_to_tell_meaning_apart"
    names = list(meanings)
    vector = _unit(np.asarray(query, dtype=np.float32))[0]
    scores = np.array([float(np.asarray(meanings[n].centroid, dtype=np.float32) @ vector) for n in names])
    order = np.argsort(-scores)
    best, second = int(order[0]), int(order[1])
    spread = float(scores.std()) or 1e-9
    stands, clear = (scores[best] - scores.mean()) / spread, (scores[best] - scores[second]) / spread
    if scores[best] < meanings[names[best]].nearest_required:
        return None, float(scores[best]), "less_typical_than_any_sealed_problem"
    if stands < STANDS_OUT or clear < CLEAR_OF:
        return None, float(scores[best]), "no_kind_meant_clearly"
    return names[best], float(scores[best]), ""


def restatement_request(request: str, examples: Sequence[Mapping[str, str]]) -> str:
    """What her model is asked when a request means a kind but is worded otherwise: the same problem,
    written the way the kind's own problems are, so the kind's procedure can read it."""
    shown = "\n\n".join(f"Problem:\n{e['problem'].strip()}" for e in examples)
    return (
        "Here are problems of one kind, each written the same way. Write the problem below as a problem "
        "of this kind, in the same form, with all of its numbers and names, and nothing else.\n\n"
        f"{shown}\n\nThe problem to write in that form:\n{request.strip()}"
    )


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: A string written in quotes, and the escapes inside one that stand for a character rather than say anything.
_QUOTED = re.compile(r"""(?:[rbfu]{0,2})(["'])((?:\\.|(?!\1).)*)\1""", re.I)
_ESCAPE = re.compile(r"\\[ntrbfvw0sSdDW]|\\.")


def _given_strings(text: str) -> set[str]:
    """The strings a text writes in quotes, as written."""
    return {m.group(2) for m in _QUOTED.finditer(str(text or ""))}


def brings_only_what_was_given(request: str, restated: str) -> bool:
    """Whether every string the restatement hands its procedure was given in the request, in quotes or code.

    A string that says something (a letter or a digit once its escapes are set aside) and that the request did not
    give is the restatement's own, and a value computed on it answers a problem nobody asked. L09's second attempt,
    19 October 2026: "how to check if email address is valid" restated as ``f("user@example.com")``, "write a
    hello world script" as ``f()`` returning ``"hello world"``, "help me write a python code express emotion" as
    ``f('happy')``; 22 of its 25 misroutes were requests for code whose restatement invented the input, and the
    value served answered nothing asked. Given in words is not given as a value: "a hello world script" does not
    give the string "hello world".
    """
    given = _given_strings(request)
    for said in _given_strings(restated):
        if not re.search(r"[A-Za-z0-9]", _ESCAPE.sub("", said)):
            continue
        if said not in given and not any(said in g for g in given):
            return False
    return True


def restates(request: str, restated: str, examples: Sequence[Mapping[str, str]] = ()) -> bool:
    """Whether a restatement is the problem asked, as far as its numbers can say.

    Nothing else checks it. Live 2026-10-16, asked to write a greatest common
    divisor in its kind's form, her model began analysing the kind's shown
    examples instead, and the kind's procedure answered the first example's
    numbers: 918, for a request whose answer is 374. A number may change its
    notation in a restatement (15% is 0.15), so the rule is on what cannot be
    the request's: a number taken from a shown example that the request does
    not have, or one of the request's numbers cut short. And it keeps at least
    one number of a request that has any.
    """
    if not brings_only_what_was_given(request, restated):
        return False
    asked = set(_NUMBER.findall(str(request or "")))
    written = set(_NUMBER.findall(str(restated or "")))
    shown = {n for e in examples for n in _NUMBER.findall(str(e.get("problem") or ""))}
    if asked and not asked & written:
        return False
    for number in written - asked:
        if number in shown or any(number in whole for whole in asked):
            return False
    return True


def shared_ending(problems: Sequence[str]) -> str:
    """The paragraphs every one of a kind's problems closes with, whole; "" when they share none.

    The ending's twin of :func:`shared_opening`. Every DeepMind mathematics problem
    closes with "Please reason step by step, and put your final answer within
    \\boxed{}.", so a kind's meaning carried that sentence and a person's request,
    which never does, fell below the bar of the kind it plainly was (live
    2026-10-18: "What are the prime factors of 21000063?" at 0.733 against 0.759).
    Cut at the first paragraph break the problems share, so the question itself
    is never split.
    """
    texts = [str(p) for p in problems]
    if len(texts) < 2:
        return ""
    tail = os.path.commonprefix([t[::-1] for t in texts])[::-1]
    cut = tail.find("\n\n")
    return tail[cut:] if cut >= 0 else ""


def shared_opening(problems: Sequence[str]) -> str:
    """The paragraphs every one of a kind's problems opens with, whole; "" when they share none.

    A benchmark that poses each problem after the same worked examples makes
    those paragraphs most of the kind's text, so a kind's meaning computed on
    them is the examples' meaning, and a request in a person's words falls
    below its bar (live 2026-10-16: 0.675 against BIG-Bench Hard word
    sorting's 0.799). Cut at the last paragraph break the problems share, so
    the question itself is never split.
    """
    texts = [str(p) for p in problems]
    if len(texts) < 2:
        return ""
    head = os.path.commonprefix(texts)
    cut = head.rfind("\n\n")
    return head[: cut + 2] if cut >= 0 else ""

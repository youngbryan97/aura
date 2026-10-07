"""Training requests that teach what three wordings per operation and short chains could not.

Two gaps in the 764 training requests showed on G04's first run (6 October):

* Every operation was said three ways, each led by a verb that alone names
  it ("subtract", "multiply"). A readout fitted there learns that the verb
  decides, and "take 7 away from 46" was read as addition at "take". Here each
  operation is also said in wordings whose meaning is only settled later in
  the phrase ("take ... away from", "share ... evenly among", "split ... into
  ... equal whole parts"), so a readout can learn where a phrase settles.
* No result was more than two operations back from a mention of it, so the
  first clause to match a repeated description ("the intermediate value") and
  the latest result were always the same clause. Chains of three to five
  steps, inside the depth everything consumed already has, separate them.

Nothing here uses a word of g04_transfer_v2's held-out table, whose lemmas
were committed first (semantic_g04_transfer_corpus._VOCABULARY_V2_SIGNATURES),
nor "times", "ways", "fit" or "group", which that table's phrasings are built
around. Every request is split ``train``.
"""

from __future__ import annotations

import random
from typing import Final

from core.learning.semantic_g04_transfer_corpus import (
    _CONSUMED_PHRASES,
    _SCALAR_OPS,
    _VOCABULARY_V2_SIGNATURES,
    _chain,
    _Request,
    _t,
    _Template,
    _then_scaffold,
    _two_step,
)
from core.learning.semantic_program_corpus import SemanticProgramExample

__all__ = [
    "BREADTH_WORDINGS",
    "TRAINING_BREADTH_CORPUS_KIND",
    "build_training_breadth_corpus",
]

TRAINING_BREADTH_CORPUS_KIND: Final = "training_breadth_v1"

#: Words a breadth wording must not contain, beyond the held-out table's own.
_ALSO_AVOIDED: Final = ("times", "ways", "fit", "group")

#: Each operation's wordings, by name. The name is the wording's group when a
#: fit holds wordings out one at a time.
BREADTH_WORDINGS: Final[dict[str, dict[str, _Template]]] = {
    "add": {
        "top_up_with": _t("{OP:top up}", " ", "{A}", " with ", "{B}"),
        "raise_by": _t("{OP:raise}", " ", "{A}", " by ", "{B}"),
        "augment_by": _t("{OP:augment}", " ", "{A}", " by ", "{B}"),
        "put_together": _t("{OP:put}", " ", "{A}", " and ", "{B}", " together"),
        "throw_in_with": _t("{OP:throw}", " ", "{B}", " in with ", "{A}"),
    },
    "sub": {
        "take_away_from": _t("{OP:take}", " ", "{B}", " away from ", "{A}"),
        "knock_off": _t("{OP:knock}", " ", "{B}", " off ", "{A}"),
        "cut_by": _t("{OP:cut}", " ", "{A}", " by ", "{B}"),
        "remove_from": _t("{OP:remove}", " ", "{B}", " from ", "{A}"),
        "reduce_by": _t("{OP:reduce}", " ", "{A}", " by ", "{B}"),
    },
    "mul": {
        "magnify_fold": _t("{OP:magnify}", " ", "{A}", " ", "{B}", "-fold"),
        "enlarge_fold": _t("{OP:enlarge}", " ", "{A}", " ", "{B}", "-fold"),
        "scale_up_by": _t("{OP:scale}", " ", "{A}", " up by ", "{B}"),
        # "take" also leads a subtraction; only "lots of" settles which.
        "take_lots_of": _t("{OP:take}", " ", "{B}", " lots of ", "{A}"),
    },
    "idiv": {
        "split_into_parts": _t("{OP:split}", " ", "{A}", " into ", "{B}", " equal whole parts"),
        "share_evenly_among": _t(
            "{OP:share}", " ", "{A}", " evenly among ", "{B}", ", keeping only whole shares"
        ),
        "deal_out_to": _t("{OP:deal}", " ", "{A}", " out evenly to ", "{B}", " people, whole items each"),
        "break_into_pieces": _t("{OP:break}", " ", "{A}", " into ", "{B}", " equal pieces, whole numbers only"),
    },
    "at": {
        "pluck_out_of": _t(
            "{OP:pluck}", " the entry under ", "{IF_LITERAL_B:selector }", "{B}", " out of ", "{A}"
        ),
        "read_off": _t("{OP:read off}", " what is at ", "{IF_LITERAL_B:selector }", "{B}", " of ", "{A}"),
        "retrieve_under": _t(
            "{OP:retrieve}", " the element under ", "{IF_LITERAL_B:selector }", "{B}", " from ", "{A}"
        ),
    },
    "count_of": {
        "how_many_shows_up": _t("{OP:find how many times}", " ", "{B}", " shows up in ", "{A}"),
        "how_often_turns_up": _t("{OP:work out how often}", " ", "{B}", " turns up in ", "{A}"),
        "number_found_in": _t("{OP:determine the number of times}", " ", "{B}", " is found in ", "{A}"),
    },
}

_CHAIN_DEPTHS: Final = (3, 4, 5)


def _words(template: _Template) -> str:
    return " ".join(text for kind, text in template if kind != "a" and kind != "b").lower()


def _check_wordings() -> None:
    for op, wordings in BREADTH_WORDINGS.items():
        for name, template in wordings.items():
            words = _words(template)
            clash = [w for w in _VOCABULARY_V2_SIGNATURES if w in words]
            clash += [w for w in _ALSO_AVOIDED if w in words and not (
                w == "times" and op == "count_of")]
            if clash:
                raise ValueError(f"breadth wording {op}/{name} uses held-out words {clash}")


def build_training_breadth_corpus(*, seed: int, requests: int) -> tuple[SemanticProgramExample, ...]:
    """``requests`` training requests, alternating a chain and a two-step request in a new wording.

    Chains use the consumed phrasings at three, four and five steps in turn.
    Two-step requests draw each operation's wording at random from
    BREADTH_WORDINGS; the wording's name is in the construction id.
    """
    if requests < 2:
        raise ValueError("the breadth corpus needs at least one request of each kind")
    _check_wordings()
    rng = random.Random(seed)
    rows: list[SemanticProgramExample] = []
    seen: set[str] = set()
    sample = 0
    chains = 0
    while len(rows) < requests:
        sample += 1
        request = _Request(random.Random(rng.getrandbits(64)))
        if sample % 2:
            depth = _CHAIN_DEPTHS[chains % len(_CHAIN_DEPTHS)]
            _chain(request, depth)
            _then_scaffold(request, _CONSUMED_PHRASES)
            row = request.example("chain", f"depth_{depth}", sample, split="train", family="breadth")
            if row.source_text not in seen:
                chains += 1
        else:
            _two_step(request)
            chosen = {op: request.rng.choice(sorted(wordings)) for op, wordings in BREADTH_WORDINGS.items()}
            phrases = {op: (BREADTH_WORDINGS[op][chosen[op]],) for op in BREADTH_WORDINGS}
            _then_scaffold(request, phrases)
            used = "+".join(chosen[op] for op, _args in request.steps)
            row = request.example("wording", used, sample, split="train", family="breadth")
        if row.source_text in seen:
            continue
        seen.add(row.source_text)
        rows.append(row)
    return tuple(rows)


assert set(_SCALAR_OPS) | {"at", "count_of"} == set(BREADTH_WORDINGS)

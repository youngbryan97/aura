"""Fresh requests for G04: construction, vocabulary, depth and family transfer.

G03's candidate was fitted on 764 training requests, selected on 500 exposed
validation requests and tuned against two five-step composition bundles. Each
stratum here differs from all of those in one declared way and keeps the rest
of what the candidate has met:

* ``vocabulary``: the consumed scaffold ("First, ... Then, ... Return the
  integer result.") with operation phrasings no consumed request or generator
  table has ("take 9 away from 46", "top up 46 with 4", "split 91 into 7 equal
  whole parts", "pluck the entry under selector 2 out of [...]");
* ``construction``: the consumed operation phrasings inside scaffolds no
  consumed request has (a question, numbered steps, a bullet list, a
  supposition), each referring to the first result in its own words;
* ``depth``: the consumed scaffold and phrasings at six and seven steps,
  where everything consumed has two to five (the composition bundles are
  five). Seven is the most the frozen candidate's geometry admits: a chain
  with a new literal at each step has one more input than steps, and the
  candidate takes at most eight inputs, so eight steps would measure a
  declared refusal, not transfer;
* ``family``: programs in which a sequence operation takes a computed argument
  (a selector or a wanted value worked out by arithmetic, or the count of an
  entry another lookup chose); in every consumed program a sequence operation
  takes the request's own inputs.

Every request is split ``test``: it is for measuring, after the candidate is
frozen, and is never fitted on. ``novelty_signatures`` names what a stratum
claims is new so a protocol can check it against the consumed inventory
rather than take it on trust.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Final

from core.learning.procedure_induction import Instruction
from core.learning.semantic_program_corpus import (
    CharacterSpan,
    SemanticInstructionAnnotation,
    SemanticProgramExample,
    _AnnotatedText,
)

__all__ = [
    "G04_STRATA",
    "G04_TRANSFER_CORPUS_KIND",
    "build_g04_transfer_corpus",
    "novelty_signatures",
]

G04_TRANSFER_CORPUS_KIND: Final = "g04_transfer_v1"
G04_STRATA: Final = ("construction", "vocabulary", "depth", "family")

#: Fixed text, the operation's own words ("op"), the first ("a") and second
#: ("b") argument slots, and text said only when the second argument is a
#: literal ("selector 2", never "selector the intermediate value").
_Template = tuple[tuple[str, str], ...]


def _t(*parts: str) -> _Template:
    out = []
    for part in parts:
        if part == "{A}":
            out.append(("a", ""))
        elif part == "{B}":
            out.append(("b", ""))
        elif part.startswith("{OP:"):
            out.append(("op", part[4:-1]))
        elif part.startswith("{IF_LITERAL_B:"):
            out.append(("if_literal_b", part[14:-1]))
        else:
            out.append(("text", part))
    return tuple(out)


#: Phrasings the consumed families use for these operations.
_CONSUMED_PHRASES: Final[dict[str, tuple[_Template, ...]]] = {
    "add": (_t("{OP:add}", " ", "{A}", " and ", "{B}"),),
    "sub": (_t("{OP:subtract}", " ", "{B}", " from ", "{A}"),),
    "mul": (_t("{OP:multiply}", " ", "{A}", " by ", "{B}"),),
    "idiv": (_t("{OP:whole-number divide}", " ", "{A}", " by ", "{B}"),),
    "at": (_t("{OP:select the item at}", " ", "{IF_LITERAL_B:selector }", "{B}", " in ", "{A}"),),
    "count_of": (_t("{OP:count}", " how often ", "{B}", " occurs in ", "{A}"),),
}

#: Phrasings no consumed request or generator table has.
_HELD_OUT_PHRASES: Final[dict[str, tuple[_Template, ...]]] = {
    "add": (
        _t("{OP:top up}", " ", "{A}", " with ", "{B}"),
        _t("{OP:raise}", " ", "{A}", " by ", "{B}"),
    ),
    "sub": (
        _t("{OP:take}", " ", "{B}", " away from ", "{A}"),
        _t("{OP:knock}", " ", "{B}", " off ", "{A}"),
    ),
    "mul": (_t("{OP:magnify}", " ", "{A}", " ", "{B}", "-fold"),),
    "idiv": (
        _t("{OP:split}", " ", "{A}", " into ", "{B}", " equal whole parts"),
        _t("{OP:share}", " ", "{A}", " evenly among ", "{B}", ", keeping only whole shares"),
    ),
    "at": (_t("{OP:pluck}", " the entry under ", "{IF_LITERAL_B:selector }", "{B}", " out of ", "{A}"),),
    "count_of": (_t("{OP:find how many times}", " ", "{B}", " shows up in ", "{A}"),),
}

#: What each stratum claims no consumed request contains, lowercased.
_SIGNATURES: Final[dict[str, tuple[str, ...]]] = {
    "vocabulary": (
        "top up", "raise", "away from", "knock", " off ", "magnify", "-fold",
        "split", "whole parts", "share", "evenly among", "pluck", "shows up",
        "how many times",
    ),
    "construction": (
        "what number do you get", "what you got", "step one", "step two",
        "instructions:", "previous line", "left at the end", "suppose",
        "which number do you have", "reply with",
    ),
}

_SCALAR_OPS: Final = ("add", "sub", "mul", "idiv")


def novelty_signatures(stratum: str) -> tuple[str, ...]:
    """Lowercased phrases a stratum claims no consumed request contains."""
    return _SIGNATURES.get(stratum, ())


def _evaluate(op: str, left: object, right: int) -> int:
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "idiv":
        return left // right
    if op == "at":
        return left[right]
    if op == "count_of":
        return sum(1 for item in left if item == right)
    raise ValueError(f"unknown operation {op}")


#: An argument: ("in", i) is the request's input i, ("step", j) step j's result.
_Ref = tuple[str, int]


class _Request:
    """A program and its rendering, built together so every span is exact.

    Steps name their arguments symbolically; registers (inputs first, then
    each step's result) are assigned once, when the example is made, because
    a literal may be introduced after the step whose result it is combined
    with.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.inputs: list[int | tuple[int, ...]] = []
        self.results: list[int] = []
        self.steps: list[tuple[str, tuple[_Ref, _Ref]]] = []
        self.text = _AnnotatedText()
        self.input_spans: dict[int, CharacterSpan] = {}

    # -- the program --------------------------------------------------------

    def value(self, ref: _Ref) -> int | tuple[int, ...]:
        return self.inputs[ref[1]] if ref[0] == "in" else self.results[ref[1]]

    def _scalars(self) -> set[int]:
        return {v for v in self.inputs if isinstance(v, int)} | set(self.results)

    def fixed(self, value: int | tuple[int, ...]) -> _Ref:
        self.inputs.append(value)
        return ("in", len(self.inputs) - 1)

    def literal(self, low: int, high: int) -> _Ref:
        """A new literal in [low, high], distinct from every value so far.

        Equal values would leave the grader unable to tell which one an
        argument names, so when the range is used up it grows upward.
        """
        return self.fixed(self.draw(low, high))

    def draw(self, low: int, high: int) -> int:
        """A value in [low, high] none of this request's values already has."""
        used = self._scalars()
        while True:
            for _ in range(64):
                value = self.rng.randint(low, high)
                if value not in used:
                    return value
            high += 10

    def sequence(self, items: list[int]) -> _Ref:
        return self.fixed(tuple(items))

    def step(self, op: str, left: _Ref, right: _Ref) -> _Ref:
        self.steps.append((op, (left, right)))
        self.results.append(_evaluate(op, self.value(left), self.value(right)))
        return ("step", len(self.steps) - 1)

    # -- the words ----------------------------------------------------------

    def say(self, text: str) -> None:
        self.text.append(text)

    def render(
        self,
        ordinal: int,
        template: _Template,
        mention: Callable[[_Ref], str],
        *,
        capitalise: bool = False,
    ) -> None:
        _op, args = self.steps[ordinal]
        for index, (kind, words) in enumerate(template):
            if kind == "op":
                if capitalise and index == 0:
                    words = words[:1].upper() + words[1:]
                self.text.append(words, label=f"g04:operation:{ordinal}")
            elif kind == "text":
                self.text.append(words)
            elif kind == "if_literal_b":
                if args[1][0] == "in":
                    self.text.append(words)
            else:
                ref = args[0 if kind == "a" else 1]
                label = f"g04:argument:{ordinal}:{0 if kind == 'a' else 1}"
                if ref[0] == "in":
                    value = self.inputs[ref[1]]
                    rendered = (
                        "[" + ", ".join(str(item) for item in value) + "]"
                        if isinstance(value, tuple)
                        else str(value)
                    )
                    self.text.append(rendered, label=label)
                    if ref[1] in self.input_spans:
                        raise ValueError("every input of a G04 request is said exactly once")
                    self.input_spans[ref[1]] = self.text.span(label)
                else:
                    self.text.append(mention(ref), label=label)

    def example(self, stratum: str, construction: str, sample: int) -> SemanticProgramExample:
        n_inputs = len(self.inputs)
        if sorted(self.input_spans) != list(range(n_inputs)):
            raise ValueError("every input of a G04 request is said exactly once")

        def register(ref: _Ref) -> int:
            return ref[1] if ref[0] == "in" else n_inputs + ref[1]

        instructions = tuple(
            SemanticInstructionAnnotation(
                instruction=Instruction(op, tuple(register(ref) for ref in refs)),
                operation_span=self.text.span(f"g04:operation:{ordinal}"),
                argument_spans=tuple(
                    self.text.span(f"g04:argument:{ordinal}:{position}") for position in range(2)
                ),
                depends_on=tuple(sorted({ref[1] for ref in refs if ref[0] == "step"})),
            )
            for ordinal, (op, refs) in enumerate(self.steps)
        )
        source = self.text.text
        construction_id = f"g04_{stratum}:{construction}"
        identity = f"{construction_id}|{sample}|{source}"
        return SemanticProgramExample(
            example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            construction_id=construction_id,
            topology_id=f"g04_{stratum}_depth{len(self.steps)}",
            split="test",
            source_text=source,
            inputs=tuple(self.inputs),
            input_spans=tuple(self.input_spans[index] for index in range(n_inputs)),
            instructions=instructions,
            report_value=n_inputs + len(self.steps) - 1,
            contrast_id=f"g04:{stratum}:{sample}",
        )


# -- programs -----------------------------------------------------------------


def _scalar_opening(request: _Request, op: str) -> _Ref:
    left = request.literal(12, 99)
    if op in {"idiv", "mul"}:
        right = request.literal(2, 9)
    elif op == "sub":
        right = request.literal(1, request.value(left) - 1)
    else:
        right = request.literal(2, 60)
    return request.step(op, left, right)


def _scalar_continuation(request: _Request, op: str, previous: _Ref) -> _Ref:
    """``op`` on the previous result and one new literal, the result kept positive."""
    value = request.value(previous)
    used = request._scalars()
    if op == "add":
        return request.step(op, previous, request.literal(2, 60))
    if op == "mul":
        return request.step(op, previous, request.literal(2, 9))
    if op == "idiv":
        divisors = [d for d in range(2, min(9, value) + 1) if d not in used]
        if divisors:
            return request.step(op, previous, request.fixed(request.rng.choice(divisors)))
        return request.step(op, request.literal(value * 2 + 10, value * 9 + 60), previous)
    if op == "sub":
        if value > 3 and request.rng.random() < 0.5 and len(used) < value - 1:
            return request.step(op, previous, request.literal(1, min(value - 1, 99)))
        return request.step(op, request.literal(value + 1, value + 60), previous)
    raise ValueError(op)


def _sequence_opening(request: _Request, op: str) -> _Ref:
    """A lookup or a count on an inline list, its second argument a literal."""
    length = request.rng.randint(6, 8)
    items = [request.rng.randint(1, 60) for _ in range(length)]
    if op == "count_of":
        wanted = items[request.rng.randrange(length)]
        for _ in range(request.rng.randint(0, 2)):
            items[request.rng.randrange(length)] = wanted
        return request.step(op, request.sequence(items), request.fixed(wanted))
    sequence = request.sequence(items)
    return request.step(op, sequence, request.fixed(request.rng.randint(1, length - 1)))


def _two_step(request: _Request) -> None:
    first = request.rng.choice(_SCALAR_OPS + ("at", "count_of"))
    if first in _SCALAR_OPS:
        previous = _scalar_opening(request, first)
    else:
        previous = _sequence_opening(request, first)
    _scalar_continuation(request, request.rng.choice(_SCALAR_OPS), previous)


def _chain(request: _Request, depth: int) -> None:
    first = request.rng.choice(_SCALAR_OPS + ("at", "count_of"))
    previous = (
        _scalar_opening(request, first)
        if first in _SCALAR_OPS
        else _sequence_opening(request, first)
    )
    for _ in range(depth - 1):
        previous = _scalar_continuation(request, request.rng.choice(_SCALAR_OPS), previous)


def _family(request: _Request, shape: str) -> None:
    """A sequence operation whose selector or wanted value is a computed result."""
    rng = request.rng
    length = rng.randint(6, 8)
    if shape == "computed_selector":
        selector = rng.randint(2, length - 1)
        # A sum of two different positive numbers needs a selector of 3 or more.
        op = rng.choice(("sub", "add")) if selector >= 3 else "sub"
        while True:
            if op == "sub":
                first = rng.randint(selector + 1, selector + 40)
                second = first - selector
            else:
                first = rng.randint(1, selector - 1)
                second = selector - first
            if first != second:
                break
        computed = request.step(op, request.fixed(first), request.fixed(second))
        items = [rng.randint(1, 60) for _ in range(length)]
        looked_up = request.step("at", request.sequence(items), computed)
    elif shape == "computed_wanted":
        op = rng.choice(("add", "mul"))
        a = request.literal(2, 12)
        b = request.literal(2, 12)
        wanted = _evaluate(op, request.value(a), request.value(b))
        computed = request.step(op, a, b)
        items = [rng.randint(1, 60) for _ in range(length)]
        for position in rng.sample(range(length), rng.randint(1, 3)):
            items[position] = wanted
        looked_up = request.step("count_of", request.sequence(items), computed)
    elif shape == "count_of_lookup":
        items = [rng.randint(1, 30) for _ in range(length)]
        sequence = request.sequence(items)
        selector = rng.randint(1, length - 1)
        chosen = items[selector]
        others = [rng.randint(1, 30) for _ in range(rng.randint(6, 8))]
        for position in rng.sample(range(len(others)), rng.randint(1, 3)):
            others[position] = chosen
        if others == items:
            others.append(chosen)
        picked = request.step("at", sequence, request.fixed(selector))
        looked_up = request.step("count_of", request.sequence(others), picked)
    else:  # pragma: no cover - the builder owns the shape inventory
        raise ValueError(shape)
    if rng.random() < 0.5:
        _scalar_continuation(request, rng.choice(_SCALAR_OPS), looked_up)


# -- words --------------------------------------------------------------------


def _then_scaffold(request: _Request, phrases: dict[str, tuple[_Template, ...]]) -> None:
    """The consumed scaffold: "First, ... Then, ... Return the integer result."."""
    for ordinal, (op, _args) in enumerate(request.steps):
        request.say("First, " if ordinal == 0 else " Then, ")
        template = request.rng.choice(phrases[op])
        request.render(ordinal, template, lambda _register: "the intermediate value")
        request.say(".")
    request.say(" Return the integer result.")


def _construction_scaffold(request: _Request, frame: str) -> None:
    phrase = lambda ordinal: request.rng.choice(_CONSUMED_PHRASES[request.steps[ordinal][0]])  # noqa: E731
    if frame == "question":
        request.say("What number do you get if you ")
        request.render(0, phrase(0), lambda _r: "")
        request.say(" and then ")
        request.render(1, phrase(1), lambda _r: "what you got")
        request.say("?")
    elif frame == "numbered_steps":
        request.say("Step one: ")
        request.render(0, phrase(0), lambda _r: "", capitalise=True)
        request.say(". Step two: ")
        request.render(1, phrase(1), lambda _r: "the step one result", capitalise=True)
        request.say(". Reply with the step two result.")
    elif frame == "bullet_list":
        request.say("Instructions:\n- ")
        request.render(0, phrase(0), lambda _r: "", capitalise=True)
        request.say("\n- ")
        request.render(1, phrase(1), lambda _r: "the number from the previous line", capitalise=True)
        request.say("\nWhat number is left at the end?")
    elif frame == "supposition":
        request.say("Suppose you ")
        request.render(0, phrase(0), lambda _r: "")
        request.say(". If you then ")
        request.render(1, phrase(1), lambda _r: "that number")
        request.say(", which number do you have?")
    else:  # pragma: no cover - the builder owns the frame inventory
        raise ValueError(frame)


_CONSTRUCTION_FRAMES: Final = ("question", "numbered_steps", "bullet_list", "supposition")
_FAMILY_SHAPES: Final = ("computed_selector", "computed_wanted", "count_of_lookup")


def build_g04_transfer_corpus(
    *, seed: int, tasks_per_stratum: int
) -> dict[str, tuple[SemanticProgramExample, ...]]:
    """``tasks_per_stratum`` fresh requests in each of the four strata.

    Deterministic in ``seed``. Constructions within a stratum are taken in
    turn, so each frame, depth or shape is represented equally.
    """
    if tasks_per_stratum < 1:
        raise ValueError("each stratum needs at least one task")
    rng = random.Random(seed)
    strata: dict[str, list[SemanticProgramExample]] = {name: [] for name in G04_STRATA}
    seen: set[str] = set()

    def keep(stratum: str, example: SemanticProgramExample) -> bool:
        if example.source_text in seen:
            return False
        seen.add(example.source_text)
        strata[stratum].append(example)
        return True

    sample = 0
    while any(len(rows) < tasks_per_stratum for rows in strata.values()):
        sample += 1
        if len(strata["vocabulary"]) < tasks_per_stratum:
            request = _Request(random.Random(rng.getrandbits(64)))
            _two_step(request)
            _then_scaffold(request, _HELD_OUT_PHRASES)
            keep("vocabulary", request.example("vocabulary", "held_out_phrasings", sample))
        if len(strata["construction"]) < tasks_per_stratum:
            frame = _CONSTRUCTION_FRAMES[len(strata["construction"]) % len(_CONSTRUCTION_FRAMES)]
            request = _Request(random.Random(rng.getrandbits(64)))
            _two_step(request)
            _construction_scaffold(request, frame)
            keep("construction", request.example("construction", frame, sample))
        if len(strata["depth"]) < tasks_per_stratum:
            depth = 6 + len(strata["depth"]) % 2
            request = _Request(random.Random(rng.getrandbits(64)))
            _chain(request, depth)
            _then_scaffold(request, _CONSUMED_PHRASES)
            keep("depth", request.example("depth", f"chain_{depth}", sample))
        if len(strata["family"]) < tasks_per_stratum:
            shape = _FAMILY_SHAPES[len(strata["family"]) % len(_FAMILY_SHAPES)]
            request = _Request(random.Random(rng.getrandbits(64)))
            _family(request, shape)
            _then_scaffold(request, _CONSUMED_PHRASES)
            keep("family", request.example("family", shape, sample))
    return {name: tuple(rows) for name, rows in strata.items()}

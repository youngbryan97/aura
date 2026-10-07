"""Which register a mention names, read from where that name was given.

After peak recognition and argument ownership, the 19 five-step composition
requests still wrong on 3 October (10 of 48 in one bundle, 9 of 48 in the
other) all fail the same way. A later clause names an earlier result
("multiply refined result by auxiliary result"), and the mention is grounded
to the wrong register: "auxiliary result" to the input "shelf reserve", which
leaves the subtraction that made the auxiliary result nothing to consume but a
later step, so the program comes out reordered with the subtraction last.

The relation head compares a mention with a register's definition, and a step
register is defined at its operation span ("subtract"). A causal model's state
at "subtract" cannot hold a name that is only given in the next sentence
("Save this output as the auxiliary result"), so nothing it scores can tell
the auxiliary result from the primary one.

Where a name was given can be read from the model's own states. For each
register there is a stretch of the request that belongs to it: an input's
declaration ("shelf reserve = 4458"), or an operation's clause and the clause
that names its result, up to the next operation. A mention is compared token by
token with every earlier window of each stretch, in the input embedding (the
same words) and the middle layer (the same words in a like context). The
features are how well the best window matches, whether this is the first
stretch to match that well, and how far it falls behind the best stretch. A
logistic readout over them, fitted on the training rows' annotated arguments
against every other register of the same request, gives
P(mention names register r); its log is one more term in each argument
option's score.

A mention that names nothing ("this output") matches every stretch about
equally, so the readout spreads its probability and the other evidence
decides. A mention whose name is given later (a cataphoric "after removing")
has no earlier window and is treated the same way.

A mention that is an input's literal value is bound exactly by the literal
grammar, and says nothing about where a name was given, so the readout is
neither fitted on nor applied to one.

How the readout enters an argument's score is a choice the readout carries.
"absolute" adds log P(register | mention). That compares different spans for
the same slot unfairly: a span whose distribution is peaked, or a literal
(scored zero), pays less than a name mention for its best register, so on
3 October a request still took two input declarations ("turbine reserve =
6283") as arguments over the names that used them. "relative" adds log P
less the span's own best, so every span's best register scores zero and only
a worse one pays: the readout then says which register a span names, and
nothing about which span to choose. Fitted with them, its margin for a
named intermediate on the five-step requests had a median of 0.55 nats;
without them, 1.2, with all 192 such mentions in each bundle ranked first.

A description repeated in every clause ("the intermediate value") is a name
given nowhere. Its words occur in each earlier clause, so the features above
favour the first clause that has them, and a readout fitted on chains of five
steps or fewer put that first result ahead of the one just made: on G04's six-
and seven-step requests (6 October) it gave the previous step's result log P
of -3 to -5 and the first operation's -0.1 to -0.9, and 54 of 62 programs came
out reordered. A reader resolves such a description to the most recent thing
it fits, and recency is counted in what came between, not in tokens: the same
clause count means the same thing in a long request and a short one. With
``recency``, the readout also counts how many operations begin between an
operation's stretch and the mention, and whether a register's stretch begins
after the mention at all. It can only learn what those are worth if training
has chains long enough for the first clause to match and the most recent
result to differ: in the 764 training requests no result is more than two
operations back, and fitted on them the count came out with the wrong sign.

Which way a mention points is in its own words. "the intermediate value"
points back to the latest result; "the subsequently computed number" points
forward, as "the following" does for a reader. Fitted with breadth chains and
no notion of direction, recency took two cataphoric validation requests and
two reserved-alias ones that v12 had right. So a recency readout carries a
second, small readout: P(the mention names something given after it), from
the mention's mean in the middle layer, fitted on training mentions. The
recency features are gated by it: the count and the after-flag weighted by
P(back), and the after-flag again weighted by P(forward). The antecedent is
fitted on cross-fitted direction estimates, so it does not learn to trust
the direction readout's confidence on rows that readout was fitted on.

Training rows only; validation and test rows are refused by the fitter.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_ir import TokenSpan

ANTECEDENT_SCHEMA: Final = "aura.semantic_argument_antecedent.v2"
#: The schema of a readout that also reads recency, ungated (UNGATED_RECENCY_FEATURES;
#: candidate v13, 6 October), and of one whose recency a mention direction gates.
UNGATED_RECENCY_ANTECEDENT_SCHEMA: Final = "aura.semantic_argument_antecedent.v3"
RECENCY_ANTECEDENT_SCHEMA: Final = "aura.semantic_argument_antecedent.v4"

#: The hidden-state channels a mention is compared in, as the bundles name them.
CHANNELS: Final = ("input_token_embedding", "middle_causal_hidden")

#: Names of the features, in the order the weights read them.
FEATURES: Final = (
    "match_embedding",
    "match_middle",
    "first_embedding",
    "first_middle",
    "behind_best_embedding",
    "behind_best_middle",
    "is_input",
    "no_earlier_window",
    "identical_share",
)

#: Read by a v3 readout after FEATURES.
UNGATED_RECENCY_FEATURES: Final = ("operations_between", "defined_after")
#: Read by a v4 readout after FEATURES.
RECENCY_FEATURES: Final = (
    "operations_between_if_back",
    "defined_after_if_back",
    "defined_after_if_forward",
)
#: The channel a mention's direction is read from.
DIRECTION_CHANNEL: Final = "middle_causal_hidden"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _channel(hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], name: str) -> np.ndarray:
    if name not in channels:
        raise ValueError(f"antecedent reading needs the {name} channel")
    offsets = np.cumsum((0, *widths))
    index = list(channels).index(name)
    block = np.asarray(hidden[:, offsets[index] : offsets[index + 1]], dtype=np.float64)
    return block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-9)


def sentence_starts(
    token_ids: Sequence[int], sentence_end_ids: Sequence[int], input_spans: Sequence[TokenSpan] = ()
) -> tuple[int, ...]:
    """Where each sentence of a request begins: after a sentence-ending token, outside a literal.

    A full stop inside an input's value ("6.5") ends no sentence.
    """
    ends = frozenset(int(token) for token in sentence_end_ids)
    starts = [0]
    for index, token in enumerate(token_ids):
        inside = any(span.start <= index < span.end for span in input_spans)
        if int(token) in ends and not inside and index + 1 < len(token_ids):
            starts.append(index + 1)
    return tuple(starts)


def register_stretches(
    input_spans: Sequence[TokenSpan],
    operation_spans: Sequence[TokenSpan],
    token_count: int,
    sentences: Sequence[int] = (),
) -> tuple[tuple[int, int], ...]:
    """The part of the request each register owns, inputs first, then operations.

    An input owns its declaration, from the end of the input or operation
    before it in the text; an operation owns its clause and the one naming its
    result, up to the next operation in the text. A declaration holds no
    operation: in "the whole-number quotient of the whole-number quotient of
    88 divided by 41" the first input's stretch ran from the start of the
    sentence to "88", operations and all, and a later "the" was read back to
    it (3 October, the readout's two training losses).

    With ``sentences`` (sentence_starts), an operation that opens its sentence
    owns the whole sentence, from after the last input declared in it. "Form
    the lead calculation by subtract return flow from intake flow" names the
    subtraction's result before the operation's word, and those tokens
    belonged to no register, so "the lead calculation", used three sentences
    later, could not be read back to the subtraction (scalar_branch_weave_five,
    5 October). Only an operation alone in its sentence takes the lead-in: in
    "the whole-number quotient of the whole-number quotient of 88 divided by
    41" every operation starts at its own word, or "Return the" goes to the
    first and a later "the" is read back to it (three arithmetic:nominal_nested
    training rows lost, 5 October, run v7).
    """
    stretches: list[tuple[int, int]] = []
    boundaries = sorted({span.end for span in input_spans} | {span.end for span in operation_spans})
    for span in input_spans:
        before = [end for end in boundaries if end <= span.start]
        stretches.append((max(before) if before else 0, span.end))
    input_ends = sorted(span.end for span in input_spans)
    begins: list[int] = []
    for span in operation_spans:
        sentence = max((start for start in sentences if start <= span.start), default=None)
        following = min((start for start in sentences if start > span.start), default=token_count)
        shares = sentence is not None and any(
            other is not span and sentence <= other.start < following for other in operation_spans
        )
        if sentence is None or shares:
            begins.append(span.start)
        else:
            declared = [end for end in input_ends if sentence < end <= span.start]
            begins.append(max([sentence, *declared]))
    for begin, span in zip(begins, operation_spans, strict=True):
        later = [other for other, op in zip(begins, operation_spans, strict=True) if op.start > span.start]
        stretches.append((begin, min(later) if later else token_count))
    return tuple(stretches)


class _Similarities:
    """Token-by-token similarity of one request with itself, in each channel."""

    def __init__(self, hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]) -> None:
        self.matrices = tuple(
            (block @ block.T) for block in (_channel(hidden, channels, widths, name) for name in CHANNELS)
        )
        self.direction_rows = _channel(hidden, channels, widths, DIRECTION_CHANNEL)
        self.token_count = len(hidden)

    def mention_vector(self, mention: TokenSpan) -> np.ndarray:
        """The mention's mean in the direction channel, unit length."""
        mean = self.direction_rows[mention.start : mention.end].mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)

    def window_means(self, mention: TokenSpan) -> tuple[np.ndarray, ...]:
        """Mean aligned similarity of the mention to each window that ends before it, by start."""
        length = mention.end - mention.start
        starts = mention.start - length + 1
        if length <= 0 or starts <= 0:
            return tuple(np.zeros(0) for _ in self.matrices)
        return tuple(
            sum(matrix[mention.start + offset, offset : offset + starts] for offset in range(length)) / length
            for matrix in self.matrices
        )


#: The last request's similarities. The decoder assigns arguments once per
#: candidate chart over the same states, so they are computed once per request.
#: The entry holds the states themselves, so their identity cannot be reused.
_last_similarities: list[Any] = []
_last_similarities_lock = threading.Lock()


def _similarities_for(hidden: Any, channels: Sequence[str], widths: Sequence[int]) -> _Similarities:
    key = (tuple(channels), tuple(widths))
    with _last_similarities_lock:
        if _last_similarities and _last_similarities[0] is hidden and _last_similarities[1] == key:
            return _last_similarities[2]
    similarities = _Similarities(np.asarray(hidden), channels, widths)
    with _last_similarities_lock:
        _last_similarities[:] = [hidden, key, similarities]
    return similarities


def antecedent_features(
    similarities: _Similarities,
    mention: TokenSpan,
    stretches: Sequence[tuple[int, int]],
    input_count: int,
    *,
    recency: bool = False,
    forward: float | None = None,
) -> list[list[float]]:
    """One feature row per register for ``mention``, then the recency features ``recency`` asks for.

    With ``forward``, P(the mention names something given after it), they are
    RECENCY_FEATURES; without it, the ungated UNGATED_RECENCY_FEATURES of v3.
    """
    if recency and forward is not None and not 0.0 <= forward <= 1.0:
        raise ValueError("a mention's direction is a probability")
    length = mention.end - mention.start
    means = similarities.window_means(mention)
    matches: list[list[float | None]] = []
    for low, high in stretches:
        last_start = min(high, mention.start) - length
        row: list[float | None] = []
        for values in means:
            if last_start < low or low >= len(values):
                row.append(None)
            else:
                row.append(float(np.max(values[low : min(last_start, len(values) - 1) + 1])))
        matches.append(row)
    order = sorted(range(len(stretches)), key=lambda index: stretches[index][0])
    # Where the mention's exact words occur earlier, and what share of those
    # places lies in each stretch. "auxiliary result" occurs once, in the
    # clause that named it; "the" occurs everywhere, so no stretch holds much
    # of it (LIVE 2026-10-03: a one-token "the" was bound to an input by its
    # exact match with an earlier "the").
    identical = means[0] >= 1.0 - 1e-6 if len(means[0]) else np.zeros(0, dtype=bool)
    everywhere = int(np.sum(identical))
    rows = []
    for register, (low, _high) in enumerate(stretches):
        row: list[float] = []
        firsts: list[float] = []
        behinds: list[float] = []
        for channel in range(len(CHANNELS)):
            value = matches[register][channel]
            present = [m[channel] for m in matches if m[channel] is not None]
            earlier = [
                matches[other][channel]
                for other in order
                if stretches[other][0] < low and matches[other][channel] is not None
            ]
            if value is None:
                row.append(0.0)
                firsts.append(0.0)
                behinds.append(0.0)
                continue
            row.append(value)
            firsts.append(value - max(earlier) if earlier else value)
            behinds.append(value - max(present))
        last_start = min(stretches[register][1], mention.start) - length
        here = int(np.sum(identical[low : last_start + 1])) if last_start >= low else 0
        rows.append(
            [
                *row,
                *firsts,
                *behinds,
                float(register < input_count),
                float(all(value is None for value in matches[register])),
                here / everywhere if everywhere else 0.0,
            ]
        )
        if recency:
            # Recency orders results; an input is given, and named or written out.
            between = 0 if register < input_count else sum(
                1 for other in range(input_count, len(stretches))
                if other != register and low < stretches[other][0] < mention.start
            )
            after = float(low >= mention.end)
            if forward is None:
                rows[-1].extend((float(between), after))
            else:
                back = 1.0 - forward
                rows[-1].extend((between * back, after * back, after * forward))
    return rows


@dataclass(frozen=True)
class MentionDirection:
    """P(a mention names something given after it), from the mention's own words in context."""

    weight: tuple[float, ...]
    bias: float

    def __post_init__(self) -> None:
        if not self.weight or not all(math.isfinite(value) for value in (*self.weight, self.bias)):
            raise ValueError("mention direction parameters are invalid")

    def forward(self, vector: np.ndarray) -> float:
        logit = float(np.asarray(vector, dtype=np.float64) @ np.asarray(self.weight) + self.bias)
        return 1.0 / (1.0 + math.exp(-logit)) if logit > -700 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"channel": DIRECTION_CHANNEL, "weight": list(self.weight), "bias": self.bias}


def mention_direction_from_dict(value: Mapping[str, Any]) -> MentionDirection:
    if value.get("channel") != DIRECTION_CHANNEL:
        raise ValueError("mention direction payload reads another channel")
    return MentionDirection(tuple(float(item) for item in value["weight"]), float(value["bias"]))


@dataclass(frozen=True)
class ArgumentAntecedent:
    """P(mention names register), from where the request gave each name."""

    weight: tuple[float, ...]
    bias: float
    fit_receipt: Mapping[str, Any]
    #: "absolute" (log P) or "relative" (log P less the span's best register's).
    scoring: str = "absolute"
    #: Whether a mention the readout says most probably names an operation's
    #: own result is kept out of that operation's arguments. "Save that output
    #: as the primary result" names the subtraction just made; on
    #: scalar_branch_weave_five-0-1 the subtraction took it as its minuend.
    own_result_is_not_an_input: bool = False
    #: Whether an input the request refers to by name is used through the name
    #: and not through its literal declaration. On scalar_branch_weave_five-0-0
    #: "turbine reserve = 6283" and "83651" were taken as arguments over
    #: "turbine reserve" and "return flow", the names that use them.
    named_inputs_are_used_by_name: bool = False
    #: The tokenizer's sentence-ending tokens. When present, an operation's
    #: register owns its whole sentence (register_stretches).
    sentence_end_token_ids: tuple[int, ...] = ()
    #: Whether an operation's argument options start no earlier than its own
    #: sentence (within_sentence, below). Needs
    #: sentence_end_token_ids.
    arguments_within_sentence: bool = False
    #: Whether the readout also reads RECENCY_FEATURES, gated by ``direction``.
    recency: bool = False
    direction: MentionDirection | None = None

    @property
    def features(self) -> tuple[str, ...]:
        if not self.recency:
            return FEATURES
        return (*FEATURES, *(RECENCY_FEATURES if self.direction is not None else UNGATED_RECENCY_FEATURES))

    @property
    def schema(self) -> str:
        if not self.recency:
            return ANTECEDENT_SCHEMA
        return RECENCY_ANTECEDENT_SCHEMA if self.direction is not None else UNGATED_RECENCY_ANTECEDENT_SCHEMA

    def __post_init__(self) -> None:
        if len(self.weight) != len(self.features) or not all(
            math.isfinite(value) for value in (*self.weight, self.bias)
        ):
            raise ValueError("argument antecedent parameters are invalid")
        if self.scoring not in ("absolute", "relative"):
            raise ValueError("argument antecedent scoring is absolute or relative")
        if self.direction is not None and not self.recency:
            raise ValueError("only a recency readout carries a mention direction")

    def scorer(
        self,
        hidden: np.ndarray,
        channels: Sequence[str],
        widths: Sequence[int],
        input_spans: Sequence[TokenSpan],
        operation_spans: Sequence[TokenSpan],
        source_token_ids: Sequence[int] | None = None,
    ) -> _AntecedentScorer:
        """Log P(register | mention) for one request, computed once per mention."""
        sentences = (
            sentence_starts(source_token_ids, self.sentence_end_token_ids, input_spans)
            if self.sentence_end_token_ids and source_token_ids is not None
            else ()
        )
        return _AntecedentScorer(
            self,
            _similarities_for(hidden, channels, widths),
            register_stretches(input_spans, operation_spans, len(hidden), sentences),
            input_spans,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "features": list(self.features),
            "channels": list(CHANNELS),
            "weight": list(self.weight),
            "bias": self.bias,
            "fit_receipt": dict(self.fit_receipt),
            **({"scoring": self.scoring} if self.scoring != "absolute" else {}),
            **({"own_result_is_not_an_input": True} if self.own_result_is_not_an_input else {}),
            **({"named_inputs_are_used_by_name": True} if self.named_inputs_are_used_by_name else {}),
            **(
                {"sentence_end_token_ids": list(self.sentence_end_token_ids)}
                if self.sentence_end_token_ids else {}
            ),
            **({"arguments_within_sentence": True} if self.arguments_within_sentence else {}),
            **({"direction": self.direction.to_dict()} if self.direction is not None else {}),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_dict())


def _literal(mention: TokenSpan, input_spans: Sequence[TokenSpan]) -> bool:
    return any(mention.start < span.end and span.start < mention.end for span in input_spans)


class _AntecedentScorer:
    def __init__(
        self,
        readout: ArgumentAntecedent,
        similarities: _Similarities,
        stretches: tuple[tuple[int, int], ...],
        input_spans: Sequence[TokenSpan],
    ) -> None:
        self.readout = readout
        self.similarities = similarities
        self.stretches = stretches
        self.input_spans = tuple(input_spans)
        self.input_count = len(self.input_spans)
        self.cache: dict[TokenSpan, tuple[float, ...]] = {}

    def log_probabilities(self, mention: TokenSpan) -> tuple[float, ...]:
        """Log P(register | mention); no evidence (all zero) for an input's literal value."""
        if _literal(mention, self.input_spans):
            return tuple(0.0 for _ in self.stretches)
        if mention not in self.cache:
            forward = (
                self.readout.direction.forward(self.similarities.mention_vector(mention))
                if self.readout.direction is not None else None
            )
            rows = np.asarray(
                antecedent_features(self.similarities, mention, self.stretches, self.input_count,
                                    recency=self.readout.recency, forward=forward)
            )
            logits = rows @ np.asarray(self.readout.weight) + self.readout.bias
            top = float(np.max(logits))
            total = top + math.log(float(np.sum(np.exp(logits - top))))
            self.cache[mention] = tuple(float(value - total) for value in logits)
        return self.cache[mention]

    def names_own_result(self, mention: TokenSpan, own_register: int) -> bool:
        """Whether the readout's most probable referent for ``mention`` is ``own_register``.

        Only when the readout carries that rule; never for an input's literal value.
        """
        if not self.readout.own_result_is_not_an_input or _literal(mention, self.input_spans):
            return False
        values = self.log_probabilities(mention)
        return int(np.argmax(values)) == own_register

    def inputs_used_by_name(self, mentions: Sequence[TokenSpan]) -> frozenset[int]:
        """Inputs some non-literal mention most probably names; empty unless the readout carries the rule."""
        if not self.readout.named_inputs_are_used_by_name:
            return frozenset()
        named = set()
        for mention in mentions:
            if _literal(mention, self.input_spans):
                continue
            best = int(np.argmax(self.log_probabilities(mention)))
            if best < self.input_count:
                named.add(best)
        return frozenset(named)

    def is_declaration_of(self, mention: TokenSpan, named: frozenset[int]) -> bool:
        """Whether ``mention`` is the literal of an input in ``named``."""
        return any(
            index in named and mention.start < span.end and span.start < mention.end
            for index, span in enumerate(self.input_spans)
        )

    def score(self, mention: TokenSpan, register: int) -> float:
        """The term added to an argument option's score, by the readout's scoring."""
        values = self.log_probabilities(mention)
        if self.readout.scoring == "relative":
            return values[register] - max(values)
        return values[register]


def argument_antecedent_from_dict(value: Mapping[str, Any]) -> ArgumentAntecedent:
    extra = {ANTECEDENT_SCHEMA: (), UNGATED_RECENCY_ANTECEDENT_SCHEMA: UNGATED_RECENCY_FEATURES,
             RECENCY_ANTECEDENT_SCHEMA: RECENCY_FEATURES}
    schema = value.get("schema")
    recency = schema in (UNGATED_RECENCY_ANTECEDENT_SCHEMA, RECENCY_ANTECEDENT_SCHEMA)
    if (
        schema not in extra
        or list(value.get("features", ())) != [*FEATURES, *extra[schema]]
        or list(value.get("channels", ())) != list(CHANNELS)
        or (schema == RECENCY_ANTECEDENT_SCHEMA) != ("direction" in value)
    ):
        raise ValueError("argument antecedent payload is not this schema")
    return ArgumentAntecedent(
        tuple(float(item) for item in value["weight"]),
        float(value["bias"]),
        dict(value["fit_receipt"]),
        str(value.get("scoring", "absolute")),
        bool(value.get("own_result_is_not_an_input", False)),
        bool(value.get("named_inputs_are_used_by_name", False)),
        tuple(int(token) for token in value.get("sentence_end_token_ids", ())),
        bool(value.get("arguments_within_sentence", False)),
        recency,
        mention_direction_from_dict(value["direction"]) if "direction" in value else None,
    )


def _item_stretches(item: Any, sentence_end_token_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ir = item.ir
    operations = [instruction.operation_span for instruction in ir.instructions]
    sentences = (
        sentence_starts(ir.source_token_ids, sentence_end_token_ids, ir.input_spans)
        if sentence_end_token_ids else ()
    )
    return register_stretches(ir.input_spans, operations, len(item.hidden_states), sentences)


def antecedent_training_groups(
    item: Any, sentence_end_token_ids: Sequence[int] = (), *, direction: MentionDirection | None = None,
    recency: bool | None = None,
) -> list[tuple[list[list[float]], int]]:
    """Per annotated mention: the feature rows of the registers its operation could read, and the gold one's index."""
    ir = item.ir
    similarities = _Similarities(
        np.asarray(item.hidden_states), item.hidden_channels, item.hidden_channel_widths
    )
    stretches = _item_stretches(item, sentence_end_token_ids)
    groups: list[tuple[list[list[float]], int]] = []
    for step, instruction in enumerate(ir.instructions):
        own = ir.n_inputs + step
        for register, mention in zip(instruction.args, instruction.argument_spans, strict=True):
            if _literal(mention, ir.input_spans):
                continue
            features = antecedent_features(
                similarities, mention, stretches, ir.n_inputs, recency=(direction is not None) if recency is None else recency,
                forward=None if direction is None else direction.forward(similarities.mention_vector(mention)))
            candidates = [index for index in range(len(features)) if index != own]
            groups.append(([features[index] for index in candidates], candidates.index(register)))
    return groups


def _fit_conditional(
    groups: Sequence[tuple[list[list[float]], int]], weights: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Weights maximising each mention's log-probability of its own register among its candidates.

    The readout is used as a distribution over a mention's registers, so it is
    fitted as one (a conditional logit), with the same unit L2 penalty the
    pairwise fit carried. The bias cancels in the softmax and stays zero.
    ``weights`` weighs each mention's term (family_weights).
    """
    from scipy.optimize import minimize

    blocks = [np.asarray(rows, dtype=np.float64) for rows, _gold in groups]
    golds = [gold for _rows, gold in groups]
    scale = [1.0] * len(blocks) if weights is None else [float(value) for value in weights]
    if len(scale) != len(blocks) or not all(math.isfinite(value) and value > 0 for value in scale):
        raise ValueError("each mention needs a positive weight")

    def loss(weight: np.ndarray) -> tuple[float, np.ndarray]:
        total = 0.5 * float(weight @ weight)
        gradient = weight.copy()
        for rows, gold, factor in zip(blocks, golds, scale, strict=True):
            logits = rows @ weight
            top = float(np.max(logits))
            exp = np.exp(logits - top)
            norm = float(np.sum(exp))
            total -= factor * float(logits[gold] - top - math.log(norm))
            gradient -= factor * (rows[gold] - (exp / norm) @ rows)
        return total, gradient

    result = minimize(loss, np.zeros(blocks[0].shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return tuple(float(value) for value in result.x)


def antecedent_training_rows(
    item: Any, sentence_end_token_ids: Sequence[int] = (), *, direction: MentionDirection | None = None,
    recency: bool | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Each annotated argument mention against every register its operation could read."""
    ir = item.ir
    similarities = _Similarities(
        np.asarray(item.hidden_states), item.hidden_channels, item.hidden_channel_widths
    )
    stretches = _item_stretches(item, sentence_end_token_ids)
    rows: list[list[float]] = []
    labels: list[int] = []
    for step, instruction in enumerate(ir.instructions):
        own = ir.n_inputs + step
        for register, mention in zip(instruction.args, instruction.argument_spans, strict=True):
            if _literal(mention, ir.input_spans):
                continue
            features = antecedent_features(
                similarities, mention, stretches, ir.n_inputs, recency=(direction is not None) if recency is None else recency,
                forward=None if direction is None else direction.forward(similarities.mention_vector(mention)))
            for candidate, row in enumerate(features):
                if candidate == own:
                    continue
                rows.append(row)
                labels.append(int(candidate == register))
    return rows, labels


def _direction_rows(examples: Sequence[Any], ends: Sequence[int]) -> tuple[list[np.ndarray], list[int]]:
    """Each training argument mention's vector, and whether what it names is given after it."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for item in examples:
        ir = item.ir
        block = _channel(np.asarray(item.hidden_states), item.hidden_channels, item.hidden_channel_widths,
                         DIRECTION_CHANNEL)
        stretches = _item_stretches(item, ends)
        for instruction in ir.instructions:
            for register, mention in zip(instruction.args, instruction.argument_spans, strict=True):
                if _literal(mention, ir.input_spans):
                    continue
                mean = block[mention.start : mention.end].mean(axis=0)
                rows.append(mean / (np.linalg.norm(mean) + 1e-9))
                labels.append(int(stretches[register][0] >= mention.end))
    return rows, labels


def fit_mention_direction(examples: Sequence[Any], sentence_end_token_ids: Sequence[int] = ()) -> MentionDirection:
    """P(a mention names something given after it), fitted on training mentions only."""
    from sklearn.linear_model import LogisticRegression

    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("mention direction is fitted on training rows only")
    rows, labels = _direction_rows(examples, tuple(sentence_end_token_ids))
    if not rows:
        raise ValueError("mention direction needs annotated mentions")
    if len(set(labels)) < 2:
        # Every mention points one way: the rule of succession, with nothing to read from the words.
        forward = (sum(labels) + 1) / (len(labels) + 2)
        return MentionDirection(tuple(0.0 for _ in rows[0]), math.log(forward / (1.0 - forward)))
    fitted = LogisticRegression(max_iter=3000, C=1.0).fit(np.stack(rows), np.asarray(labels))
    return MentionDirection(tuple(float(value) for value in fitted.coef_[0]), float(fitted.intercept_[0]))


def _cross_fitted_directions(
    examples: Sequence[Any], ends: Sequence[int], folds: int = 3
) -> dict[str, MentionDirection]:
    """For each source, a direction readout fitted without it, so its estimates are out of sample."""
    assignment = {
        item.ir.source_text_sha256: int(hashlib.sha256(item.ir.source_text_sha256.encode()).hexdigest()[:8], 16) % folds
        for item in examples
    }
    readouts = {
        fold: fit_mention_direction([item for item in examples if assignment[item.ir.source_text_sha256] != fold], ends)
        for fold in range(folds)
    }
    return {source: readouts[fold] for source, fold in assignment.items()}


def family_weights(families: Sequence[str]) -> list[float]:
    """Each construction family the same total weight, the mean weight one.

    A family is a kind of sentence; how many requests a generator made of it
    is not. Pooled, 300 breadth chains outweighed every cataphoric training
    mention, and the fit gave up two cataphoric validation requests to read
    the chains (candidate v13, 6 October).
    """
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    return [len(families) / (len(counts) * counts[family]) for family in families]


def fit_argument_antecedent(
    examples: Sequence[Any],
    *,
    objective: str = "pairwise",
    sentence_end_token_ids: Sequence[int] = (),
    recency: bool = False,
    direction: bool = True,
    balance_families: bool = False,
) -> ArgumentAntecedent:
    """Fit on each training argument mention against every other register of its request.

    ``objective`` "pairwise" fits each (mention, register) pair as a yes or no;
    "conditional" fits each mention's distribution over its registers, which is
    how the readout is used. ``recency`` adds RECENCY_FEATURES.
    """
    from sklearn.linear_model import LogisticRegression

    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("argument antecedents are fitted on training rows only")
    if objective not in ("pairwise", "conditional"):
        raise ValueError("argument antecedents are fitted pairwise or conditionally")
    if objective == "conditional":
        ends = tuple(int(token) for token in sentence_end_token_ids)
        gated = recency and direction
        held = _cross_fitted_directions(examples, ends) if gated else {}
        groups, families = [], []
        for item in examples:
            item_groups = antecedent_training_groups(
                item, ends, direction=held.get(item.ir.source_text_sha256), recency=recency)
            groups.extend(item_groups)
            families.extend(str(getattr(item, "construction_id", "")).split(":")[0] for _ in item_groups)
        if not groups:
            raise ValueError("argument antecedents need annotated mentions")
        receipt = {
            "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
            "training_rows": len(examples),
            "mentions": len(groups),
            "objective": "conditional",
            "splits_used": ["train"],
            **({"stretches": "sentence"} if ends else {}),
            **({"recency": "gated" if gated else "ungated"} if recency else {}),
            **({"balance": "construction_family"} if balance_families else {}),
        }
        return ArgumentAntecedent(
            _fit_conditional(groups, family_weights(families) if balance_families else None), 0.0,
            {**receipt, "receipt_sha256": _sha(receipt)},
            sentence_end_token_ids=ends, recency=recency,
            direction=fit_mention_direction(examples, ends) if gated else None,
        )
    gated = recency and direction
    held = _cross_fitted_directions(examples, tuple(sentence_end_token_ids)) if gated else {}
    rows: list[list[float]] = []
    labels: list[int] = []
    for item in examples:
        item_rows, item_labels = antecedent_training_rows(
            item, tuple(sentence_end_token_ids), direction=held.get(item.ir.source_text_sha256), recency=recency)
        rows.extend(item_rows)
        labels.extend(item_labels)
    if len(set(labels)) < 2:
        raise ValueError("argument antecedents need requests with more than one register")
    fitted = LogisticRegression(max_iter=2000, C=1.0).fit(np.asarray(rows), np.asarray(labels))
    receipt = {
        "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
        "training_rows": len(examples),
        "pairs": len(labels),
        "named_pairs": int(sum(labels)),
        "splits_used": ["train"],
        **({"recency": "gated" if gated else "ungated"} if recency else {}),
    }
    return ArgumentAntecedent(
        tuple(float(value) for value in fitted.coef_[0]),
        float(fitted.intercept_[0]),
        {**receipt, "receipt_sha256": _sha(receipt)},
        sentence_end_token_ids=tuple(int(token) for token in sentence_end_token_ids),
        recency=recency,
        direction=fit_mention_direction(examples, sentence_end_token_ids) if gated else None,
    )


def argument_sentences(
    argument_antecedent: Any, source_token_ids: Sequence[int] | None, input_spans: Sequence[TokenSpan]
) -> tuple[int, ...]:
    """Where the request's sentences start, when the readout bounds arguments by them."""
    if (
        not argument_antecedent
        or not getattr(argument_antecedent, "arguments_within_sentence", False)
        or not argument_antecedent.sentence_end_token_ids
        or source_token_ids is None
    ):
        return ()
    return sentence_starts(source_token_ids, argument_antecedent.sentence_end_token_ids, input_spans)


def within_sentence(
    by_operation: Sequence[tuple[tuple[TokenSpan, float], ...]],
    operation_nodes: Sequence[Any],
    sentences: Sequence[int],
) -> tuple[tuple[tuple[TokenSpan, float], ...], ...]:
    """Drop each operation's options that start before its sentence or the operation before it.

    scalar_branch_weave_five-0-0: "Build the primary path by add intake flow
    and return flow" took "turbine reserve = 6283", from the declarations two
    sentences earlier, as its second argument, and the subtraction took
    return flow's literal "83651". Every input literal is offered to every
    operation, and the first operation's clause began at token 0. Across
    train, validation, test and both composition bundles (1,860 requests) no
    annotated argument starts before its operation's sentence or the
    operation before it; 96 nominal_nested arguments come after the clause,
    and those are kept.
    """
    ordered = sorted(range(len(operation_nodes)), key=lambda index: operation_nodes[index].span.start)
    bounded = list(by_operation)
    for position, index in enumerate(ordered):
        span = operation_nodes[index].span
        previous = operation_nodes[ordered[position - 1]].span.end if position else 0
        sentence = max((start for start in sentences if start <= span.start), default=0)
        floor = max(previous, sentence)
        bounded[index] = tuple(item for item in by_operation[index] if item[0].start >= floor)
    return tuple(bounded)


__all__ = [
    "ANTECEDENT_SCHEMA",
    "MentionDirection",
    "fit_mention_direction",
    "RECENCY_ANTECEDENT_SCHEMA",
    "RECENCY_FEATURES",
    "UNGATED_RECENCY_ANTECEDENT_SCHEMA",
    "UNGATED_RECENCY_FEATURES",
    "ArgumentAntecedent",
    "argument_sentences",
    "antecedent_features",
    "antecedent_training_rows",
    "argument_antecedent_from_dict",
    "fit_argument_antecedent",
    "register_stretches",
    "sentence_end_token_ids",
    "sentence_starts",
    "within_sentence",
]


def sentence_end_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """The tokenizer's tokens that end a sentence: closing punctuation around a ".", "?" or "!"."""
    vocabulary = tokenizer.get_vocab()
    ends = []
    for token_id in sorted(vocabulary.values()):
        text = tokenizer.decode([token_id]).strip()
        if text and set(text) <= set(".?!\"'”’)") and set(text) & set(".?!"):
            ends.append(int(token_id))
    if not ends:
        raise ValueError("the tokenizer has no sentence-ending token")
    return tuple(ends)


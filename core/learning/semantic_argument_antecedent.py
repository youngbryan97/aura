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


def register_stretches(
    input_spans: Sequence[TokenSpan], operation_spans: Sequence[TokenSpan], token_count: int
) -> tuple[tuple[int, int], ...]:
    """The part of the request each register owns, inputs first, then operations.

    An input owns its declaration, from the end of the input or operation
    before it in the text; an operation owns its clause and the one naming its
    result, up to the next operation in the text. A declaration holds no
    operation: in "the whole-number quotient of the whole-number quotient of
    88 divided by 41" the first input's stretch ran from the start of the
    sentence to "88", operations and all, and a later "the" was read back to
    it (3 October, the readout's two training losses).
    """
    stretches: list[tuple[int, int]] = []
    boundaries = sorted({span.end for span in input_spans} | {span.end for span in operation_spans})
    for span in input_spans:
        before = [end for end in boundaries if end <= span.start]
        stretches.append((max(before) if before else 0, span.end))
    starts = sorted(span.start for span in operation_spans)
    for span in operation_spans:
        later = [start for start in starts if start > span.start]
        stretches.append((span.start, min(later) if later else token_count))
    return tuple(stretches)


class _Similarities:
    """Token-by-token similarity of one request with itself, in each channel."""

    def __init__(self, hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]) -> None:
        self.matrices = tuple(
            (block @ block.T) for block in (_channel(hidden, channels, widths, name) for name in CHANNELS)
        )
        self.token_count = len(hidden)

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
) -> list[list[float]]:
    """One feature row per register for ``mention``."""
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
    return rows


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

    def __post_init__(self) -> None:
        if len(self.weight) != len(FEATURES) or not all(
            math.isfinite(value) for value in (*self.weight, self.bias)
        ):
            raise ValueError("argument antecedent parameters are invalid")
        if self.scoring not in ("absolute", "relative"):
            raise ValueError("argument antecedent scoring is absolute or relative")

    def scorer(
        self,
        hidden: np.ndarray,
        channels: Sequence[str],
        widths: Sequence[int],
        input_spans: Sequence[TokenSpan],
        operation_spans: Sequence[TokenSpan],
    ) -> _AntecedentScorer:
        """Log P(register | mention) for one request, computed once per mention."""
        return _AntecedentScorer(
            self,
            _similarities_for(hidden, channels, widths),
            register_stretches(input_spans, operation_spans, len(hidden)),
            input_spans,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANTECEDENT_SCHEMA,
            "features": list(FEATURES),
            "channels": list(CHANNELS),
            "weight": list(self.weight),
            "bias": self.bias,
            "fit_receipt": dict(self.fit_receipt),
            **({"scoring": self.scoring} if self.scoring != "absolute" else {}),
            **({"own_result_is_not_an_input": True} if self.own_result_is_not_an_input else {}),
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
            rows = np.asarray(
                antecedent_features(self.similarities, mention, self.stretches, self.input_count)
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

    def score(self, mention: TokenSpan, register: int) -> float:
        """The term added to an argument option's score, by the readout's scoring."""
        values = self.log_probabilities(mention)
        if self.readout.scoring == "relative":
            return values[register] - max(values)
        return values[register]


def argument_antecedent_from_dict(value: Mapping[str, Any]) -> ArgumentAntecedent:
    if (
        value.get("schema") != ANTECEDENT_SCHEMA
        or list(value.get("features", ())) != list(FEATURES)
        or list(value.get("channels", ())) != list(CHANNELS)
    ):
        raise ValueError("argument antecedent payload is not this schema")
    return ArgumentAntecedent(
        tuple(float(item) for item in value["weight"]),
        float(value["bias"]),
        dict(value["fit_receipt"]),
        str(value.get("scoring", "absolute")),
        bool(value.get("own_result_is_not_an_input", False)),
    )


def antecedent_training_groups(item: Any) -> list[tuple[list[list[float]], int]]:
    """Per annotated mention: the feature rows of the registers its operation could read, and the gold one's index."""
    ir = item.ir
    operations = [instruction.operation_span for instruction in ir.instructions]
    similarities = _Similarities(
        np.asarray(item.hidden_states), item.hidden_channels, item.hidden_channel_widths
    )
    stretches = register_stretches(ir.input_spans, operations, len(item.hidden_states))
    groups: list[tuple[list[list[float]], int]] = []
    for step, instruction in enumerate(ir.instructions):
        own = ir.n_inputs + step
        for register, mention in zip(instruction.args, instruction.argument_spans, strict=True):
            if _literal(mention, ir.input_spans):
                continue
            features = antecedent_features(similarities, mention, stretches, ir.n_inputs)
            candidates = [index for index in range(len(features)) if index != own]
            groups.append(([features[index] for index in candidates], candidates.index(register)))
    return groups


def _fit_conditional(groups: Sequence[tuple[list[list[float]], int]]) -> tuple[float, ...]:
    """Weights maximising each mention's log-probability of its own register among its candidates.

    The readout is used as a distribution over a mention's registers, so it is
    fitted as one (a conditional logit), with the same unit L2 penalty the
    pairwise fit carried. The bias cancels in the softmax and stays zero.
    """
    from scipy.optimize import minimize

    blocks = [np.asarray(rows, dtype=np.float64) for rows, _gold in groups]
    golds = [gold for _rows, gold in groups]

    def loss(weight: np.ndarray) -> tuple[float, np.ndarray]:
        total = 0.5 * float(weight @ weight)
        gradient = weight.copy()
        for rows, gold in zip(blocks, golds, strict=True):
            logits = rows @ weight
            top = float(np.max(logits))
            exp = np.exp(logits - top)
            norm = float(np.sum(exp))
            total -= float(logits[gold] - top - math.log(norm))
            gradient -= rows[gold] - (exp / norm) @ rows
        return total, gradient

    result = minimize(loss, np.zeros(len(FEATURES)), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return tuple(float(value) for value in result.x)


def antecedent_training_rows(item: Any) -> tuple[list[list[float]], list[int]]:
    """Each annotated argument mention against every register its operation could read."""
    ir = item.ir
    operations = [instruction.operation_span for instruction in ir.instructions]
    similarities = _Similarities(
        np.asarray(item.hidden_states), item.hidden_channels, item.hidden_channel_widths
    )
    stretches = register_stretches(ir.input_spans, operations, len(item.hidden_states))
    rows: list[list[float]] = []
    labels: list[int] = []
    for step, instruction in enumerate(ir.instructions):
        own = ir.n_inputs + step
        for register, mention in zip(instruction.args, instruction.argument_spans, strict=True):
            if _literal(mention, ir.input_spans):
                continue
            features = antecedent_features(similarities, mention, stretches, ir.n_inputs)
            for candidate, row in enumerate(features):
                if candidate == own:
                    continue
                rows.append(row)
                labels.append(int(candidate == register))
    return rows, labels


def fit_argument_antecedent(examples: Sequence[Any], *, objective: str = "pairwise") -> ArgumentAntecedent:
    """Fit on each training argument mention against every other register of its request.

    ``objective`` "pairwise" fits each (mention, register) pair as a yes or no;
    "conditional" fits each mention's distribution over its registers, which is
    how the readout is used.
    """
    from sklearn.linear_model import LogisticRegression

    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("argument antecedents are fitted on training rows only")
    if objective not in ("pairwise", "conditional"):
        raise ValueError("argument antecedents are fitted pairwise or conditionally")
    if objective == "conditional":
        groups = [group for item in examples for group in antecedent_training_groups(item)]
        if not groups:
            raise ValueError("argument antecedents need annotated mentions")
        receipt = {
            "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
            "training_rows": len(examples),
            "mentions": len(groups),
            "objective": "conditional",
            "splits_used": ["train"],
        }
        return ArgumentAntecedent(_fit_conditional(groups), 0.0, {**receipt, "receipt_sha256": _sha(receipt)})
    rows: list[list[float]] = []
    labels: list[int] = []
    for item in examples:
        item_rows, item_labels = antecedent_training_rows(item)
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
    }
    return ArgumentAntecedent(
        tuple(float(value) for value in fitted.coef_[0]),
        float(fitted.intercept_[0]),
        {**receipt, "receipt_sha256": _sha(receipt)},
    )


__all__ = [
    "ANTECEDENT_SCHEMA",
    "ArgumentAntecedent",
    "antecedent_features",
    "antecedent_training_rows",
    "argument_antecedent_from_dict",
    "fit_argument_antecedent",
    "register_stretches",
]

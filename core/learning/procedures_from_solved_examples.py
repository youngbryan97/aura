"""Procedures she writes from solved examples, kept only when every known answer agrees.

When a kind of problem recurs and some of its answers are known, the logic
that carries a problem to its answer can be written down once and run again.
Here her model writes that logic as a Python function, ``solve(problem) ->
answer``, from a few solved examples of the kind. The function is then run in
the OS sandbox (core/sandbox/untrusted_python.py) on every problem whose
answer is known, and it is kept only if every answer it returns agrees.

The known examples are split in two by a hash of their keys. Her model sees
examples, and later the problems its function got wrong, only from the first
part. The second part is sealed: it is run, never shown, so a function that
merely memorised what it was shown fails there. A function that fails is
revised from its own counterexamples, or replaced by a fresh one written from
different examples; the rounds are bounded by the caller.

A kept procedure answers a new problem only when the problem reads like its
own kind. Every kind she has examples of has a signature: the share of its
problems containing each word. A new problem goes to the kind whose
signature it is nearest by cosine, and only if it is at least as near as the
least typical of that kind's sealed problems. A problem nearest a kind with
no kept procedure is left to her ordinary answer, and so is one whose
procedure raises or returns nothing.

Nothing here knows a benchmark. The caller supplies the examples, a way to
ask her model for text, and optionally the comparison that says two answers
agree with a known example (the default compares the returned answer with the
shown one as normalised strings, and numbers by value).

core/learning/procedure_induction.py finds procedures with no model at all,
by enumerating compositions of a fixed primitive set over small typed values.
A problem stated in prose needs a reader of that prose first, and no small
primitive set reaches one, so here her model writes the whole procedure and
the known answers decide whether it is kept.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

SOLVED_EXAMPLE_PROCEDURES_SCHEMA = "aura.solved_example_procedures.v1"

Propose = Callable[[list[str]], list[str]]
Agree = Callable[[str, "SolvedExample"], bool]

_WORD = re.compile(r"[a-z]+")
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SolvedExample:
    key: str
    problem: str
    answer: str


@dataclass(frozen=True)
class Outcome:
    key: str
    agreed: bool
    returned: str | None
    error: str = ""


@dataclass
class Candidate:
    """One function her model wrote, and what running it on the known examples showed."""

    family: str
    line: int
    round: int
    code: str
    sha256: str
    shown_keys: list[str]
    pool_total: int = 0
    sealed_total: int = 0
    #: What was run, in order. A check stops once it has the counterexamples a
    #: repair can use, so these may be shorter than the totals.
    pool: list[Outcome] = field(default_factory=list)
    sealed: list[Outcome] = field(default_factory=list)

    @property
    def pool_agreed(self) -> int:
        return sum(outcome.agreed for outcome in self.pool)

    @property
    def sealed_agreed(self) -> int:
        return sum(outcome.agreed for outcome in self.sealed)

    @property
    def admitted(self) -> bool:
        """Every known answer, shown pool and sealed half, run and agreed."""
        return (0 < self.pool_total == self.pool_agreed == len(self.pool)
                and 0 < self.sealed_total == self.sealed_agreed == len(self.sealed))


@dataclass
class Signature:
    """A kind of problem as the share of its problems that contain each word."""

    weights: dict[str, float]
    nearest_required: float

    def similarity(self, problem: str) -> float:
        return _cosine(_words(problem), self.weights)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _cosine(words: set[str], weights: dict[str, float]) -> float:
    if not words or not weights:
        return 0.0
    dot = sum(weights.get(word, 0.0) for word in words)
    norm = math.sqrt(sum(value * value for value in weights.values()))
    return dot / (math.sqrt(len(words)) * norm) if norm else 0.0


def signature_of(pool: Sequence[SolvedExample], sealed: Sequence[SolvedExample]) -> Signature:
    """Word shares from the pool; the bar is the least typical sealed problem's similarity."""
    counts: Counter[str] = Counter()
    for example in pool:
        counts.update(_words(example.problem))
    weights = {word: count / len(pool) for word, count in counts.items()} if pool else {}
    nearest = min((_cosine(_words(e.problem), weights) for e in sealed), default=1.0)
    return Signature(weights=weights, nearest_required=nearest)


def split_examples(examples: Sequence[SolvedExample]) -> tuple[list[SolvedExample], list[SolvedExample]]:
    """The pool her model may see and the sealed half it never does, by key hash."""
    ordered = sorted(examples, key=lambda e: hashlib.sha256(e.key.encode()).hexdigest())
    return ordered[0::2], ordered[1::2]


def _number(text: str) -> Fraction | None:
    try:
        return Fraction(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def answers_agree(returned: str, expected: str) -> bool:
    """Equal after case, surrounding space and inner runs of space; numbers by value."""
    got = " ".join(str(returned).split()).casefold()
    want = " ".join(str(expected).split()).casefold()
    if got == want:
        return True
    got_number, want_number = _number(got), _number(want)
    return got_number is not None and got_number == want_number


def agrees_with_shown_answer(returned: str, example: SolvedExample) -> bool:
    return answers_agree(returned, example.answer)


def extract_program(text: str) -> str | None:
    """The last fenced code block that defines ``solve``."""
    for block in reversed(_CODE_BLOCK.findall(text or "")):
        if re.search(r"^def solve\s*\(", block, re.MULTILINE):
            return block.strip() + "\n"
    return None


def _example_text(index: int, example: SolvedExample) -> str:
    return f"Problem {index}:\n{example.problem.strip()}\n\nAnswer {index}: {example.answer.strip()}"


def _available(mechanisms: str | None) -> str:
    if not mechanisms:
        return "Only the Python standard library is available."
    return ("The Python standard library is available, and `import mechanisms` gives these exact "
            "mechanisms:\n\n" + mechanisms.strip() + "\n")


def proposal_request(shown: Sequence[SolvedExample], mechanisms: str | None = None) -> str:
    examples = "\n\n".join(_example_text(i + 1, e) for i, e in enumerate(shown))
    return (
        "Each problem below is followed by its correct answer. Write a Python function "
        "`solve(problem: str) -> str` that computes the correct answer for any problem of this kind, "
        "including new ones, returned in the same form as these answers. " + _available(mechanisms)
        + " Reply with one Python code block.\n\n" + examples
    )


def repair_request(code: str, shown: Sequence[SolvedExample], failures: Sequence[tuple[SolvedExample, Outcome]],
                   mechanisms: str | None = None) -> str:
    examples = "\n\n".join(_example_text(i + 1, e) for i, e in enumerate(shown))
    wrong = "\n\n".join(
        f"Problem:\n{example.problem.strip()}\n\nCorrect answer: {example.answer.strip()}\n"
        + (f"solve raised: {outcome.error}" if outcome.error else f"solve returned: {outcome.returned}")
        for example, outcome in failures
    )
    return (
        "Each problem below is followed by its correct answer. This function was written to compute "
        "the answer for any problem of this kind:\n\n```python\n" + code.strip() + "\n```\n\n"
        "It is wrong on the problems that follow the examples. Write a corrected "
        "`solve(problem: str) -> str` for every problem of this kind. " + _available(mechanisms)
        + " Reply with one Python code block.\n\n" + examples
        + "\n\nWhere it was wrong:\n\n" + wrong
    )


def _mechanisms_directory() -> str:
    from core.reasoning import mechanisms

    return str(Path(mechanisms.__file__).resolve().parent)


_FRAME = re.compile(r'File "[^"]*candidate\.py", line (\d+), in (\S+)\n\s*(.*)')


def _where_it_failed(stderr: str, prelude_lines: int) -> str:
    """The innermost frame of the procedure's own code in a traceback: its line, function and text."""
    frames = _FRAME.findall(stderr)
    if not frames:
        return ""
    line, function, text = frames[-1]
    return f"line {int(line) - prelude_lines}, in {function}: {text.strip()}"


def run_procedure(code: str, problem: str, *, timeout_s: float | None = None) -> tuple[str | None, str]:
    """One call of ``solve`` in the OS sandbox: what it returned, or why it did not.

    The child may read core/reasoning, so ``import mechanisms`` works there
    (core/reasoning/mechanisms.py imports only the standard library and its
    sibling constraint engine).
    """
    from core.sandbox.untrusted_python import DEFAULT_TIMEOUT_S, call_untrusted_function

    directory = _mechanisms_directory()
    prelude = f"import sys as _aura_sys\n_aura_sys.path.insert(0, {directory!r})\n"
    outcome = call_untrusted_function(prelude + code, "solve", calls=[[problem]],
                                      timeout_s=timeout_s or DEFAULT_TIMEOUT_S, extra_read_paths=(directory,),
                                      source="procedures_from_solved_examples")
    if outcome.status != "ok":
        where = _where_it_failed(outcome.stderr or "", prelude.count("\n"))
        return None, f"{outcome.status}: {str(outcome.error or '')[:300]}" + (f" ({where})" if where else "")
    results = list(getattr(outcome, "results", None) or [])
    if not results:
        return None, "no result"
    value = results[0]
    if value is None or not str(value).strip():
        return None, "returned nothing"
    return str(value), ""


def check_program(code: str, examples: Sequence[SolvedExample], agree: Agree = agrees_with_shown_answer,
                  *, workers: int = 1, stop_after_failures: int | None = None) -> list[Outcome]:
    """Run ``solve`` on each example in order, one sandboxed child each, so one hang costs one example.

    With ``stop_after_failures``, no further examples start once that many
    have disagreed: a function that is wrong somewhere is not kept, and a
    repair needs only a few of the problems it got wrong, so a slow wrong
    function does not cost a run over every known problem.
    """

    def one(example: SolvedExample) -> Outcome:
        returned, error = run_procedure(code, example.problem)
        agreed = returned is not None and bool(agree(returned, example))
        return Outcome(key=example.key, agreed=agreed, returned=returned, error=error)

    width = max(1, workers)
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=width) as pool:
        for start in range(0, len(examples), width):
            outcomes.extend(pool.map(one, examples[start:start + width]))
            failed = sum(not outcome.agreed for outcome in outcomes)
            if stop_after_failures is not None and failed >= stop_after_failures:
                break
    return outcomes


@dataclass
class Family:
    name: str
    pool: list[SolvedExample]
    sealed: list[SolvedExample]
    signature: Signature
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def admitted(self) -> list[Candidate]:
        return [c for c in self.candidates if c.admitted]


def _shown_for(pool: Sequence[SolvedExample], line: int, attempt: int, shots: int) -> list[SolvedExample]:
    """A different draw of the pool for each line and attempt, so fresh proposals differ."""
    salt = f"{line}:{attempt}"
    ordered = sorted(pool, key=lambda e: hashlib.sha256(f"{salt}:{e.key}".encode()).hexdigest())
    return ordered[:shots]


def induce(
    families: dict[str, Sequence[SolvedExample]],
    propose: Propose,
    *,
    shots: int,
    lines: int,
    rounds: int,
    agree: Agree = agrees_with_shown_answer,
    workers: int = 1,
    mechanisms: str | None = None,
    on_candidate: Callable[[Candidate], None] | None = None,
) -> dict[str, Family]:
    """Write, check and revise procedures for every family, all families' requests batched per round.

    Each family runs ``lines`` independent lines. A line starts from its own
    draw of ``shots`` shown examples; after a failure on the pool its next
    request shows the function and up to ``shots`` of the pool problems it got
    wrong; after a failure confined to the sealed half (or no code at all) it
    starts again from a fresh draw. A line stops once a function is admitted.
    ``mechanisms`` is the reference text of core/reasoning/mechanisms.py when
    her model is told it may import them.
    """
    state: dict[str, Family] = {}
    for name, examples in families.items():
        pool, sealed = split_examples(examples)
        state[name] = Family(name=name, pool=pool, sealed=sealed, signature=signature_of(pool, sealed))
    # (family, line) -> (code to repair or None, attempt counter)
    lineage: dict[tuple[str, int], tuple[Candidate | None, int]] = {
        (name, line): (None, 0) for name in state for line in range(lines)}
    finished: set[tuple[str, int]] = set()
    for round_index in range(rounds):
        active = [k for k in lineage if k not in finished]
        if not active:
            break
        requests, shown_sets = [], []
        for name, line in active:
            family = state[name]
            previous, attempt = lineage[(name, line)]
            shown = _shown_for(family.pool, line, attempt, shots)
            by_key = {e.key: e for e in family.pool}
            failures = [(by_key[o.key], o) for o in previous.pool if not o.agreed][:shots] \
                if previous is not None else []
            if previous is not None and failures:
                requests.append(repair_request(previous.code, shown, failures, mechanisms))
            else:
                requests.append(proposal_request(shown, mechanisms))
            shown_sets.append(shown)
        texts = propose(requests)
        for (name, line), shown, text in zip(active, shown_sets, texts, strict=True):
            family = state[name]
            _previous, attempt = lineage[(name, line)]
            code = extract_program(text)
            if code is None:
                lineage[(name, line)] = (None, attempt + 1)
                continue
            candidate = Candidate(family=name, line=line, round=round_index, code=code,
                                  sha256=hashlib.sha256(code.encode()).hexdigest(),
                                  shown_keys=[e.key for e in shown],
                                  pool_total=len(family.pool), sealed_total=len(family.sealed))
            candidate.pool = check_program(code, family.pool, agree, workers=workers, stop_after_failures=shots)
            if candidate.pool_agreed == len(family.pool):
                candidate.sealed = check_program(code, family.sealed, agree, workers=workers,
                                                 stop_after_failures=1)
            family.candidates.append(candidate)
            if on_candidate is not None:
                on_candidate(candidate)
            if candidate.admitted:
                finished.add((name, line))
            elif candidate.sealed:
                # Every shown-pool answer agreed and a sealed one did not: the
                # sealed problems are never shown, so start again from new examples.
                lineage[(name, line)] = (None, attempt + 1)
            else:
                lineage[(name, line)] = (candidate, attempt)
    return state


@dataclass
class ProcedureBook:
    """Kept procedures, how many known answers each kind's agreed with, and every kind's signature."""

    signatures: dict[str, Signature]
    procedures: dict[str, list[str]]
    agreed: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_families(cls, families: dict[str, Family]) -> ProcedureBook:
        kept = {name: family for name, family in families.items() if family.admitted}
        return cls(signatures={name: family.signature for name, family in families.items()},
                   procedures={name: [c.code for c in family.admitted] for name, family in kept.items()},
                   agreed={name: len(family.pool) + len(family.sealed) for name, family in kept.items()})

    @classmethod
    def merged(cls, books: Sequence[ProcedureBook]) -> ProcedureBook:
        """One book from several, each kind from the one book that has it; a kind in two is refused."""
        out = cls(signatures={}, procedures={}, agreed={})
        for book in books:
            twice = set(out.signatures) & set(book.signatures)
            if twice:
                raise ValueError(f"kinds in more than one book: {sorted(twice)}")
            out.signatures.update(book.signatures)
            out.procedures.update(book.procedures)
            out.agreed.update(book.agreed)
        return out

    def route(self, problem: str) -> tuple[str | None, float]:
        """The nearest kind, if the problem is at least as near as that kind's least typical sealed problem."""
        if not self.signatures:
            return None, 0.0
        scored = sorted(((signature.similarity(problem), name) for name, signature in self.signatures.items()),
                        reverse=True)
        similarity, name = scored[0]
        return (name if similarity >= self.signatures[name].nearest_required else None), similarity

    def answer(self, problem: str) -> dict[str, Any]:
        """The kept procedures' answer when they apply and agree; otherwise a reason there is none."""
        family, similarity = self.route(problem)
        receipt: dict[str, Any] = {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": family,
                                   "similarity": round(similarity, 4), "answer": None}
        if family is not None:
            # Laplace's rule of succession over the known answers its procedures all matched.
            known = int(self.agreed.get(family, 0))
            receipt["agreed_known"] = known
            receipt["confidence"] = round((known + 1) / (known + 2), 4)
        if family is None:
            receipt["declined"] = "no_kind_near_enough"
            return receipt
        codes = self.procedures.get(family) or []
        if not codes:
            receipt["declined"] = "kind_has_no_kept_procedure"
            return receipt
        returned = [run_procedure(code, problem) for code in codes]
        receipt["procedures"] = [{"sha256": hashlib.sha256(c.encode()).hexdigest()[:16],
                                  "returned": r, "error": e} for c, (r, e) in zip(codes, returned, strict=True)]
        values = [r for r, _e in returned if r is not None]
        if len(values) != len(codes):
            receipt["declined"] = "a_kept_procedure_did_not_answer"
        elif any(not answers_agree(value, values[0]) for value in values[1:]):
            receipt["declined"] = "kept_procedures_disagree"
        else:
            receipt["answer"] = values[0]
        return receipt

    def to_json(self) -> dict[str, Any]:
        return {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA,
                "signatures": {n: asdict(s) for n, s in self.signatures.items()},
                "procedures": self.procedures, "agreed": self.agreed}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ProcedureBook:
        if value.get("schema") != SOLVED_EXAMPLE_PROCEDURES_SCHEMA:
            raise ValueError("not a procedure book")
        return cls(signatures={n: Signature(**s) for n, s in value["signatures"].items()},
                   procedures={n: list(c) for n, c in value["procedures"].items()},
                   agreed={n: int(k) for n, k in (value.get("agreed") or {}).items()})


#: Her own book, under her data directory. Nothing writes it yet but an
#: induction run she is given; the live route reads it when it exists.
KEPT_BOOK_RELATIVE = "procedures/kept_from_solved_examples.json"
_kept_cache: dict[str, Any] = {}


def kept_procedure_book() -> ProcedureBook | None:
    """Her kept book, reread when the file changes; None when she has none."""
    from core.utils.paths import aura_data_dir

    path = aura_data_dir() / KEPT_BOOK_RELATIVE
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    if _kept_cache.get("stamp") != (str(path), stamp):
        _kept_cache["book"] = ProcedureBook.from_json(json.loads(path.read_text(encoding="utf-8")))
        _kept_cache["stamp"] = (str(path), stamp)
    return _kept_cache["book"]


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    return {"family": candidate.family, "line": candidate.line, "round": candidate.round,
            "sha256": candidate.sha256, "shown_keys": candidate.shown_keys,
            # agreed, run, known
            "pool": [candidate.pool_agreed, len(candidate.pool), candidate.pool_total],
            "sealed": [candidate.sealed_agreed, len(candidate.sealed), candidate.sealed_total],
            "admitted": candidate.admitted, "code": candidate.code,
            "failures": [asdict(o) for o in candidate.pool + candidate.sealed if not o.agreed][:12]}


def dumps(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True)

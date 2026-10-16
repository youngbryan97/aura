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


def distinct_signatures(families: dict[str, tuple[Sequence[SolvedExample], Sequence[SolvedExample]]]
                        ) -> dict[str, Signature]:
    """Each kind's signature with every word weighted by how few kinds use it, and its bar taken again.

    A word share alone lets a template that many kinds share outweigh the
    words that tell them apart: every DeepMind module's request ends "Please
    reason step by step, and put your final answer within boxed", and one
    module's problem cleared four modules' bars within a hundredth (15
    October). Each word is weighted by the logarithm of how many kinds there
    are over how many use it (inverse document frequency across kinds), so a
    word every kind uses counts for nothing, and each bar is again the least
    typical sealed problem's similarity, as signature_of takes it.
    """
    shares: dict[str, dict[str, float]] = {}
    for name, (pool, _sealed) in families.items():
        counts: Counter[str] = Counter()
        for example in pool:
            counts.update(_words(example.problem))
        shares[name] = {word: count / len(pool) for word, count in counts.items()} if pool else {}
    kinds = len(shares)
    used_by = Counter(word for share in shares.values() for word in share)
    out: dict[str, Signature] = {}
    for name, share in shares.items():
        weights = {word: value * math.log(kinds / used_by[word]) for word, value in share.items()
                   if used_by[word] < kinds}
        nearest = min((_cosine(_words(e.problem), weights) for e in families[name][1]), default=1.0)
        out[name] = Signature(weights=weights, nearest_required=nearest)
    return out


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


def _available(mechanisms: str | None, steps: str | None = None) -> str:
    if not mechanisms:
        text = "Only the Python standard library is available."
    else:
        text = ("The Python standard library is available, and `import mechanisms` gives these exact "
                "mechanisms:\n\n" + mechanisms.strip() + "\n")
    if steps:
        # core/learning/kept_steps.py: kinds already kept, callable as steps of a larger procedure.
        text += ("\nThe function `kept_answer` is already defined, and a step that is a problem of one of "
                 "these kinds can be answered with it:\n\n" + steps.strip() + "\n")
    return text


def proposal_request(shown: Sequence[SolvedExample], mechanisms: str | None = None,
                     steps: str | None = None) -> str:
    examples = "\n\n".join(_example_text(i + 1, e) for i, e in enumerate(shown))
    return (
        "Each problem below is followed by its correct answer. Write a Python function "
        "`solve(problem: str) -> str` that computes the correct answer for any problem of this kind, "
        "including new ones, returned in the same form as these answers. " + _available(mechanisms, steps)
        + " Reply with one Python code block.\n\n" + examples
    )


def repair_request(code: str, shown: Sequence[SolvedExample], failures: Sequence[tuple[SolvedExample, Outcome]],
                   mechanisms: str | None = None, steps: str | None = None) -> str:
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
        "`solve(problem: str) -> str` for every problem of this kind. " + _available(mechanisms, steps)
        + " Reply with one Python code block.\n\n" + examples
        + "\n\nWhere it was wrong:\n\n" + wrong
    )


def _mechanisms_directory() -> str:
    from core.reasoning import mechanisms

    return str(Path(mechanisms.__file__).resolve().parent)


def _mechanisms_prelude() -> str:
    return f"import sys as _aura_sys\n_aura_sys.path.insert(0, {_mechanisms_directory()!r})\n"


def singled_out(book: ProcedureBook, text: str) -> list[str]:
    """The kept kinds whose problems ``text`` shares words with by more than the rest do.

    A kind is named when its score stands STANDS_OUT deviations above the mean
    of every kept kind's score, the rule her capabilities are chosen by
    (core/intent/capability_by_meaning.py). Among n scores none can stand
    further out than sqrt(n - 1) deviations, so while a book keeps too few
    kinds for the rule to name one (fewer than nine), every kind is named;
    there are few.
    """
    import numpy as np

    from core.intent.capability_by_meaning import STANDS_OUT

    kept = sorted(k for k, codes in book.procedures.items() if codes and k in book.signatures)
    if math.sqrt(max(0, len(kept) - 1)) < STANDS_OUT:
        return kept
    scores = np.array([book.signatures[k].similarity(text) for k in kept])
    spread = float(scores.std())
    if spread == 0.0:
        return []
    return [k for k, score in zip(kept, scores, strict=True) if (score - scores.mean()) / spread >= STANDS_OUT]


@dataclass(frozen=True)
class KeptSteps:
    """What a sandboxed program needs to call a book's kept kinds as steps (core/learning/kept_steps.py).

    ``prelude`` is the lines put before the program and ``read_paths`` what the
    child may read: the mechanisms, the helper, and the book itself.
    ``listing_for`` is what her model reads for one request: only the kinds
    its words single out, so a book of a hundred kinds does not become a
    hundred examples in every request.
    """

    prelude: str
    read_paths: tuple[str, ...]
    book: ProcedureBook = field(compare=False, repr=False)

    def listing_for(self, text: str) -> str:
        """The kept kinds ``text`` singles out, each with one of its problems and answer; empty when none is."""
        from core.learning import kept_steps

        kinds = singled_out(self.book, text)
        if not kinds:
            return ""
        return kept_steps.reference({"procedures": {k: self.book.procedures[k] for k in kinds},
                                     "examples": self.book.examples})


def kept_steps_from(book_path: str | Path, book: ProcedureBook | None = None) -> KeptSteps | None:
    """The kept kinds of the book at ``book_path`` as steps; None when it keeps no procedure."""
    from core.learning import kept_steps

    path = Path(book_path).resolve()
    book = book if book is not None else ProcedureBook.from_json(json.loads(path.read_text(encoding="utf-8")))
    if not book.procedures:
        return None
    helper = str(Path(kept_steps.__file__).resolve().parent)
    return KeptSteps(prelude=_mechanisms_prelude() + kept_steps.prelude(helper, str(path)),
                     read_paths=(_mechanisms_directory(), helper, str(path)), book=book)


_FRAME = re.compile(r'File "[^"]*candidate\.py", line (\d+), in (\S+)\n\s*(.*)')


def _where_it_failed(stderr: str, prelude_lines: int) -> str:
    """The innermost frame of the procedure's own code in a traceback: its line, function and text."""
    frames = _FRAME.findall(stderr)
    if not frames:
        return ""
    line, function, text = frames[-1]
    return f"line {int(line) - prelude_lines}, in {function}: {text.strip()}"


def run_procedure(code: str, problem: str, *, timeout_s: float | None = None,
                  steps: KeptSteps | None = None) -> tuple[str | None, str]:
    """One call of ``solve`` in the OS sandbox: what it returned, or why it did not.

    The child may read core/reasoning, so ``import mechanisms`` works there
    (core/reasoning/mechanisms.py imports only the standard library and its
    sibling constraint engine). With ``steps``, ``kept_answer`` is defined
    too, and the child may read the book it answers from.
    """
    from core.sandbox.untrusted_python import DEFAULT_TIMEOUT_S, call_untrusted_function

    prelude = steps.prelude if steps is not None else _mechanisms_prelude()
    read_paths = steps.read_paths if steps is not None else (_mechanisms_directory(),)
    outcome = call_untrusted_function(prelude + code, "solve", calls=[[problem]],
                                      timeout_s=timeout_s or DEFAULT_TIMEOUT_S, extra_read_paths=read_paths,
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
                  *, workers: int = 1, stop_after_failures: int | None = None,
                  steps: KeptSteps | None = None) -> list[Outcome]:
    """Run ``solve`` on each example in order, one sandboxed child each, so one hang costs one example.

    With ``stop_after_failures``, no further examples start once that many
    have disagreed: a function that is wrong somewhere is not kept, and a
    repair needs only a few of the problems it got wrong, so a slow wrong
    function does not cost a run over every known problem.
    """

    def one(example: SolvedExample) -> Outcome:
        returned, error = run_procedure(code, example.problem, steps=steps)
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


def request_for(family: Family, line: int, previous: Candidate | None, attempt: int, *, shots: int,
                mechanisms: str | None = None, steps: KeptSteps | None = None
                ) -> tuple[str, list[SolvedExample]]:
    """The next request on one line of a family, and the examples it shows.

    After a failure on the pool the request shows the function and up to
    ``shots`` pool problems it got wrong; otherwise it is a fresh proposal
    from this line's draw for ``attempt``.
    """
    shown = _shown_for(family.pool, line, attempt, shots)
    listing = steps.listing_for(" ".join(e.problem for e in shown)) if steps is not None else None
    by_key = {e.key: e for e in family.pool}
    failures = [(by_key[o.key], o) for o in previous.pool if not o.agreed][:shots] if previous is not None else []
    if previous is not None and failures:
        return repair_request(previous.code, shown, failures, mechanisms, listing), shown
    return proposal_request(shown, mechanisms, listing), shown


def judge(family: Family, line: int, round_index: int, shown: Sequence[SolvedExample], text: str, *, shots: int,
          agree: Agree = agrees_with_shown_answer, workers: int = 1, steps: KeptSteps | None = None
          ) -> Candidate | None:
    """Run the function in ``text`` on the family's known examples; None when the reply has no function.

    The shown pool runs first and stops after ``shots`` disagreements (the
    counterexamples a repair can use); the sealed half runs only when every
    pool answer agreed, and stops at its first disagreement.
    """
    code = extract_program(text)
    if code is None:
        return None
    candidate = Candidate(family=family.name, line=line, round=round_index, code=code,
                          sha256=hashlib.sha256(code.encode()).hexdigest(), shown_keys=[e.key for e in shown],
                          pool_total=len(family.pool), sealed_total=len(family.sealed))
    candidate.pool = check_program(code, family.pool, agree, workers=workers, stop_after_failures=shots,
                                   steps=steps)
    if candidate.pool_agreed == len(family.pool):
        candidate.sealed = check_program(code, family.sealed, agree, workers=workers, stop_after_failures=1,
                                         steps=steps)
    return candidate


def next_on_line(candidate: Candidate | None, attempt: int) -> tuple[Candidate | None, int]:
    """What a line carries into its next request: the function to repair, or none and a fresh draw.

    A failure on the shown pool is repaired from its counterexamples. A
    function that agreed on every pool answer and failed on the sealed half
    is not repaired, because the sealed problems are never shown: the line
    starts again from new examples, as it does when a reply has no function.
    """
    if candidate is None or candidate.sealed:
        return None, attempt + 1
    return candidate, attempt


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
    steps: KeptSteps | None = None,
    on_candidate: Callable[[Candidate], None] | None = None,
) -> dict[str, Family]:
    """Write, check and revise procedures for every family, all families' requests batched per round.

    Each family runs ``lines`` independent lines (``request_for``,
    ``judge``, ``next_on_line``). A line stops once a function is admitted.
    ``mechanisms`` is the reference text of core/reasoning/mechanisms.py when
    her model is told it may import them, and ``steps`` the kinds already
    kept, which a procedure may call for a step and is checked calling.
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
            previous, attempt = lineage[(name, line)]
            request, shown = request_for(state[name], line, previous, attempt, shots=shots,
                                         mechanisms=mechanisms, steps=steps)
            requests.append(request)
            shown_sets.append(shown)
        texts = propose(requests)
        for (name, line), shown, text in zip(active, shown_sets, texts, strict=True):
            family = state[name]
            _previous, attempt = lineage[(name, line)]
            candidate = judge(family, line, round_index, shown, text, shots=shots, agree=agree, workers=workers,
                              steps=steps)
            if candidate is not None:
                family.candidates.append(candidate)
                if on_candidate is not None:
                    on_candidate(candidate)
                if candidate.admitted:
                    finished.add((name, line))
                    continue
            lineage[(name, line)] = next_on_line(candidate, attempt)
    return state


@dataclass
class ProcedureBook:
    """Kept procedures, how many known answers each kind's agreed with, and every kind's signature.

    ``meanings`` holds a kind's direction in her embedding space
    (core/learning/kinds_by_meaning.py) and ``examples`` a few of its solved
    problems as shown, which is what a request worded otherwise is restated by.
    ``posing`` is what every problem of a kind opens with before its question,
    as a benchmark poses it (BIG-Bench Hard's three worked examples): left out
    of the kind's meaning and examples, and put back before its procedures
    read a restated request, so they read it as they were kept on.
    ``steps`` is set on a book read from a file, so a procedure that calls
    another kept kind (core/learning/kept_steps.py) can run; it is not saved.
    """

    signatures: dict[str, Signature]
    procedures: dict[str, list[str]]
    agreed: dict[str, int] = field(default_factory=dict)
    meanings: dict[str, Any] = field(default_factory=dict)
    examples: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    posing: dict[str, str] = field(default_factory=dict)
    steps: KeptSteps | None = field(default=None, compare=False, repr=False)

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
            out.meanings.update(book.meanings)
            out.examples.update(book.examples)
            out.posing.update(book.posing)
        return out

    @classmethod
    def combined(cls, books: Sequence[ProcedureBook]) -> ProcedureBook:
        """One book from several that may share kinds: each kind keeps the procedures of every book that keeps it.

        A kind's procedures were each checked against all of its known answers,
        so two books' procedures for one kind are both kept and must agree when
        they answer. Its count of agreed answers is the largest any book has;
        its signature, meaning and examples come from the first book that keeps
        a procedure for it, or else from the first that has it at all.
        """
        out = cls(signatures={}, procedures={}, agreed={})
        kinds = list(dict.fromkeys(kind for book in books for kind in book.signatures))
        for kind in kinds:
            having = [b for b in books if kind in b.signatures]
            keeping = [b for b in having if b.procedures.get(kind)]
            source = (keeping or having)[0]
            out.signatures[kind] = source.signatures[kind]
            for part in ("meanings", "examples", "posing"):
                value = next((getattr(b, part)[kind] for b in [source, *having] if kind in getattr(b, part)), None)
                if value is not None:
                    getattr(out, part)[kind] = value
            if keeping:
                out.procedures[kind] = list(dict.fromkeys(code for b in keeping for code in b.procedures[kind]))
                out.agreed[kind] = max(int(b.agreed.get(kind, 0)) for b in keeping)
        return out

    def route(self, problem: str) -> tuple[str | None, float]:
        """The nearest kind, if the problem is at least as near as that kind's least typical sealed problem."""
        if not self.signatures:
            return None, 0.0
        scored = sorted(((signature.similarity(problem), name) for name, signature in self.signatures.items()),
                        reverse=True)
        similarity, name = scored[0]
        return (name if similarity >= self.signatures[name].nearest_required else None), similarity

    def route_by_meaning(self, query: Any) -> tuple[str | None, float, str]:
        """The kind a request's query vector means (core/learning/kinds_by_meaning.py); None with a reason."""
        from core.learning.kinds_by_meaning import nearest_kind

        return nearest_kind(query, self.meanings)

    def near_enough(self, problem: str) -> list[tuple[float, str]]:
        """Every kind the problem is at least as near as that kind's least typical sealed problem, nearest first."""
        scored = ((signature.similarity(problem), name) for name, signature in self.signatures.items())
        return sorted(((similarity, name) for similarity, name in scored
                       if similarity >= self.signatures[name].nearest_required), reverse=True)

    def answer(self, problem: str) -> dict[str, Any]:
        """The kept procedures' answer when they apply and agree; otherwise a reason there is none.

        A problem near enough to one kind is that kind's. Near enough to
        several, its words do not say which kind it is: kinds whose problems
        share a template (every DeepMind module's "Calculate ... Please
        reason step by step") clear each other's bars by thousandths. Then
        each of those kinds' procedures answers, and the answer stands only if
        they all agree; when one of the kinds keeps no procedure, or they
        disagree, none is given and the draft stands.
        """
        near = self.near_enough(problem)
        if not near:
            family, similarity = self.route(problem)
            return {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": None, "similarity": round(similarity, 4),
                    "answer": None, "form": "words", "declined": "no_kind_near_enough"}
        similarity, family = near[0]
        receipt: dict[str, Any] = {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": family,
                                   "similarity": round(similarity, 4), "answer": None, "form": "words"}
        if len(near) == 1:
            return self._run_kind(family, problem, receipt)
        kinds = [name for _similarity, name in near]
        receipt["near_enough"] = kinds
        if any(not self.procedures.get(name) for name in kinds):
            receipt["declined"] = "a_kind_near_enough_keeps_no_procedure"
            return receipt
        answers = [self._run_kind(name, problem, {"family": name}) for name in kinds]
        receipt["agreed_known"] = min(int(a.get("agreed_known") or 0) for a in answers)
        receipt["confidence"] = min(float(a.get("confidence") or 0.0) for a in answers)
        receipt["procedures"] = [p for a in answers for p in a.get("procedures", [])]
        values = [a.get("answer") for a in answers]
        if any(v is None for v in values):
            receipt["declined"] = "a_kind_near_enough_did_not_answer"
        elif any(not answers_agree(v, values[0]) for v in values[1:]):
            receipt["declined"] = "kinds_near_enough_disagree"
        else:
            receipt["answer"] = values[0]
        return receipt

    def answer_restated(self, family: str, restated: str) -> dict[str, Any]:
        """The answer of ``family``'s procedures to a request restated in that kind's own form.

        Routing chose the kind by meaning, so no word-share bar applies; the
        procedures read the restated text, and the receipt says it was restated.
        """
        receipt: dict[str, Any] = {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": family,
                                   "answer": None, "form": "restated", "restated": restated[:2000]}
        if family not in self.signatures:
            receipt["declined"] = "no_such_kind"
            return receipt
        return self._run_kind(family, self.posing.get(family, "") + restated, receipt)

    def _run_kind(self, family: str, problem: str, receipt: dict[str, Any]) -> dict[str, Any]:
        # Laplace's rule of succession over the known answers its procedures all matched.
        known = int(self.agreed.get(family, 0))
        receipt["agreed_known"] = known
        receipt["confidence"] = round((known + 1) / (known + 2), 4)
        codes = self.procedures.get(family) or []
        if not codes:
            receipt["declined"] = "kind_has_no_kept_procedure"
            return receipt
        returned = [run_procedure(code, problem, steps=self.steps) for code in codes]
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
                "procedures": self.procedures, "agreed": self.agreed,
                "meanings": {n: asdict(m) for n, m in self.meanings.items()}, "examples": self.examples,
                "posing": self.posing}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ProcedureBook:
        if value.get("schema") != SOLVED_EXAMPLE_PROCEDURES_SCHEMA:
            raise ValueError("not a procedure book")
        from core.learning.kinds_by_meaning import KindMeaning

        return cls(signatures={n: Signature(**s) for n, s in value["signatures"].items()},
                   procedures={n: list(c) for n, c in value["procedures"].items()},
                   agreed={n: int(k) for n, k in (value.get("agreed") or {}).items()},
                   meanings={n: KindMeaning(**m) for n, m in (value.get("meanings") or {}).items()},
                   examples={n: [dict(e) for e in v] for n, v in (value.get("examples") or {}).items()},
                   posing={n: str(p) for n, p in (value.get("posing") or {}).items()})


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
        book = ProcedureBook.from_json(json.loads(path.read_text(encoding="utf-8")))
        book.steps = kept_steps_from(path, book)
        _kept_cache["book"] = book
        _kept_cache["stamp"] = (str(path), stamp)
    return _kept_cache["book"]


def her_kept_steps() -> KeptSteps | None:
    """Her kept kinds as steps a sandboxed program may call; None when her book keeps no procedure."""
    book = kept_procedure_book()
    return book.steps if book is not None else None


def keep_for_her(book: ProcedureBook) -> Path:
    """Write ``book`` where her live route reads it, through the governed file-write gateway."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.utils.paths import aura_data_dir

    path = aura_data_dir() / KEPT_BOOK_RELATIVE
    with local_internal_governed_scope("procedures_from_solved_examples.keep_for_her", domain="file_write"):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="procedures_from_solved_examples.keep_for_her")
        gateway.write_text(path, dumps(book.to_json()), source="procedures_from_solved_examples.keep_for_her")
    _kept_cache.clear()
    return path


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

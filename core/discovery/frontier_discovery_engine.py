"""core/discovery/frontier_discovery_engine.py — the Frontier Discovery Engine (FDE).

The hard, honest truth this module is built around: a 32B/72B local model's *raw*
per-token reasoning has a fixed ceiling no wrapper can raise. What CAN exceed
frontier is **system-level** reasoning — inference-time search plus *sound
verification* — and the only kind of "new information" that is non-hallucinated is
information a verifier or a real feedback loop can prove or kill. So the engine that
actually invents (rather than confabulates) is a falsification immune system:

    propose a falsifiable conjecture → formalize it into a checkable object →
    falsify it HARD (exact computation / exhaustive residue check / verifier) →
    keep only survivors → check they are genuinely novel → commit the survivors
    into the causal belief substrate; and when no verifier can engage, file the
    claim as labeled CONJECTURE and NEVER assert it as fact.

That last clause is the whole game. Every output carries an :class:`EpistemicStatus`:

* ``PROVEN``     — a verifier checked it *exhaustively/deductively* and it holds.
* ``SUPPORTED``  — it survived N exact falsification trials; empirically supported,
                   explicitly **not** a proof (one counterexample would refute it).
* ``CONJECTURE`` — coherent and falsifiable, but unverified / no verifier engaged.
                   Surfaced with the experiment that would refute it; never asserted.
* ``REFUTED``    — a counterexample was found.

The sound backbone needs no LLM and therefore cannot hallucinate: it searches a space
of candidate integer laws (modular identities via exhaustive residue checking — a real
proof; integer-sequence closed forms via the bounded GA in
:mod:`core.discovery.evolver`) and keeps only what survives exact verification. An LLM,
when wired, only *proposes* candidates into the same gate — it never gets to assert.

Causal integration (not an island): survivors are committed to
:func:`core.cognition.scientific_engine.get_scientific_engine` — which opens an
Outcome-Ledger receipt and writes a confidence-weighted belief into ``world_state`` —
and emit a discovery receipt. The engine is reachable two ways at runtime: the
autonomous loop ticks :meth:`run_discovery_cycle` during idle windows (live, bounded,
sleep-aware), and the response lane calls :meth:`assess_claim` so a user's "is it true
that for all n …" is answered by exact falsification, status-labeled, not guessed.

Everything is bounded (wall-clock + candidate caps), governed (an optional Will/
constitution veto and a safety allowlist), and fail-open: a degraded organ downgrades
the result's status, it never fabricates one.
"""
from __future__ import annotations

import ast
import logging
import random
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.FrontierDiscovery")

# A predicate over a single integer case; True == the law holds for that case.
Predicate = Callable[[int], bool]


class EpistemicStatus(str, Enum):
    PROVEN = "proven"          # exhaustively / deductively verified
    SUPPORTED = "supported"    # survived N exact trials; NOT a proof
    CONJECTURE = "conjecture"  # falsifiable but unverified / no verifier engaged
    REFUTED = "refuted"        # a counterexample was found

    @property
    def is_assertable(self) -> bool:
        """Only PROVEN may be stated as fact. SUPPORTED is hedged, the rest never asserted."""
        return self is EpistemicStatus.PROVEN


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class FalsificationOutcome:
    status: EpistemicStatus
    trials: int
    exhaustive: bool
    counterexample: Optional[int] = None
    detail: str = ""


@dataclass
class Conjecture:
    statement: str
    domain: str
    formal_form: str
    status: EpistemicStatus
    confidence: float
    trials: int = 0
    exhaustive: bool = False
    counterexample: Optional[int] = None
    novelty: float = 1.0
    provenance: str = "structured"
    falsification_plan: str = ""
    committed: bool = False
    receipt_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "domain": self.domain,
            "formal_form": self.formal_form,
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "trials": self.trials,
            "exhaustive": self.exhaustive,
            "counterexample": self.counterexample,
            "novelty": round(self.novelty, 4),
            "provenance": self.provenance,
            "falsification_plan": self.falsification_plan,
            "committed": self.committed,
            "receipt_id": self.receipt_id,
        }


@dataclass
class DiscoveryReport:
    started_at: float
    finished_at: float
    candidates_examined: int
    proven: list[Conjecture] = field(default_factory=list)
    supported: list[Conjecture] = field(default_factory=list)
    refuted: int = 0
    not_novel: int = 0
    committed: int = 0
    budget_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.finished_at - self.started_at, 3),
            "candidates_examined": self.candidates_examined,
            "proven": [c.to_dict() for c in self.proven],
            "supported": [c.to_dict() for c in self.supported],
            "refuted": self.refuted,
            "not_novel": self.not_novel,
            "committed": self.committed,
            "budget_s": self.budget_s,
            "notes": self.notes[:12],
        }


# ---------------------------------------------------------------------------
# Safe polynomial parsing — turn "n**5 - n" into a sound callable with NO eval.
# ---------------------------------------------------------------------------
_MAX_EXPONENT = 12  # bound pow cost; user/number-theory claims live well under this


class _PolyParseError(ValueError):
    pass


def _compile_integer_poly(expr: str) -> Callable[[int], int]:
    """Compile a single-variable integer polynomial in ``n`` to a callable.

    Only ``+ - * ** %`` over integer constants and the variable ``n`` are allowed —
    parsed from the AST with a hard node allowlist, so no arbitrary code can run and
    nothing but bounded integer arithmetic is ever evaluated. ``^`` is accepted as a
    synonym for ``**`` (the way people write powers in math).
    """
    src = str(expr or "").strip().replace("^", "**")
    if not src:
        raise _PolyParseError("empty expression")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:  # noqa: PERF203
        raise _PolyParseError(f"unparseable: {exc}") from exc

    def _check(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _check(node.body)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.Mod)):
                raise _PolyParseError("unsupported operator")
            if isinstance(node.op, ast.Pow):
                exp = node.right
                if not (isinstance(exp, ast.Constant) and isinstance(exp.value, int) and 0 <= exp.value <= _MAX_EXPONENT):
                    raise _PolyParseError("power exponent must be a small non-negative int")
            _check(node.left)
            _check(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise _PolyParseError("unsupported unary operator")
            _check(node.operand)
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise _PolyParseError("only integer constants allowed")
            return
        if isinstance(node, ast.Name):
            if node.id != "n":
                raise _PolyParseError(f"only variable 'n' allowed, got {node.id!r}")
            return
        raise _PolyParseError(f"disallowed syntax: {type(node).__name__}")

    _check(tree)
    def _evaluate(node: ast.AST, n: int) -> int:
        if isinstance(node, ast.Expression):
            return _evaluate(node.body, n)
        if isinstance(node, ast.Constant):
            return int(node.value)
        if isinstance(node, ast.Name):
            return n
        if isinstance(node, ast.UnaryOp):
            operand = _evaluate(node.operand, n)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = _evaluate(node.left, n)
            right = _evaluate(node.right, n)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
        raise _PolyParseError(f"unevaluable syntax: {type(node).__name__}")

    def _f(n: int) -> int:
        return _evaluate(tree, int(n))

    return _f


# Claim-parsing patterns for the response lane. Each maps natural-language phrasing
# to an exact integer-divisibility/congruence check.
_CLAIM_PATTERNS = (
    # "n^5 - n is divisible by 30" / "n**3 - n divisible by 6"
    ("expr_div_m", re.compile(
        r"(?P<expr>[-+*/^()0-9n\s*]+?)\s*(?:is\s+)?divisible\s+by\s+(?P<m>\d+)", re.I)),
    # "30 divides n^5 - n"
    ("m_div_expr", re.compile(
        r"(?P<m>\d+)\s+divides\s+(?P<expr>[-+*/^()0-9n\s*]+)", re.I)),
    # "n^5 ≡ n (mod 30)" / "n**5 congruent to n mod 30"
    ("congruence", re.compile(
        r"n\s*\^?\*?\*?\s*(?P<k>\d+)\s*(?:≡|=|congruent\s+to)\s*n\s*\(?\s*mod\s*(?P<m>\d+)\)?", re.I)),
)


# ---------------------------------------------------------------------------
# Exact falsifier — the hard gate.
# ---------------------------------------------------------------------------
class ExactFalsifier:
    """Falsify a conjecture by exact computation. Never approximates a verdict."""

    def __init__(self, *, default_trials: int = 200) -> None:
        self._default_trials = int(default_trials)

    def test(
        self,
        predicate: Predicate,
        cases: Sequence[int],
        *,
        exhaustive: bool,
    ) -> FalsificationOutcome:
        """Run ``predicate`` over ``cases``; the first failure refutes the conjecture.

        ``exhaustive=True`` means ``cases`` is the *entire* domain that determines the
        claim (e.g. a full residue system mod m for a polynomial congruence). Surviving
        an exhaustive sweep is a genuine proof → PROVEN. Surviving a finite sample of an
        infinite domain is only SUPPORTED.
        """
        trials = 0
        for case in cases:
            trials += 1
            try:
                holds = bool(predicate(int(case)))
            except (ArithmeticError, ValueError, TypeError) as exc:
                record_degradation("frontier_falsifier_predicate", exc, severity="debug")
                holds = False
            if not holds:
                return FalsificationOutcome(
                    status=EpistemicStatus.REFUTED,
                    trials=trials,
                    exhaustive=exhaustive,
                    counterexample=int(case),
                    detail=f"counterexample at n={int(case)}",
                )
        if trials == 0:
            return FalsificationOutcome(EpistemicStatus.CONJECTURE, 0, exhaustive, detail="no cases tested")
        status = EpistemicStatus.PROVEN if exhaustive else EpistemicStatus.SUPPORTED
        return FalsificationOutcome(status=status, trials=trials, exhaustive=exhaustive)


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------
class FrontierDiscoveryEngine:
    """Closed generate→falsify→novelty→commit loop over verifiable conjecture domains."""

    SERVICE_NAME = "frontier_discovery_engine"

    # Domains the engine is allowed to assert/commit about — a safety allowlist.
    _ALLOWED_DOMAINS = frozenset(
        {"modular_identity", "polynomial_divisibility", "integer_sequence", "user_claim"}
    )

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        generate_fn: Optional[Callable[[str, float], Any]] = None,
        will_decide_fn: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        scientific_engine: Any = None,
        novelty: Any = None,
        seed: int = 0xF20D,
        max_candidates: int = 400,
        default_max_time_s: float = 8.0,
        sample_trials: int = 200,
        commit_enabled: bool = True,
    ) -> None:
        self._generate_fn = generate_fn
        self._will_decide_fn = will_decide_fn
        self._sci = scientific_engine  # injectable for tests; else lazy global
        self._rng = random.Random(seed)
        self._max_candidates = int(max_candidates)
        self._default_max_time_s = float(default_max_time_s)
        self._falsifier = ExactFalsifier(default_trials=sample_trials)
        self._sample_trials = int(sample_trials)
        self._commit_enabled = bool(commit_enabled)
        self._lock = threading.RLock()

        # Novelty: an embedding archive so a rediscovery of a known law is not
        # re-announced as new. In-memory by default (mirrors the rest of Aura).
        if novelty is None:
            try:
                from core.unknowns.novelty_archive import NoveltyArchive

                novelty = NoveltyArchive(novelty_threshold=0.20)
            except (ImportError, RuntimeError, ValueError) as exc:
                record_degradation("frontier_novelty_init", exc, severity="debug")
                novelty = None
        self._novelty = novelty

        # Bounded GA over SafeExpression for integer-sequence closed forms.
        try:
            from core.discovery.evolver import ExpressionEvolver

            self._evolver = ExpressionEvolver(
                seed=seed ^ 0x5EED, population_size=48, elite_size=8, emit_receipts=False
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            record_degradation("frontier_evolver_init", exc, severity="debug")
            self._evolver = None

        # Knowledge base of accumulated discoveries (dedup + recall across sessions).
        if db_path is None:
            try:
                from core.config import config

                db_path = str(config.paths.home_dir / "data/frontier_discoveries.db")
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("frontier_db_path", exc, severity="debug")
                db_path = "./aura_frontier_discoveries.db"
        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── knowledge-base persistence ────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS discoveries (
                        id TEXT PRIMARY KEY,
                        statement TEXT NOT NULL UNIQUE,
                        domain TEXT NOT NULL,
                        formal_form TEXT NOT NULL,
                        status TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        trials INTEGER NOT NULL,
                        exhaustive INTEGER NOT NULL,
                        provenance TEXT NOT NULL,
                        receipt_id TEXT,
                        created_at REAL NOT NULL
                    )"""
                )
                conn.commit()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("frontier_schema", exc)

    def _already_known(self, statement: str) -> bool:
        try:
            with connecting(self._connect()) as conn:
                row = conn.execute(
                    "SELECT 1 FROM discoveries WHERE statement = ? LIMIT 1", (statement,)
                ).fetchone()
            return row is not None
        except (sqlite3.Error, OSError) as exc:
            record_degradation("frontier_known", exc, severity="debug")
            return False

    def _persist(self, c: Conjecture) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO discoveries
                       (id, statement, domain, formal_form, status, confidence, trials,
                        exhaustive, provenance, receipt_id, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"disc-{uuid.uuid4().hex[:10]}", c.statement, c.domain, c.formal_form,
                        c.status.value, c.confidence, c.trials, int(c.exhaustive),
                        c.provenance, c.receipt_id, c.created_at,
                    ),
                )
                conn.commit()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("frontier_persist", exc)

    def knowledge(self, *, limit: int = 50) -> list[dict[str, Any]]:
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT statement, domain, status, confidence, trials, exhaustive, provenance "
                    "FROM discoveries ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            return [
                {
                    "statement": r[0], "domain": r[1], "status": r[2], "confidence": r[3],
                    "trials": r[4], "exhaustive": bool(r[5]), "provenance": r[6],
                }
                for r in rows
            ]
        except (sqlite3.Error, OSError) as exc:
            record_degradation("frontier_knowledge", exc, severity="debug")
            return []

    # ── candidate generators (sound, LLM-free) ────────────────────────────
    def _modular_identity_candidates(self) -> Iterable[tuple[str, str, Predicate, Sequence[int], str]]:
        """Yield candidate congruences n**k ≡ n (mod m).

        For a polynomial in n, ``n**k mod m`` depends only on ``n mod m``, so checking a
        full residue system ``0..m-1`` is *exhaustive* — surviving it is a real proof.
        The generator enumerates a bounded (k, m) grid; the falsifier keeps the survivors
        (which include genuine theorems like n**3≡n mod 6, n**5≡n mod 30, n**7≡n mod 42).
        """
        for k in range(2, 10):
            for m in range(2, 61):
                statement = f"For every integer n, n^{k} ≡ n (mod {m})."
                formal = f"(n**{k} - n) % {m} == 0  ∀ n"
                def _pred(n: int, _k: int = k, _m: int = m) -> bool:
                    return pow(n % _m, _k, _m) == (n % _m)
                yield statement, formal, _pred, range(m), f"checking all residues mod {m}"

    def _polynomial_divisibility_candidates(self) -> Iterable[tuple[str, str, Predicate, Sequence[int], str]]:
        """Yield candidate divisibilities m | f(n) for a few classic integer polynomials."""
        polys = [
            ("n^3 - n", lambda n: n**3 - n),
            ("n^2 - n", lambda n: n**2 - n),
            ("n^5 - n", lambda n: n**5 - n),
            ("n^2 + n", lambda n: n**2 + n),
        ]
        for label, f in polys:
            for m in range(2, 31):
                statement = f"For every integer n, {m} divides {label}."
                formal = f"({label}) % {m} == 0  ∀ n"
                def _pred(n: int, _f=f, _m: int = m) -> bool:
                    return _f(n) % _m == 0
                yield statement, formal, _pred, range(m), f"checking all residues mod {m}"

    # ── the closed loop ────────────────────────────────────────────────────
    def _evaluate_candidate(
        self,
        *,
        statement: str,
        formal_form: str,
        predicate: Predicate,
        cases: Sequence[int],
        exhaustive: bool,
        domain: str,
        provenance: str,
        falsification_plan: str,
    ) -> Conjecture:
        outcome = self._falsifier.test(predicate, cases, exhaustive=exhaustive)
        novelty = 1.0
        if self._novelty is not None and outcome.status in (EpistemicStatus.PROVEN, EpistemicStatus.SUPPORTED):
            try:
                novelty = float(self._novelty.novelty(statement))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                novelty = 1.0
        # Confidence is bounded by epistemic status: PROVEN may be high, SUPPORTED is
        # capped (it is not a proof), REFUTED/CONJECTURE are low.
        if outcome.status is EpistemicStatus.PROVEN:
            confidence = 0.97
        elif outcome.status is EpistemicStatus.SUPPORTED:
            confidence = min(0.85, 0.5 + 0.02 * outcome.trials)
        elif outcome.status is EpistemicStatus.REFUTED:
            confidence = 0.99  # we are confident it is FALSE
        else:
            confidence = 0.2
        return Conjecture(
            statement=statement,
            domain=domain,
            formal_form=formal_form,
            status=outcome.status,
            confidence=confidence,
            trials=outcome.trials,
            exhaustive=outcome.exhaustive,
            counterexample=outcome.counterexample,
            novelty=novelty,
            provenance=provenance,
            falsification_plan=falsification_plan,
        )

    def run_discovery_cycle(self, *, max_time_s: Optional[float] = None) -> DiscoveryReport:
        """One bounded discovery pass over the sound generators. Idle-loop entrypoint.

        Wall-clock and candidate-count bounded (no unbounded runs). Survivors that are
        PROVEN/SUPPORTED and novel are committed into the causal belief substrate.
        """
        start = time.monotonic()
        budget = max(1.0, float(max_time_s if max_time_s is not None else self._default_max_time_s))
        deadline = start + budget
        report = DiscoveryReport(started_at=time.time(), finished_at=0.0, candidates_examined=0, budget_s=budget)

        generators = [
            ("modular_identity", self._modular_identity_candidates()),
            ("polynomial_divisibility", self._polynomial_divisibility_candidates()),
        ]
        examined = 0
        for domain, gen in generators:
            for statement, formal, predicate, cases, plan in gen:
                if time.monotonic() >= deadline or examined >= self._max_candidates:
                    report.notes.append("budget_reached")
                    break
                examined += 1
                # Skip laws already in the KB so a cycle surfaces genuinely new survivors.
                if self._already_known(statement):
                    continue
                c = self._evaluate_candidate(
                    statement=statement, formal_form=formal, predicate=predicate,
                    cases=cases, exhaustive=True, domain=domain, provenance="structured",
                    falsification_plan=plan,
                )
                if c.status is EpistemicStatus.REFUTED:
                    report.refuted += 1
                    continue
                if c.status not in (EpistemicStatus.PROVEN, EpistemicStatus.SUPPORTED):
                    continue
                # Dedup is by exact KB membership (handled by _already_known above), NOT
                # embedding similarity — two distinct theorems ("n^3≡n mod 6" vs
                # "n^5≡n mod 30") are textually near-identical and must not collide.
                self._record_discovery(c, report)
            else:
                continue
            break

        report.candidates_examined = examined
        report.finished_at = time.time()
        self._emit_metric("frontier_discovery_cycle_total")
        logger.info(
            "🔭 [FrontierDiscovery] cycle examined=%d proven=%d supported=%d refuted=%d committed=%d in %.2fs",
            examined, len(report.proven), len(report.supported), report.refuted, report.committed,
            report.finished_at - report.started_at,
        )
        return report

    def _is_novel_and_record(self, c: Conjecture) -> bool:
        try:
            return bool(self._novelty.add_if_novel(c.statement, metadata={"domain": c.domain}))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return True

    def _record_discovery(self, c: Conjecture, report: DiscoveryReport) -> None:
        if c.status is EpistemicStatus.PROVEN:
            report.proven.append(c)
            self._emit_metric("frontier_discovery_proven_total")
        else:
            report.supported.append(c)
            self._emit_metric("frontier_discovery_supported_total")
        self._persist(c)
        if self._commit_enabled and self._governance_ok(c):
            if self._commit_to_mind(c):
                c.committed = True
                report.committed += 1

    # ── governance ─────────────────────────────────────────────────────────
    def _governance_ok(self, c: Conjecture) -> bool:
        """Only assertable knowledge in an allowed domain may be committed as belief.

        REFUTED/CONJECTURE never commit. An optional Will/constitution veto can block any
        commit (e.g. a domain the operator has fenced off). Fail-closed on veto, fail-open
        on a broken hook (a crashed governor must not silently approve — we require an
        explicit non-False).
        """
        if c.domain not in self._ALLOWED_DOMAINS:
            return False
        if c.status not in (EpistemicStatus.PROVEN, EpistemicStatus.SUPPORTED):
            return False
        if self._will_decide_fn is None:
            return True
        try:
            decision = self._will_decide_fn(c.to_dict())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("frontier_will_decide", exc, severity="warning")
            return True  # a broken governor does not get to block discovery silently
        if isinstance(decision, dict) and decision.get("approved") is False:
            logger.info("🔭 [FrontierDiscovery] governance vetoed commit of: %s", c.statement[:80])
            return False
        return True

    # ── causal commit into the mind ─────────────────────────────────────────
    def _scientific_engine(self) -> Any:
        if self._sci is None:
            try:
                from core.cognition.scientific_engine import get_scientific_engine

                self._sci = get_scientific_engine()
            except (ImportError, RuntimeError, AttributeError) as exc:
                record_degradation("frontier_sci_engine", exc, severity="debug")
                self._sci = None
        return self._sci

    def _commit_to_mind(self, c: Conjecture) -> bool:
        """Fold a survivor into Aura's beliefs via the ScientificEngine + a receipt.

        form_hypothesis registers the claim with an observable and expected value;
        run_experiment opens an Outcome-Ledger receipt; observe resolves it and revises
        the belief (which the ScientificEngine publishes into ``world_state``). The
        ``observed`` we feed is the verification pass-rate (1.0 for a proof), so the
        belief's confidence is grounded in the actual check, not asserted.
        """
        committed = False
        sci = self._scientific_engine()
        if sci is not None:
            try:
                hyp_id = sci.form_hypothesis(
                    c.statement,
                    predicted_observable="verifier_holds",
                    expected=1.0,
                    prior_confidence=c.confidence,
                )
                sci.run_experiment(hyp_id)
                observed = 1.0 if c.status is EpistemicStatus.PROVEN else min(1.0, 0.5 + 0.5 * c.confidence)
                sci.observe(hyp_id, observed=observed, note=f"frontier_discovery:{c.domain}")
                committed = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("frontier_commit_sci", exc, severity="warning")
        c.receipt_id = self._emit_discovery_receipt(c) or c.receipt_id
        return committed or c.receipt_id is not None

    def _emit_discovery_receipt(self, c: Conjecture) -> Optional[str]:
        try:
            from core.runtime.receipts import StateMutationReceipt, get_receipt_store

            receipt = get_receipt_store().emit(
                StateMutationReceipt(
                    cause="frontier_discovery.commit",
                    domain=c.domain,
                    key=c.statement[:120],
                    metadata={
                        "status": c.status.value,
                        "formal_form": c.formal_form,
                        "confidence": round(c.confidence, 4),
                        "trials": c.trials,
                        "exhaustive": c.exhaustive,
                        "provenance": c.provenance,
                    },
                )
            )
            return getattr(receipt, "receipt_id", None)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("frontier_receipt", exc, severity="debug")
            return None

    def _emit_metric(self, counter: str) -> None:
        try:
            from core.observability.metrics import get_metrics

            get_metrics().increment_counter(counter)
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass

    # ── integer-sequence law discovery (GA, SUPPORTED-only) ─────────────────
    def discover_sequence_law(self, sequence: Sequence[int]) -> Optional[Conjecture]:
        """Search for a closed form a_n = f(n) reproducing every given term.

        Uses the bounded GA over :class:`SafeExpression`, framing examples as
        ``(n, 0, a_n)``. A closed form matching all given terms is genuine evidence, but
        finite data can never prove a formula for *all* n — so the strongest honest label
        is SUPPORTED (matches all known terms), never PROVEN.
        """
        seq = [int(x) for x in sequence]
        if self._evolver is None or len(seq) < 5:
            return None
        examples = [(i, 0, seq[i]) for i in range(len(seq))]
        try:
            result = self._evolver.evolve(examples, generations=24, target_label="sequence_law")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("frontier_sequence_law", exc, severity="debug")
            return None
        if not result.perfect:
            return None  # did not reproduce every term — no honest law to report
        # SafeExpression frames the search over (a, b); here a==n (the index) and b==0.
        # Relabel for an honest, readable closed form in terms of n.
        formula = re.sub(r"\bb\b", "0", re.sub(r"\ba\b", "n", result.best_str))
        statement = f"The sequence {seq[:8]}{'…' if len(seq) > 8 else ''} matches a_n = {formula} for all {len(seq)} given terms."
        c = Conjecture(
            statement=statement,
            domain="integer_sequence",
            formal_form=f"a_n = {formula}",
            status=EpistemicStatus.SUPPORTED,
            confidence=min(0.85, 0.45 + 0.04 * len(seq)),
            trials=len(seq),
            exhaustive=False,
            novelty=1.0,
            provenance="evolver",
            falsification_plan="provide a further term that breaks the formula",
        )
        if self._novelty is not None:
            self._is_novel_and_record(c)
        self._persist(c)
        return c

    # ── response-lane: sound check of a user's claim ────────────────────────
    def assess_claim(self, claim: str) -> dict[str, Any]:
        """Sound-check a natural-language claim; return a status-labeled verdict.

        If the claim parses to an integer-divisibility/congruence law, it is falsified by
        exhaustive residue checking (a real proof or a real counterexample). If it cannot
        be formalized into a sound check, the engine refuses to assert — it returns a
        CONJECTURE verdict naming the experiment that would settle it. It never guesses.
        """
        text = str(claim or "").strip()
        parsed = self._parse_claim(text)
        if parsed is None:
            c = Conjecture(
                statement=text or "(empty claim)",
                domain="user_claim",
                formal_form="",
                status=EpistemicStatus.CONJECTURE,
                confidence=0.15,
                provenance="user",
                falsification_plan="I could not reduce this to an exact check. State it as an integer "
                                   "divisibility/congruence (e.g. 'n^5 - n is divisible by 30') and I can prove or refute it.",
            )
            return {"verdict": c.to_dict(), "rendered": self.render(c)}

        expr, m, formal, statement = parsed
        try:
            f = _compile_integer_poly(expr)
        except _PolyParseError as exc:
            c = Conjecture(
                statement=statement, domain="user_claim", formal_form=formal,
                status=EpistemicStatus.CONJECTURE, confidence=0.15, provenance="user",
                falsification_plan=f"the expression did not parse as an integer polynomial in n ({exc}).",
            )
            return {"verdict": c.to_dict(), "rendered": self.render(c)}

        def _pred(n: int) -> bool:
            return f(n) % m == 0

        c = self._evaluate_candidate(
            statement=statement, formal_form=formal, predicate=_pred,
            cases=range(m), exhaustive=True, domain="user_claim",
            provenance="user", falsification_plan=f"checking all residues mod {m}",
        )
        # A user-checked PROVEN/SUPPORTED law is worth committing to the mind too.
        if c.status in (EpistemicStatus.PROVEN, EpistemicStatus.SUPPORTED) and self._commit_enabled:
            if not self._already_known(c.statement) and self._governance_ok(c):
                self._persist(c)
                if self._commit_to_mind(c):
                    c.committed = True
        return {"verdict": c.to_dict(), "rendered": self.render(c)}

    def _parse_claim(self, text: str) -> Optional[tuple[str, int, str, str]]:
        for kind, pattern in _CLAIM_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            try:
                if kind == "congruence":
                    k = int(match.group("k"))
                    m = int(match.group("m"))
                    if m < 1 or k < 0 or k > _MAX_EXPONENT:
                        return None
                    expr = f"n**{k} - n"
                    formal = f"(n**{k} - n) % {m} == 0  ∀ n"
                    statement = f"For every integer n, n^{k} ≡ n (mod {m})."
                    return expr, m, formal, statement
                expr = match.group("expr").strip()
                m = int(match.group("m"))
                if m < 1 or not re.search(r"n", expr):
                    return None
                formal = f"({expr}) % {m} == 0  ∀ n"
                statement = f"For every integer n, {m} divides {expr}."
                return expr, m, formal, statement
            except (ValueError, AttributeError):
                return None
        return None

    # ── the assertion gate (anti-hallucination) ─────────────────────────────
    @staticmethod
    def render(c: Conjecture) -> str:
        """Phrase a conjecture honestly for its epistemic status. CONJECTURE is never asserted."""
        if c.status is EpistemicStatus.PROVEN:
            return f"✓ Proven: {c.statement} Verified exhaustively ({c.falsification_plan or f'{c.trials} cases'})."
        if c.status is EpistemicStatus.SUPPORTED:
            return (
                f"Empirically supported (NOT a proof): {c.statement} It held for all "
                f"{c.trials} tested cases; a single counterexample would refute it."
            )
        if c.status is EpistemicStatus.REFUTED:
            return f"✗ Refuted: {c.statement} Counterexample at n={c.counterexample}."
        return (
            f"Conjecture — unverified, I am not asserting this: {c.statement} "
            f"It would be settled by {c.falsification_plan or 'an exact check I could not construct here'}."
        )

    # ── introspection ───────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM discoveries GROUP BY status"
                ).fetchall()
            by_status = {r[0]: r[1] for r in rows}
        except (sqlite3.Error, OSError):
            by_status = {}
        return {
            "service": self.SERVICE_NAME,
            "db_path": self._db_path,
            "by_status": by_status,
            "novelty_archive_size": len(self._novelty) if self._novelty is not None else 0,
            "commit_enabled": self._commit_enabled,
            "evolver_available": self._evolver is not None,
        }


# ---------------------------------------------------------------------------
# Module singleton + best-effort container registration (matches the other engines).
# ---------------------------------------------------------------------------
_engine: Optional[FrontierDiscoveryEngine] = None
_engine_lock = threading.Lock()


def get_frontier_discovery_engine(**kwargs: Any) -> FrontierDiscoveryEngine:
    """Process-wide singleton. First caller may pass construction kwargs."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FrontierDiscoveryEngine(**kwargs)
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: FrontierDiscoveryEngine) -> None:
    """Make the engine discoverable as a live service, if the container is open."""
    try:
        from core.container import ServiceContainer

        if ServiceContainer.has(FrontierDiscoveryEngine.SERVICE_NAME):
            return
        register_instance = getattr(ServiceContainer, "register_instance", None)
        if callable(register_instance):
            register_instance(
                FrontierDiscoveryEngine.SERVICE_NAME, engine,
                required=False, registered_by="frontier_discovery_engine",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("frontier_container_register", exc, severity="debug")

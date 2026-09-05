"""Trusted proof kernel — the de Bruijn criterion for Aura's deduction.

Lean's architecture splits proving into an untrusted *elaborator* (tactics,
search, heuristics — free to be clever and wrong) and a small trusted *kernel*
that re-checks every produced proof term against the fixed rules. This module
gives Aura the same discipline over the Pantheon tableau prover:

- :func:`check_proof` is the kernel: a deliberately small, independent checker
  that re-verifies a :class:`~core.reasoning.natural_deduction.CertStep`
  certificate node by node against its own copy of the α/β rule schemas. It
  shares only the formula AST with the search — never the search code — so a
  bug in proof *search* cannot produce an accepted false theorem.
- The checker also computes the **axiom audit** (Lean's ``#print axioms``):
  which premises a refutation actually touches, so every theorem records what
  it truly depends on rather than what it was handed.
- :class:`TheoremLedger` is the epistemic bookkeeping: kernel-verified theorems,
  **admitted claims** (Lean's ``sorry`` — assertions accepted without proof),
  and taint propagation: any theorem transitively resting on an admitted claim
  is visibly tainted until the admission is discharged by a real proof.

Live wiring: ``SymbolicBridge.prove_logic`` refuses to report success for a
kernel-rejected proof (fail closed), ``deduction_governance`` surfaces ledger
stats in its governance signal, belief-consistency contradictions are
kernel-certified (``Γcore ⊢ ⊥``), and the live inference audit records every
non-sequitur conclusion in Aura's own reasoning as an admitted claim.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.reasoning.natural_deduction import (
    And,
    Bot,
    CertStep,
    Formula,
    Implies,
    Not,
    Or,
    Proof,
    formula_from_dict,
    formula_to_dict,
    parse,
    prove,
)

# Hard caps so an adversarial or degenerate certificate cannot wedge the
# checker; both bounds fail closed (reject, never accept).
_KERNEL_MAX_NODES = 200_000
_KERNEL_MAX_DEPTH = 900

_GOAL_ORIGIN = -1  # origin index of the negated goal in the root branch


class _KernelReject(Exception):
    """Internal: certificate failed a kernel check (carries the reason)."""


# ── The kernel's own rule table ───────────────────────────────────────────
# Independent re-statement of the tableau schemas. Deliberately duplicated
# from the search: this small table *is* the trusted base.

def _kernel_negate(f: Formula) -> Formula:
    return f.f if isinstance(f, Not) else Not(f)


def _kernel_expand(f: Formula) -> list[list[Formula]] | None:
    """Branches mandated by ``f``'s schema, or None when ``f`` is a literal."""
    if isinstance(f, Not):
        g = f.f
        if isinstance(g, Not):                      # ¬¬A ⇒ A
            return [[g.f]]
        if isinstance(g, Bot):                      # ¬⊥ ⇒ (nothing)
            return [[]]
        if isinstance(g, And):                      # ¬(A∧B) ⇒ ¬A | ¬B
            return [[_kernel_negate(g.a)], [_kernel_negate(g.b)]]
        if isinstance(g, Or):                       # ¬(A∨B) ⇒ ¬A, ¬B
            return [[_kernel_negate(g.a), _kernel_negate(g.b)]]
        if isinstance(g, Implies):                  # ¬(A→B) ⇒ A, ¬B
            return [[g.a, _kernel_negate(g.b)]]
        return None                                  # ¬atom: literal
    if isinstance(f, And):                          # A∧B ⇒ A, B
        return [[f.a, f.b]]
    if isinstance(f, Or):                           # A∨B ⇒ A | B
        return [[f.a], [f.b]]
    if isinstance(f, Implies):                      # A→B ⇒ ¬A | B
        return [[_kernel_negate(f.a)], [f.b]]
    return None                                      # atom / ⊥: literal


# ── Verdicts ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KernelVerdict:
    """Outcome of an independent kernel check of one proof certificate."""

    verified: bool
    reason: str
    used_premises: tuple[str, ...] = ()
    uses_goal_negation: bool = False
    nodes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "used_premises": list(self.used_premises),
            "uses_goal_negation": self.uses_goal_negation,
            "nodes": self.nodes,
        }


class _WalkState:
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0


def _walk(
    branch: dict[Formula, frozenset[int]],
    node: CertStep,
    state: _WalkState,
    depth: int,
) -> frozenset[int]:
    """Check one certificate node against the branch; return used origins."""
    state.nodes += 1
    if state.nodes > _KERNEL_MAX_NODES:
        raise _KernelReject(f"certificate exceeds {_KERNEL_MAX_NODES} nodes")
    if depth > _KERNEL_MAX_DEPTH:
        raise _KernelReject(f"certificate exceeds depth {_KERNEL_MAX_DEPTH}")

    if node.kind == "close":
        t = node.target
        if isinstance(t, Bot):
            if t not in branch:
                raise _KernelReject("closure claims ⊥ but ⊥ is not in the branch")
            return branch[t]
        neg = Not(t)
        if t not in branch or neg not in branch:
            raise _KernelReject(f"closure pair {{{t}, ¬{t}}} not present in the branch")
        return branch[t] | branch[neg]

    if node.kind == "expand":
        t = node.target
        if t not in branch:
            raise _KernelReject(f"expansion target {t} is not in the branch")
        schema = _kernel_expand(t)
        if schema is None:
            raise _KernelReject(f"expansion of literal {t} is not a rule")
        if len(node.children) != len(schema):
            raise _KernelReject(
                f"schema for {t} mandates {len(schema)} branch(es), certificate has {len(node.children)}"
            )
        target_origins = branch[t]
        used: frozenset[int] = frozenset()
        for child, adds in zip(node.children, schema):
            child_branch = dict(branch)
            del child_branch[t]
            for a in adds:
                child_branch[a] = child_branch.get(a, frozenset()) | target_origins
            used |= _walk(child_branch, child, state, depth + 1)
        return used

    raise _KernelReject(f"unknown certificate node kind {node.kind!r}")


def check_proof(
    premises: Sequence[Formula],
    goal: Formula,
    certificate: CertStep,
) -> KernelVerdict:
    """Independently verify that ``certificate`` is a closed tableau for
    ``premises ∪ {¬goal}`` — i.e. a real proof of ``premises ⊢ goal``.

    Never trusts the search: the root branch is rebuilt here, every expansion
    is re-derived from the kernel's own schema table, and every closure pair is
    re-checked for membership. Origins are threaded through so the verdict
    reports exactly which premises the refutation used (the axiom audit).
    """
    branch: dict[Formula, frozenset[int]] = {}
    for i, p in enumerate(premises):
        branch[p] = branch.get(p, frozenset()) | {i}
    neg_goal = _kernel_negate(goal)
    branch[neg_goal] = branch.get(neg_goal, frozenset()) | {_GOAL_ORIGIN}

    state = _WalkState()
    try:
        used = _walk(branch, certificate, state, 0)
    except _KernelReject as exc:
        return KernelVerdict(False, f"kernel rejected: {exc}", nodes=state.nodes)
    except RecursionError:
        return KernelVerdict(False, "kernel rejected: certificate recursion overflow", nodes=state.nodes)

    used_premises = tuple(
        sorted({str(premises[i]) for i in used if i >= 0})
    )
    return KernelVerdict(
        True,
        "verified: closed tableau checked against kernel schemas",
        used_premises=used_premises,
        uses_goal_negation=_GOAL_ORIGIN in used,
        nodes=state.nodes,
    )


# ── Universal checker registry ────────────────────────────────────────────
# The kernel discipline is engine-agnostic: any proof engine that emits a
# checkable certificate registers an independent checker here and gains the
# same fail-closed verification, axiom audit, and ledger bookkeeping. The
# tableau checker is the first citizen; resolution/SMT-trace checkers can
# join without touching the ledger or its consumers.

CheckerFn = Callable[[Sequence[Formula], Formula, Any], KernelVerdict]

_checker_lock = threading.Lock()
_checkers: dict[str, CheckerFn] = {}


def register_checker(method: str, checker: CheckerFn) -> None:
    with _checker_lock:
        _checkers[method] = checker


def registered_checkers() -> list[str]:
    with _checker_lock:
        return sorted(_checkers)


def check_certificate(
    method: str,
    premises: Sequence[Formula],
    goal: Formula,
    certificate: Any,
) -> KernelVerdict:
    """Dispatch a certificate to the registered checker for its proof method."""
    with _checker_lock:
        checker = _checkers.get(method)
    if checker is None:
        return KernelVerdict(False, f"no kernel checker registered for method {method!r}")
    return checker(premises, goal, certificate)


register_checker("analytic_tableau", check_proof)

# Replay codecs: per-method deserializers so the ledger can re-verify stored
# theorems of ANY registered proof method from their serialized encodings.
_replay_codec_lock = threading.Lock()
_replay_codecs: dict[str, Callable[[Any], bool]] = {}


def register_replay_codec(method: str, replayer: Callable[[Any], bool]) -> None:
    """Register ``replayer(encoded) -> bool`` for a proof method's stored form."""
    with _replay_codec_lock:
        _replay_codecs[method] = replayer


# ── Theorem ledger: axiom audit + admitted (sorry) discipline ─────────────

@dataclass(frozen=True)
class AdmittedClaim:
    """A claim accepted without proof — Aura's ``sorry``. Tracked, never free."""

    claim_id: str
    statement: str
    reason: str
    source: str
    admitted_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "reason": self.reason,
            "source": self.source,
            "admitted_at": self.admitted_at,
        }


@dataclass(frozen=True)
class Theorem:
    """A kernel-verified theorem with its true dependency footprint."""

    theorem_id: str
    goal: str
    premises: tuple[str, ...]
    used_premises: tuple[str, ...]
    admitted_deps: tuple[str, ...]     # admitted claim_ids this rests on, transitively
    certificate_sha256: str
    certificate_nodes: int
    checked_at: float

    @property
    def tainted(self) -> bool:
        return bool(self.admitted_deps)

    def to_dict(self) -> dict[str, object]:
        return {
            "theorem_id": self.theorem_id,
            "goal": self.goal,
            "premises": list(self.premises),
            "used_premises": list(self.used_premises),
            "admitted_deps": list(self.admitted_deps),
            "tainted": self.tainted,
            "certificate_sha256": self.certificate_sha256,
            "certificate_nodes": self.certificate_nodes,
            "checked_at": self.checked_at,
        }


def _cert_fingerprint(cert: CertStep) -> str:
    def ser(n: CertStep) -> str:
        return f"({n.kind}|{n.target}|{','.join(ser(c) for c in n.children)})"

    return hashlib.sha256(ser(cert).encode("utf-8")).hexdigest()


def json_dumps_canonical(data: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _norm_text(statement: str) -> str:
    return " ".join(str(statement or "").strip().lower().split())


class TheoremLedger:
    """Process-wide record of what is proved, what is admitted, and what is tainted.

    Statements are keyed by their canonical formula string when a formula is
    available (``str(Formula)`` — parse-normalized), else by normalized text
    with a ``text:`` prefix, so NL admissions and formal proofs share one
    namespace without colliding.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._admitted: dict[str, AdmittedClaim] = {}
        self._theorems: dict[str, Theorem] = {}
        # Replay store: full structural encodings so every recorded theorem
        # can be re-checked from scratch later (Lean's "re-elaborate the
        # olean" discipline — trust nothing that cannot be re-verified).
        self._replayable: dict[str, dict[str, Any]] = {}
        self._kernel_checks = 0
        self._kernel_rejections = 0
        self._discharged = 0

    @staticmethod
    def statement_key(statement: str, formula: Formula | None = None) -> str:
        if formula is not None:
            return str(formula)
        try:
            return str(parse(statement))
        except (ValueError, RuntimeError):
            return f"text:{_norm_text(statement)}"

    # ── admissions (sorry) ────────────────────────────────────────────

    def admit(
        self,
        statement: str,
        *,
        reason: str = "",
        source: str = "",
        formula: Formula | None = None,
    ) -> AdmittedClaim:
        """Record a claim accepted without proof. Idempotent per statement."""
        key = self.statement_key(statement, formula)
        with self._lock:
            existing = self._admitted.get(key)
            if existing is not None:
                return existing
            claim = AdmittedClaim(
                claim_id=f"adm-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}",
                statement=statement,
                reason=reason,
                source=source,
                admitted_at=time.time(),
            )
            self._admitted[key] = claim
            return claim

    def retract(self, statement: str, formula: Formula | None = None) -> bool:
        key = self.statement_key(statement, formula)
        with self._lock:
            return self._admitted.pop(key, None) is not None

    # ── theorems ──────────────────────────────────────────────────────

    def note_check(self, verified: bool) -> None:
        with self._lock:
            self._kernel_checks += 1
            if not verified:
                self._kernel_rejections += 1

    def record(
        self,
        goal: Formula,
        premises: Sequence[Formula],
        verdict: KernelVerdict,
        certificate: CertStep,
    ) -> Theorem:
        """Record a kernel-verified theorem, computing its admitted-dep closure.

        Only call with ``verdict.verified``; the ledger refuses anything else.
        A verified, untainted proof of a previously admitted statement
        *discharges* the admission (the ``sorry`` is closed by a real proof).
        """
        if not verdict.verified:
            raise ValueError("TheoremLedger.record requires a kernel-verified proof")
        goal_key = str(goal)
        with self._lock:
            deps: set[str] = set()
            for p in verdict.used_premises:
                admitted = self._admitted.get(p)
                if admitted is not None:
                    deps.add(admitted.claim_id)
                prior = self._theorems.get(p)
                if prior is not None:
                    deps.update(prior.admitted_deps)
            theorem = Theorem(
                theorem_id=f"thm-{hashlib.sha256((goal_key + '|' + '|'.join(verdict.used_premises)).encode('utf-8')).hexdigest()[:12]}",
                goal=goal_key,
                premises=tuple(str(p) for p in premises),
                used_premises=verdict.used_premises,
                admitted_deps=tuple(sorted(deps)),
                certificate_sha256=_cert_fingerprint(certificate),
                certificate_nodes=certificate.node_count(),
                checked_at=time.time(),
            )
            self._theorems[goal_key] = theorem
            self._replayable[theorem.theorem_id] = {
                "method": "analytic_tableau",
                "goal": formula_to_dict(goal),
                "premises": [formula_to_dict(p) for p in premises],
                "certificate": certificate.to_dict(),
                "certificate_sha256": theorem.certificate_sha256,
            }
            if not deps and goal_key in self._admitted:
                del self._admitted[goal_key]
                self._discharged += 1
            return theorem

    def record_external(
        self,
        *,
        method: str,
        goal: str,
        premises: Sequence[str],
        used_premises: Sequence[str],
        encoded: Mapping[str, Any],
    ) -> Theorem:
        """Record a kernel-verified theorem from a non-tableau proof method.

        The caller must have already run the method's registered checker
        (fail closed at the call site); the ledger stores the serialized
        encoding so :meth:`replay` can re-verify it via the replay codec.
        Admitted-claim taint propagates exactly as for tableau theorems.
        """
        goal_key = goal
        with self._lock:
            deps: set[str] = set()
            for p in used_premises:
                admitted = self._admitted.get(p)
                if admitted is not None:
                    deps.add(admitted.claim_id)
                prior = self._theorems.get(p)
                if prior is not None:
                    deps.update(prior.admitted_deps)
            canonical = json_dumps_canonical(dict(encoded))
            theorem = Theorem(
                theorem_id=f"thm-{hashlib.sha256((method + '|' + goal_key).encode('utf-8')).hexdigest()[:12]}",
                goal=goal_key,
                premises=tuple(premises),
                used_premises=tuple(used_premises),
                admitted_deps=tuple(sorted(deps)),
                certificate_sha256=hashlib.sha256(canonical).hexdigest(),
                certificate_nodes=0,
                checked_at=time.time(),
            )
            self._theorems[goal_key] = theorem
            self._replayable[theorem.theorem_id] = {
                "method": method,
                "encoded": dict(encoded),
                "certificate_sha256": theorem.certificate_sha256,
            }
            if not deps and goal_key in self._admitted:
                del self._admitted[goal_key]
                self._discharged += 1
            return theorem

    # ── replay: nothing recorded is beyond re-verification ────────────

    def export_replayable(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Structural encodings of recorded theorems, most recent first."""
        with self._lock:
            items = sorted(
                self._replayable.items(),
                key=lambda kv: self._theorem_checked_at(kv[0]),
                reverse=True,
            )
        out = [dict(v, theorem_id=k) for k, v in items]
        return out[:limit] if limit is not None else out

    def _theorem_checked_at(self, theorem_id: str) -> float:
        for t in self._theorems.values():
            if t.theorem_id == theorem_id:
                return t.checked_at
        return 0.0

    def replay(self, *, limit: int | None = None) -> dict[str, Any]:
        """Re-check recorded certificates from their stored encodings.

        Deserializes every certificate and runs it back through the kernel —
        a stored theorem that no longer re-verifies (corruption, drift, or a
        historical checker bug) is surfaced, never silently kept.
        """
        from core.reasoning.natural_deduction import CertStep

        checked = 0
        failed: list[str] = []
        for entry in self.export_replayable(limit=limit):
            checked += 1
            method = str(entry.get("method", "analytic_tableau"))
            try:
                if method == "analytic_tableau":
                    goal = formula_from_dict(entry["goal"])
                    premises = [formula_from_dict(p) for p in entry["premises"]]
                    cert = CertStep.from_dict(entry["certificate"])
                    if _cert_fingerprint(cert) != entry["certificate_sha256"]:
                        failed.append(str(entry["theorem_id"]))
                        continue
                    verdict = check_certificate(method, premises, goal, cert)
                    if not verdict.verified:
                        failed.append(str(entry["theorem_id"]))
                    continue
                with _replay_codec_lock:
                    codec = _replay_codecs.get(method)
                if codec is None:
                    failed.append(str(entry["theorem_id"]))
                    continue
                encoded = entry["encoded"]
                if hashlib.sha256(json_dumps_canonical(encoded)).hexdigest() != entry["certificate_sha256"]:
                    failed.append(str(entry["theorem_id"]))
                    continue
                if not codec(encoded):
                    failed.append(str(entry["theorem_id"]))
            except (ValueError, TypeError, KeyError):
                failed.append(str(entry.get("theorem_id", "?")))
        return {"checked": checked, "failed": failed, "ok": not failed}

    # ── queries (the #print axioms surface) ───────────────────────────

    def axioms_of(self, statement: str, formula: Formula | None = None) -> dict[str, object]:
        """What does this statement rest on? Lean's ``#print axioms``."""
        key = self.statement_key(statement, formula)
        with self._lock:
            theorem = self._theorems.get(key)
            admitted = self._admitted.get(key)
            admitted_index = {c.claim_id: c for c in self._admitted.values()}
        if theorem is None and admitted is None:
            return {"status": "unknown", "statement": statement}
        if theorem is None and admitted is not None:
            return {
                "status": "admitted",
                "statement": statement,
                "admitted": admitted.to_dict(),
            }
        assert theorem is not None
        return {
            "status": "tainted" if theorem.tainted else "proved",
            "statement": statement,
            "used_premises": list(theorem.used_premises),
            "admitted_deps": [
                admitted_index[cid].to_dict() if cid in admitted_index else {"claim_id": cid}
                for cid in theorem.admitted_deps
            ],
        }

    def theorem_for(self, statement: str, formula: Formula | None = None) -> Theorem | None:
        key = self.statement_key(statement, formula)
        with self._lock:
            return self._theorems.get(key)

    def open_admissions(self) -> list[AdmittedClaim]:
        with self._lock:
            return sorted(self._admitted.values(), key=lambda c: c.admitted_at)

    def stats(self) -> dict[str, int]:
        with self._lock:
            tainted = sum(1 for t in self._theorems.values() if t.tainted)
            return {
                "kernel_checks": self._kernel_checks,
                "kernel_rejections": self._kernel_rejections,
                "theorems": len(self._theorems),
                "theorems_tainted": tainted,
                "admitted_open": len(self._admitted),
                "admitted_discharged": self._discharged,
            }


_ledger_lock = threading.Lock()
_ledger: TheoremLedger | None = None


def get_theorem_ledger() -> TheoremLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = TheoremLedger()
        return _ledger


def reset_theorem_ledger_for_test() -> TheoremLedger:
    global _ledger
    with _ledger_lock:
        _ledger = TheoremLedger()
        return _ledger


# ── Certified proving: search → kernel → ledger, fail closed ──────────────

@dataclass
class CertifiedProof:
    """A proof-search result together with the kernel's independent verdict."""

    proof: Proof
    verdict: KernelVerdict | None = None
    theorem: Theorem | None = None

    @property
    def verified(self) -> bool:
        return bool(self.verdict and self.verdict.verified)

    def to_dict(self) -> dict[str, object]:
        return {
            "proof": self.proof.to_dict(),
            "kernel": self.verdict.to_dict() if self.verdict else None,
            "theorem": self.theorem.to_dict() if self.theorem else None,
            "verified": self.verified,
        }


def _coerce(f: Formula | str) -> Formula:
    return f if isinstance(f, Formula) else parse(f)


def prove_certified(
    premises: Iterable[Formula | str],
    goal: Formula | str,
    *,
    ledger: TheoremLedger | None = None,
) -> CertifiedProof:
    """Prove, then make the kernel check it, then record the theorem.

    The one honest entry point for consequential deduction: a proof only counts
    when the independent kernel accepts its certificate. A search-claimed proof
    the kernel rejects is a prover soundness bug — recorded as a CRITICAL
    degradation and reported *unverified* (fail closed), never silently trusted.
    """
    prem = [_coerce(p) for p in premises]
    goal_f = _coerce(goal)
    proof = prove(prem, goal_f)
    if not proof.provable:
        return CertifiedProof(proof=proof)

    reg = ledger if ledger is not None else get_theorem_ledger()
    if proof.certificate is None:
        # Search fell back to the certificateless path (budget); the verdict is
        # honest but unchecked — surface that instead of pretending.
        reg.note_check(False)
        return CertifiedProof(
            proof=proof,
            verdict=KernelVerdict(False, "no certificate produced (search budget fallback)"),
        )

    verdict = check_certificate(proof.method, prem, goal_f, proof.certificate)
    reg.note_check(verdict.verified)
    if not verdict.verified:
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "proof_kernel",
                RuntimeError(verdict.reason),
                severity="critical",
                action="kernel rejected a search-claimed proof; reporting unverified",
                extra={"goal": str(goal_f), "premises": [str(p) for p in prem]},
                enforce_failure_policy=False,
            )
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
            # Degradation reporting must never mask the (already fail-closed) verdict.
            pass
        return CertifiedProof(proof=proof, verdict=verdict)

    theorem = reg.record(goal_f, prem, verdict, proof.certificate)
    return CertifiedProof(proof=proof, verdict=verdict, theorem=theorem)


def prove_certified_text(
    premises: Iterable[str],
    goal: str,
    *,
    ledger: TheoremLedger | None = None,
) -> CertifiedProof:
    """String-friendly :func:`prove_certified`."""
    return prove_certified(list(premises), goal, ledger=ledger)


def prove_equivalent(
    f: Formula | str,
    g: Formula | str,
    *,
    ledger: TheoremLedger | None = None,
) -> tuple[bool, CertifiedProof, CertifiedProof | None]:
    """Kernel-certified logical equivalence: ``f ⊢ g`` and ``g ⊢ f``.

    Both directions must be independently verified. Returns
    ``(equivalent, forward, backward)`` — the discipline behind semantic
    deduplication: two claims are the *same* belief only when the kernel
    certifies both entailments, never on string similarity.
    """
    forward = prove_certified([f], g, ledger=ledger)
    if not forward.verified:
        return False, forward, None
    backward = prove_certified([g], f, ledger=ledger)
    return backward.verified, forward, backward


__all__ = [
    "AdmittedClaim",
    "CertifiedProof",
    "KernelVerdict",
    "Theorem",
    "TheoremLedger",
    "check_certificate",
    "check_proof",
    "get_theorem_ledger",
    "prove_certified",
    "prove_certified_text",
    "prove_equivalent",
    "register_checker",
    "register_replay_codec",
    "registered_checkers",
    "reset_theorem_ledger_for_test",
]

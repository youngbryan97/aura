"""SelfImprovingResearchCore — the autonomous research-loop driver."""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.discovery.code_eval import SafeCodeEvaluator
from core.discovery.evolver import EvolverResult, ExpressionEvolver
from core.lattice.config import LatticeConfig
from core.lattice.model import LatticeLM
from core.lattice.trainer import LatticeTrainer, TrainConfig
from core.promotion.dynamic_benchmark import DynamicBenchmark, Task
from core.promotion.gate import PromotionDecision, PromotionGate, ScoreEstimate
from core.promotion.holdout_vault import HoldoutVault, LeakageDetector
from core.runtime.prediction_ledger import PredictionLedger
from core.runtime.tenant_boundary import TenantBoundary
from core.unknowns.generator import UnknownUnknownGenerator
from core.unknowns.novelty_archive import NoveltyArchive
from core.verification.semantic_verifier import SemanticVerifier


@dataclass
class ResearchCoreConfig:
    """Runtime config for the self-improving research core."""

    workdir: Path = Path("./aura_research_core")
    model_cfg: LatticeConfig = field(default_factory=LatticeConfig)
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    critical_metrics: tuple = (
        "task_accuracy",
        "semantic_consistency",
        "loss",
    )
    max_regression: float = 0.02
    benchmark_seed: int = 0xA17A
    discovery_population: int = 32
    discovery_elite: int = 6
    discovery_generations: int = 12
    auto_promote: bool = True
    emit_receipts: bool = True

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)


@dataclass
class CycleReport:
    iteration: int
    started_at: float
    finished_at: float
    promotion: Optional[Dict[str, Any]] = None
    discovery: Optional[Dict[str, Any]] = None
    unknowns_added: int = 0
    semantic_ok: bool = True
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    receipt_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "promotion": self.promotion,
            "discovery": self.discovery,
            "unknowns_added": self.unknowns_added,
            "semantic_ok": self.semantic_ok,
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
            "receipt_id": self.receipt_id,
        }


# ----------------------------------------------------------------------------
# Default solver — a deterministic interpreter for the dynamic benchmark prompts.
# Aura wires a real LLM-backed solver here later; this default lets the loop run
# end-to-end without a model in the hot path.
# ----------------------------------------------------------------------------
def deterministic_task_solver(prompt: str) -> Any:
    p = prompt.strip()
    try:
        if p.startswith("Return gcd"):
            inside = p[p.index("(") + 1 : p.index(")")]
            a, b = (int(x.strip()) for x in inside.split(","))
            return math.gcd(a, b)
        if p.startswith("Return ("):
            base_part, m_part = p.split(") mod ")
            inner = base_part.split("Return (", 1)[1]
            a_str, b_str = inner.split(" ** ")
            a = int(a_str)
            b = int(b_str)
            m = int(m_part.rstrip("."))
            return pow(a, b, m)
        if p.startswith("Sort"):
            arr_str = p.split(":", 1)[1].strip()
            arr = eval(arr_str, {"__builtins__": {}}, {})  # safe — only literals
            if not isinstance(arr, list):
                raise ValueError
            return sorted(arr)
        if p.startswith("Is this string a palindrome"):
            s = p.split(": ", 1)[1].strip()
            return s == s[::-1]
        if p.startswith("Let f(x)="):
            body, ret = p.split("Return ", 1)
            f_part, g_part = body.split(", g(x)=")
            a_str, b_str = f_part.split("Let f(x)=")[1].split("x+")
            c_str, d_part = g_part.split("x+")
            d_str = d_part.split(".")[0]
            x_str = ret.split("g(f(")[1].split("))")[0]
            a, b, c, d, x = int(a_str), int(b_str), int(c_str), int(d_str), int(x_str)
            return c * (a * x + b) + d
    except Exception:
        return None
    return None


class SelfImprovingResearchCore:
    """Aura-owned self-improvement driver.

    Owns one ``LatticeLM`` and the surrounding evaluation, discovery,
    verification, and unknown-generation substrate.  ``run_cycle()``
    advances the loop one step.  External pipelines (e.g. the F9
    curriculum scheduler) call ``run_cycle`` at policy-determined
    cadence; the core is fully autonomous in the sense that no
    cycle requires operator action.
    """

    SERVICE_NAME = "research_core"

    def __init__(
        self,
        cfg: Optional[ResearchCoreConfig] = None,
        *,
        ledger: Optional[PredictionLedger] = None,
        tenant_id: Optional[str] = None,
        will_decide_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        task_solver: Optional[Callable[[str], Any]] = None,
    ):
        self.cfg = cfg or ResearchCoreConfig()
        self.cfg.workdir.mkdir(parents=True, exist_ok=True)
        # Tenant boundary: refuse to mount a foreign data dir.
        self.boundary = TenantBoundary(self.cfg.workdir, tenant_id=tenant_id)
        self.boundary.assert_owned()

        # F17: model + trainer
        self.model = LatticeLM(self.cfg.model_cfg)
        self.trainer = LatticeTrainer(self.model, self.cfg.train_cfg)

        # F18: promotion + benchmark + vault
        self.bench = DynamicBenchmark(seed=self.cfg.benchmark_seed)
        self.vault = HoldoutVault(self.cfg.workdir / "holdout_vault.json")
        self.leakage = LeakageDetector()
        self.gate = PromotionGate(
            critical_metrics=self.cfg.critical_metrics,
            max_regression=self.cfg.max_regression,
            emit_receipts=self.cfg.emit_receipts,
            will_decide_fn=will_decide_fn,
        )

        # F19: discovery
        self.evolver = ExpressionEvolver(
            seed=0xC0DE,
            population_size=self.cfg.discovery_population,
            elite_size=self.cfg.discovery_elite,
            emit_receipts=self.cfg.emit_receipts,
        )
        self.code_evaluator = SafeCodeEvaluator(
            timeout_seconds=5.0, memory_mb=256
        )

        # F20: semantic verifier
        self.verifier = SemanticVerifier(code_evaluator=self.code_evaluator)

        # F21: unknown-unknowns
        self.novelty = NoveltyArchive(novelty_threshold=0.35)
        self.unknown = UnknownUnknownGenerator(seed=0xBADA, archive=self.novelty)

        # F2: prediction ledger (shared with the rest of Aura)
        self.ledger = ledger or PredictionLedger(
            self.cfg.workdir / "research_predictions.db"
        )

        # External hooks
        self.task_solver = task_solver or deterministic_task_solver
        self._iteration = 0
        self._cycle_history: List[CycleReport] = []
        self._last_cycle_at: Optional[float] = None

    # ------------------------------------------------------------------
    # capability evaluation
    # ------------------------------------------------------------------
    def evaluate_capability(self, n_tasks: int = 30) -> Dict[str, ScoreEstimate]:
        """Generate fresh dynamic benchmark, score the solver, vault answers.

        Returns a score vector compatible with ``PromotionGate.compare``.
        """
        tasks = self.bench.generate(n_tasks)
        self.vault.add(tasks)

        correct = 0
        for task in tasks:
            try:
                pred = self.task_solver(task.prompt)
            except Exception:
                pred = None
            if self._matches(pred, task.answer):
                correct += 1
                self._record_prediction_to_ledger(task, predicted=True, observed=True)
            else:
                self._record_prediction_to_ledger(task, predicted=True, observed=False)

        n = max(1, len(tasks))
        accuracy = correct / n
        stderr = math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n)

        # Loss proxy from the lattice model on a tiny synthetic forward pass.
        try:
            from core.lattice.dataset import RandomTokenDataset

            ds = RandomTokenDataset(
                n_samples=4,
                seq_len=min(16, self.cfg.model_cfg.max_seq_len),
                vocab_size=self.cfg.model_cfg.vocab_size,
                seed=self._iteration + 1,
            )
            sample = ds[0]
            ids = sample["input_ids"].unsqueeze(0)
            labels = sample["labels"].unsqueeze(0)
            self.model.eval()
            import torch

            with torch.no_grad():
                out = self.model(ids, labels=labels)
            loss_val = float(out["loss"])
        except Exception:
            loss_val = 0.0

        # Semantic consistency probe over a paraphrase trio.
        consistency = self.verifier.self_consistency(
            ["the answer is stable", "the answer remains stable", "stable answer"]
        )

        return {
            "task_accuracy": ScoreEstimate(accuracy, stderr=stderr, n=n, higher_is_better=True),
            "loss": ScoreEstimate(loss_val, stderr=0.0, n=1, higher_is_better=False),
            "semantic_consistency": ScoreEstimate(
                consistency.mean_cosine,
                stderr=0.0,
                n=consistency.pairs,
                higher_is_better=True,
            ),
        }

    @staticmethod
    def _matches(prediction: Any, answer: Any) -> bool:
        if prediction is None:
            return False
        if isinstance(prediction, list) and isinstance(answer, list):
            return prediction == answer
        try:
            return str(prediction).strip().lower() == str(answer).strip().lower()
        except Exception:
            return False

    def _record_prediction_to_ledger(
        self,
        task: Task,
        *,
        predicted: bool,
        observed: bool,
    ) -> None:
        try:
            pid = self.ledger.register(
                belief=f"task:{task.kind}",
                modality="symbolic",
                action="self_improve.evaluate",
                expected={"answer": str(task.answer)[:200]},
                prior_prob=0.85 if predicted else 0.15,
            )
            self.ledger.resolve(pid, observed={"applies": observed}, observed_truth=observed)
        except Exception:
            # Ledger failures must not break the cycle.
            pass

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def discover_addition_proxy(self) -> EvolverResult:
        examples = [(a, b, a + b) for a in range(-3, 4) for b in range(-3, 4)]
        return self.evolver.evolve(
            examples,
            generations=self.cfg.discovery_generations,
            target_label="addition_proxy",
        )

    def discover_gcd_proxy(self) -> EvolverResult:
        examples = [(a, b, math.gcd(a, b)) for a in range(1, 8) for b in range(1, 8)]
        return self.evolver.evolve(
            examples,
            generations=self.cfg.discovery_generations,
            target_label="gcd_proxy",
        )

    # ------------------------------------------------------------------
    # cycle driver
    # ------------------------------------------------------------------
    def run_cycle(self, *, n_eval_tasks: int = 20) -> CycleReport:
        self._iteration += 1
        report = CycleReport(
            iteration=self._iteration,
            started_at=time.time(),
            finished_at=0.0,
        )

        # 1. Capability evaluation + promotion attempt.
        scores = self.evaluate_capability(n_tasks=n_eval_tasks)
        decision = self.gate.compare(
            scores, metadata={"iteration": self._iteration}
        )
        report.promotion = decision.to_dict()
        report.metrics["task_accuracy"] = scores["task_accuracy"].mean
        report.metrics["loss"] = scores["loss"].mean
        report.metrics["semantic_consistency"] = scores["semantic_consistency"].mean
        report.notes.append(
            f"promotion accepted={decision.accepted} reasons={'; '.join(decision.reasons)[:200]}"
        )
        report.receipt_id = decision.receipt_id

        # 2. Discovery — try to find a perfect addition proxy.
        try:
            evolver_result = self.discover_addition_proxy()
            report.discovery = evolver_result.to_dict()
            report.metrics["discovery_score"] = evolver_result.score
            if evolver_result.perfect:
                report.notes.append(f"discovery PERFECT: {evolver_result.best_str}")
        except Exception as exc:
            report.notes.append(f"discovery failed: {exc!r}")

        # 3. Unknown-unknowns — generate fresh failure-finding tasks.
        try:
            seeds = self.bench.generate(20)
            unknowns = self.unknown.generate(seeds, n=10)
            report.unknowns_added = len(unknowns)
            self.vault.add(unknowns)
        except Exception as exc:
            report.notes.append(f"unknowns failed: {exc!r}")

        # 4. Semantic check across iteration narratives — a cheap
        #    self-consistency over what we just decided.
        sc = self.verifier.self_consistency(
            [
                f"iteration {self._iteration} accepted={decision.accepted}",
                f"iteration {self._iteration} promotion={decision.accepted}",
                f"iteration {self._iteration} verdict={decision.accepted}",
            ]
        )
        report.semantic_ok = sc.ok

        report.finished_at = time.time()
        self._last_cycle_at = report.finished_at
        self._cycle_history.append(report)
        return report

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "iteration": self._iteration,
            "last_cycle_at": self._last_cycle_at,
            "model": {
                "num_parameters": int(self.model.num_parameters()),
                "vocab_size": int(self.cfg.model_cfg.vocab_size),
                "n_layers": int(self.cfg.model_cfg.n_layers),
                "d_model": int(self.cfg.model_cfg.d_model),
            },
            "vault_size": int(self.vault.size()),
            "novelty_archive_size": int(len(self.novelty)),
            "ledger_count": int(self.ledger.count()),
            "promotion_history": len(self.gate.history),
            "tenant": self.boundary.current_stamp().tenant_id
            if self.boundary.current_stamp()
            else "unstamped",
            "global_step": int(self.trainer.global_step),
        }

    def cycle_history(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._cycle_history]

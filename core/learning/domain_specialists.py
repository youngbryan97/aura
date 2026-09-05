"""Domain-specialist adapter training — the supply side of the expert-LoRA library.

The general compounding loop fuses its adapter into the next serving artifact;
specialists have different physics on purpose:

  * adapter-ONLY artifacts (~40MB) — never fused. A fused generation already
    contains its delta; registering it as an expert too would double-apply.
  * trained on ONE domain's verified contrast pairs from the canonical
    preference store (rows carry ``domain`` provenance);
  * gated on a domain-CONCENTRATED sealed battery (specialist seeds live at
    2000+, disjoint from training seeds <1000 and the general gate's 1000+),
    plus a general non-collapse check;
  * registered into the ExpertLoRALibrary on promotion, where the router
    hot-attaches them onto the resident model for matching background work
    (0.01s attach, measured on a real adapter).

Every train/eval runs in a subprocess so at most one extra model is ever in
memory beside the serving one; every decision lands in a receipt on disk.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.learning.heldout_battery import BatterySpec, generate_battery, text_collides_with_battery
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DomainSpecialists")

SPECIALIST_SEED_BASE = 2000     # specialist gate seeds: ≥2000, per-domain offset
GENERAL_SEED = 1500             # non-collapse check seed (≥1000 eval floor, ≠ general gate)
_RECOVERABLE = (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError)


@dataclass
class SpecialistConfig:
    work_root: Path
    store_path: Path                    # canonical preference store (domain rows)
    base_model: str = ""                # "" → resolve the serving cortex fresh
    min_pairs: int = 24
    max_pairs: int = 240
    battery_size: int = 24
    general_size: int = 24
    epsilon: float = 0.0                # candidate must BEAT base on-domain
    general_drop_tolerance: float = 0.10
    iters: int = 60
    batch_size: int = 1
    learning_rate: float = 5e-6
    num_layers: int = 16
    max_seq_length: int = 1024
    train_timeout_s: int = 3600
    eval_timeout_s: int = 1800
    eval_max_tokens: int = 256


@dataclass
class SpecialistReceipt:
    domain: str
    base_model: str = ""
    status: str = "started"             # blocked | train_failed | eval_failed | refused | promoted
    reasons: list[str] = field(default_factory=list)
    pair_count: int = 0
    domain_base_accuracy: float | None = None
    domain_candidate_accuracy: float | None = None
    general_base_accuracy: float | None = None
    general_candidate_accuracy: float | None = None
    adapter_path: str = ""
    registered_as: str = ""
    run_dir: str = ""
    elapsed_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _domain_seed(domain: str) -> int:
    import zlib

    return SPECIALIST_SEED_BASE + (zlib.crc32(domain.encode()) % 997)


class DomainSpecialistTrainer:
    """Train, gate, and register one domain specialist at a time."""

    def __init__(
        self,
        config: SpecialistConfig,
        *,
        command_runner: Callable[..., Any] | None = None,
        library: Any = None,
    ) -> None:
        self.config = config
        self.config.work_root.mkdir(parents=True, exist_ok=True)
        self._library = library
        if command_runner is None:
            from core.learning.weight_compounding import _default_command_runner

            command_runner = _default_command_runner
        self._run_command = command_runner

    # ── supply ────────────────────────────────────────────────────────────────

    def domain_pair_counts(self) -> dict[str, int]:
        """Pairs per domain in the store (only rows that carry provenance)."""
        counts: dict[str, int] = {}
        store = Path(self.config.store_path)
        if not store.exists():
            return counts
        try:
            with store.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    domain = str(row.get("domain", "") or "").strip()
                    # selfplay tags domains as "selfplay:<domain>" — normalize
                    domain = domain.split(":", 1)[-1] if ":" in domain else domain
                    if domain:
                        counts[domain] = counts.get(domain, 0) + 1
        except OSError as exc:
            record_degradation("domain_specialists", exc,
                               action="treated preference store as empty after read failure")
        return counts

    def eligible_domains(self) -> list[str]:
        return sorted(
            d for d, n in self.domain_pair_counts().items() if n >= self.config.min_pairs
        )

    def _load_domain_rows(self, domain: str, battery_tasks) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        store = Path(self.config.store_path)
        if not store.exists():
            return rows
        with store.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_domain = str(row.get("domain", "") or "").strip()
                row_domain = row_domain.split(":", 1)[-1] if ":" in row_domain else row_domain
                if row_domain != domain:
                    continue
                prompt = str(row.get("prompt", "")).strip()
                chosen = str(row.get("chosen", "")).strip()
                rejected = str(row.get("rejected", "")).strip()
                if not prompt or not chosen or chosen == rejected:
                    continue
                if text_collides_with_battery(f"{prompt}\n{chosen}\n{rejected}", battery_tasks):
                    continue
                rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                if len(rows) >= self.config.max_pairs:
                    break
        return rows

    # ── the cycle ─────────────────────────────────────────────────────────────

    def resolve_base(self) -> str:
        if self.config.base_model:
            return self.config.base_model
        from core.brain.llm.model_registry import get_model_path

        return get_model_path()

    def train_domain(self, domain: str) -> SpecialistReceipt:
        started = time.time()
        receipt = SpecialistReceipt(domain=domain)
        run_dir = self.config.work_root / f"{domain}-{int(started)}"
        receipt.run_dir = str(run_dir)
        try:
            base = self.resolve_base()
            receipt.base_model = base

            seed = _domain_seed(domain)
            pool = generate_battery(BatterySpec(seed=seed, size=self.config.battery_size * 10))
            battery_tasks = [t for t in pool if t.domain == domain][: self.config.battery_size]
            general_tasks = generate_battery(
                BatterySpec(seed=GENERAL_SEED, size=self.config.general_size)
            )
            if len(battery_tasks) < max(8, self.config.battery_size // 2):
                receipt.status = "blocked"
                receipt.reasons.append(f"battery_too_small:{len(battery_tasks)}")
                return receipt

            rows = self._load_domain_rows(domain, battery_tasks + general_tasks)
            receipt.pair_count = len(rows)
            if len(rows) < self.config.min_pairs:
                receipt.status = "blocked"
                receipt.reasons.append(f"insufficient_pairs:{len(rows)}<{self.config.min_pairs}")
                return receipt

            run_dir.mkdir(parents=True, exist_ok=True)
            data_dir = run_dir / "data"
            data_dir.mkdir(exist_ok=True)
            split = max(2, len(rows) // 10)
            atomic_write_text(
                data_dir / "train.jsonl",
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split:]),
                encoding="utf-8",
            )
            atomic_write_text(
                data_dir / "valid.jsonl",
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:split]),
                encoding="utf-8",
            )
            atomic_write_text(
                data_dir / "test.jsonl",
                json.dumps(rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            adapter_dir = run_dir / "adapter"
            trained, detail = self._train(base, data_dir, adapter_dir)
            if not trained:
                receipt.status = "train_failed"
                receipt.reasons.append(detail)
                return receipt
            receipt.adapter_path = str(adapter_dir)

            evals = {
                "domain_base": self._eval(base, "", seed, self.config.battery_size,
                                          run_dir / "domain_base.json", domain),
                "domain_candidate": self._eval(base, str(adapter_dir), seed, self.config.battery_size,
                                               run_dir / "domain_candidate.json", domain),
                "general_base": self._eval(base, "", GENERAL_SEED, self.config.general_size,
                                           run_dir / "general_base.json", ""),
                "general_candidate": self._eval(base, str(adapter_dir), GENERAL_SEED,
                                                self.config.general_size,
                                                run_dir / "general_candidate.json", ""),
            }
            if any(v is None for v in evals.values()):
                receipt.status = "eval_failed"
                receipt.reasons.append("eval_subprocess_failed")
                return receipt
            receipt.domain_base_accuracy = evals["domain_base"]
            receipt.domain_candidate_accuracy = evals["domain_candidate"]
            receipt.general_base_accuracy = evals["general_base"]
            receipt.general_candidate_accuracy = evals["general_candidate"]

            improved = evals["domain_candidate"] > evals["domain_base"] + self.config.epsilon
            collapsed = evals["general_candidate"] < (
                evals["general_base"] - self.config.general_drop_tolerance
            )
            if not improved:
                receipt.status = "refused"
                receipt.reasons.append(
                    f"no_domain_gain:{evals['domain_candidate']:.3f}<="
                    f"{evals['domain_base']:.3f}+{self.config.epsilon}"
                )
                return receipt
            if collapsed:
                receipt.status = "refused"
                receipt.reasons.append(
                    f"general_collapse:{evals['general_candidate']:.3f}<"
                    f"{evals['general_base']:.3f}-{self.config.general_drop_tolerance}"
                )
                return receipt

            receipt.registered_as = self._register(domain, adapter_dir, base, evals)
            receipt.status = "promoted"
            return receipt
        except _RECOVERABLE as exc:
            receipt.status = "failed"
            receipt.reasons.append(f"{type(exc).__name__}:{exc}")
            record_degradation(
                "domain_specialists",
                exc,
                action=f"specialist cycle for '{domain}' recorded failure receipt and stopped",
            )
            return receipt
        finally:
            receipt.elapsed_s = round(time.time() - started, 2)
            self._persist_receipt(receipt)

    # ── seams ─────────────────────────────────────────────────────────────────

    def _train(self, base_model: str, data_dir: Path, adapter_dir: Path) -> tuple[bool, str]:
        """Reuse the compounding loop's DPO train seam (same trainer, same guards)."""
        from core.learning.weight_compounding import CompoundingConfig, WeightCompoundingLoop

        cfg = self.config
        loop = WeightCompoundingLoop(
            CompoundingConfig(
                work_root=self.config.work_root / ".trainer",
                fused_root=self.config.work_root / ".trainer" / "fused",
                iters=cfg.iters,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                num_layers=cfg.num_layers,
                max_seq_length=cfg.max_seq_length,
                train_timeout_s=cfg.train_timeout_s,
            ),
            command_runner=self._run_command,
        )
        return loop.train(base_model, data_dir, adapter_dir, "dpo")

    def _eval(
        self,
        model: str,
        adapter: str,
        seed: int,
        size: int,
        output: Path,
        domain: str,
    ) -> float | None:
        import sys

        eval_tool = Path(__file__).resolve().parent.parent.parent / "tools" / "heldout_eval.py"
        command = [
            sys.executable, str(eval_tool),
            "--model", model,
            "--seed", str(seed),
            "--size", str(size),
            "--max-tokens", str(self.config.eval_max_tokens),
            "--output", str(output),
        ]
        if adapter:
            command += ["--adapter-path", adapter]
        if domain:
            command += ["--domains", domain]
        result = self._run_command(tuple(command), float(self.config.eval_timeout_s))
        returncode = getattr(result, "returncode", 1)
        if returncode != 0 or not output.exists():
            return None
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
            return float(report.get("accuracy"))
        except _RECOVERABLE:
            return None

    def _register(self, domain: str, adapter_dir: Path, base: str, evals: dict) -> str:
        library = self._library
        if library is None:
            from core.brain.expert_lora_library import get_expert_lora_library

            library = get_expert_lora_library()
        from core.brain.expert_lora_library import LoRAAdapter

        name = f"{domain}-specialist-{time.strftime('%Y%m%d-%H%M%S')}"
        size_mb = sum(
            f.stat().st_size for f in adapter_dir.glob("*") if f.is_file()
        ) / (1024 * 1024)
        library.register(
            LoRAAdapter(
                name=name,
                path=str(adapter_dir),
                base_model=str(base),
                task_types={domain},
                keywords={domain} | set(domain.split("_")),
                size_mb=size_mb,
                quality=float(evals["domain_candidate"]),
                source="domain_specialist",
            )
        )
        return name

    def _persist_receipt(self, receipt: SpecialistReceipt) -> None:
        try:
            receipts = self.config.work_root / "receipts"
            receipts.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                receipts / f"{receipt.domain}-{int(receipt.created_at)}.json",
                json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "domain_specialists",
                exc,
                action="continued with unpersisted specialist receipt",
            )


__all__ = [
    "SPECIALIST_SEED_BASE",
    "GENERAL_SEED",
    "SpecialistConfig",
    "SpecialistReceipt",
    "DomainSpecialistTrainer",
]

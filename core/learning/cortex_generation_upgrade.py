"""core/learning/cortex_generation_upgrade.py

Cortex generation upgrades — the governed path to frontier-grade compiled
understanding in her OWN weights.

The Compiled Understanding Layer closes the assimilation gap with machinery;
this pipeline closes it at the substrate: replacing the base checkpoint with
a newer generation (e.g. Qwen2.5→Qwen3 class) whose weights carry richer
conceptual machinery — while preserving identity, and never deciding alone.

The pipeline is complete, tested software; the DECISION is deliberately not
software. Swapping the mind's base model is an identity-level act, so
activation hard-requires an explicit operator authorization string and a
PASS evaluation receipt. That gate is the design, not a limitation.

Stages, each receipted:

  evaluate   — run the capability battery (breadth cloze probes, verifiable
               reasoning micro-tasks, identity behavior snapshot) on the
               current and candidate models, behind a MEMORY GUARD that
               refuses to load a candidate the host cannot afford beside
               the live processes (a second 32B during a training run is a
               memory incident, not an experiment).
  plan       — enumerate the identity artifacts a generation swap must
               migrate (fused persona/CRSM deltas, CAA steering vectors,
               expert adapters), with per-step lane: what is automatic and
               what requires an operator-launched training run.
  stage      — write the staged activation pointer + a byte-exact rollback
               copy of the current pointer. Nothing live changes.
  activate   — flip training/fused-model/active.json to the staged target
               (governed write). Requires authorization + PASS evaluation.
               Takes effect at the next boot — the running mind is never
               hot-swapped.
  rollback   — restore the rollback copy, byte-exact, verified.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.brain.llm.model_artifact_profile import (
    build_model_artifact_descriptor,
    validate_model_artifact_descriptor,
    validate_model_serving_profile,
)
from core.learning.cortex_migration_authority import validate_component_authority
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CortexGenerationUpgrade")

EVALUATION_SCHEMA = "aura.cortex_upgrade.evaluation.v3"
EVALUATION_PROGRESS_SCHEMA = "aura.cortex_upgrade.evaluation_progress.v1"
MIGRATION_PLAN_SCHEMA = "aura.cortex_upgrade.migration_plan.v1"
MIGRATION_CONTRACT_SCHEMA = "aura.cortex_upgrade.migration_contract.v2"
STAGING_SCHEMA = "aura.cortex_upgrade.staging.v2"
ACTIVATION_SCHEMA = "aura.cortex_upgrade.activation.v2"
IDENTITY_NORMALIZATION_SCHEMA = "aura.cortex_upgrade.identity_normalization.v1"
IDENTITY_TRANSITION_SCHEMA = "aura.cortex_upgrade.identity_transition.v1"

_REQUIRED_CRITICAL_GATES = frozenset(
    {
        "template",
        "complete_answer",
        "tool_contract",
        "code_contract",
        "context",
        "identity_migration",
        "latency",
        "memory",
    }
)
_REQUIRED_MIGRATION_COMPONENTS = frozenset(
    {"persona_crsm", "steering", "expert_adapters", "recurrence_native"}
)

STAGED_POINTER_NAME = "active.json.staged"
ROLLBACK_POINTER_NAME = "active.json.rollback"
IDENTITY_BACKUP_POINTER_NAME = "active.json.identity-backup"

# Memory guard: candidate projected RSS = on-disk weight bytes × this factor
# (activation buffers, cache); the host must retain this many GB free AFTER
# the load or the guard refuses.
_LOAD_OVERHEAD_FACTOR = 1.3
_FREE_MARGIN_GB = 8.0
# Qwen3.8's non-thinking lane can still spend roughly one hundred tokens on a
# short derivation before emitting the factual object.  Evidence-matched probes
# stop as soon as the object appears, so this is a completion ceiling rather
# than a mandatory decode length.  Keep it aligned with the bounded reasoning
# envelope instead of silently grading a correct, unfinished derivation as a
# knowledge regression.
_BREADTH_MAX_TOKENS = 256

# Breadth probes: factual cloze items with acceptable-answer alternates.
# Deterministic greedy decoding, case-insensitive containment scoring. These
# measure compiled knowledge, not retrieval — no tools, no context.
BREADTH_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The chemical symbol for gold is", ("au",)),
    ("The powerhouse of the cell is the", ("mitochondri",)),
    ("The speed of light in vacuum is approximately", ("3", "300,000", "299")),
    ("The author of 'On the Origin of Species' was", ("darwin",)),
    ("The capital of Japan is", ("tokyo",)),
    ("Water's chemical formula is", ("h2o",)),
    ("The largest planet in the solar system is", ("jupiter",)),
    ("The derivative of x squared is", ("2x",)),
    ("DNA stands for", ("deoxyribonucleic",)),
    ("The French Revolution began in the year", ("1789",)),
    ("In computing, CPU stands for", ("central processing",)),
    ("The square root of 144 is", ("12",)),
    ("The theory of general relativity was published by", ("einstein",)),
    ("The smallest prime number is", ("2",)),
    ("Photosynthesis converts carbon dioxide and water into", ("glucose", "sugar", "oxygen")),
    ("The longest river in Africa is the", ("nile",)),
    ("An algorithm's O(n log n) sorting example is", ("merge", "heap", "quick")),
    ("The currency of the United Kingdom is the", ("pound", "sterling")),
    ("Shakespeare wrote the tragedy of Prince Hamlet of", ("denmark",)),
    ("The boiling point of water at sea level in Celsius is", ("100",)),
    ("The human heart has this many chambers:", ("4", "four")),
    ("The most abundant gas in Earth's atmosphere is", ("nitrogen",)),
    ("In SQL, the command to retrieve rows is", ("select",)),
    ("The Pythagorean theorem relates the sides of a", ("right", "triangle")),
)


def _greedy_decode(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 10,
    cognitive_mode: str = "reactive",
    stop_strings: tuple[str, ...] = (),
) -> str:
    """Architecture-native deterministic decode for battery probes.

    Qwen3.8 mixes full-attention and linear-attention blocks. Hand-building a
    list of plain KVCache instances silently evaluates a different machine.
    ``generate_step`` asks the loaded model to construct its own cache and is
    the same generic primitive MLX-LM uses for normal generation.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    try:
        from core.brain.llm.chat_format import (
            render_chat_template,
            thinking_enabled_for_request,
        )

        rendered = render_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=thinking_enabled_for_request(
                None,
                cognitive_mode=cognitive_mode,
            ),
        )
        tokens = list(tokenizer.encode(rendered))
    except (AttributeError, TypeError, ValueError):
        tokens = list(tokenizer.encode(prompt))
    eos: set[int] = set()
    eid = getattr(tokenizer, "eos_token_id", None)
    if eid is not None:
        eos.add(int(eid))
    for extra in getattr(tokenizer, "eos_token_ids", None) or ():
        eos.add(int(extra))

    out: list[int] = []
    prompt_tokens = mx.array(tokens, dtype=mx.int32)
    for raw_token, _logprobs in generate_step(
        prompt_tokens,
        model,
        max_tokens=max(0, int(max_tokens)),
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        prefill_step_size=2048,
    ):
        token = int(raw_token)
        if token in eos:
            break
        out.append(token)
        if stop_strings:
            try:
                partial = str(tokenizer.decode(out))
            except (TypeError, ValueError, KeyError):
                partial = ""
            if _answer_matches(partial, stop_strings):
                break
    try:
        return str(tokenizer.decode(out))
    except (TypeError, ValueError, KeyError):
        return ""


def _answer_matches(answer: str, accepted: tuple[str, ...]) -> bool:
    """Match factual evidence independent of harmless display markup.

    Model generations commonly render formulas as ``H_2O`` or ``H_{2}O`` and
    names in Markdown emphasis. The old raw substring scorer counted those as
    knowledge failures. Compact alphanumeric comparison preserves the old
    containment contract while removing presentation-only punctuation.
    """
    compact_answer = re.sub(r"[^a-z0-9]+", "", str(answer).casefold())
    for option in accepted:
        compact_option = re.sub(r"[^a-z0-9]+", "", str(option).casefold())
        if compact_option and compact_option in compact_answer:
            return True
    return False


def _progress_rows_by_cell(events: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in events or ():
        if (
            isinstance(event, dict)
            and event.get("schema") == EVALUATION_PROGRESS_SCHEMA
            and isinstance(event.get("cell_id"), str)
            and isinstance(event.get("row"), dict)
        ):
            rows[event["cell_id"]] = dict(event["row"])
    return rows


def _emit_battery_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    label: str,
    phase: str,
    cell_id: str,
    completed: int,
    total: int,
    row: dict[str, Any],
) -> None:
    if callback is None:
        return
    callback(
        {
            "schema": EVALUATION_PROGRESS_SCHEMA,
            "label": label,
            "phase": phase,
            "cell_id": cell_id,
            "completed": completed,
            "total": total,
            "row": dict(row),
            "updated_at": time.time(),
        }
    )


def capability_battery(
    model,
    tokenizer,
    *,
    label: str = "model",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    resume_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic capability measurement: breadth + reasoning + identity.

    Breadth: factual cloze accuracy (compiled knowledge, closed book).
    Reasoning: verifiable micro-tasks from the falsification generators.
    Identity: the natural-probe behavior snapshot (for migration DELTAS —
    identity is compared across models, never pass/failed in isolation).
    """
    from core.brain.llm.latent_cortex.experiments import modular_chain, nested_boolean
    from core.learning.interference_battery import (
        natural_stability_probes,
        snapshot_probe_behavior,
    )

    started = time.monotonic()
    resumed = _progress_rows_by_cell(resume_events)
    breadth_hits = 0
    breadth_rows: list[dict[str, Any]] = []
    for index, (prompt, accepted) in enumerate(BREADTH_PROBES):
        cell_id = f"breadth:{index}"
        row = resumed.get(cell_id)
        if not (
            isinstance(row, dict)
            and row.get("prompt") == prompt
            and row.get("accepted") == list(accepted)
            and row.get("max_tokens") == _BREADTH_MAX_TOKENS
            and isinstance(row.get("answer"), str)
            and type(row.get("hit")) is bool
        ):
            answer = _greedy_decode(
                model,
                tokenizer,
                prompt,
                max_tokens=_BREADTH_MAX_TOKENS,
                cognitive_mode="reactive",
                stop_strings=accepted,
            ).lower()
            row = {
                "prompt": prompt,
                "accepted": list(accepted),
                "answer": answer[:1000],
                "hit": _answer_matches(answer, accepted),
                "max_tokens": _BREADTH_MAX_TOKENS,
            }
            _emit_battery_progress(
                progress_callback,
                label=label,
                phase="breadth",
                cell_id=cell_id,
                completed=index + 1,
                total=len(BREADTH_PROBES),
                row=row,
            )
        hit = bool(row["hit"])
        breadth_hits += int(hit)
        breadth_rows.append(dict(row))

    reasoning_hits = 0
    reasoning_rows: list[dict[str, Any]] = []
    reasoning_tasks = [
        task
        for seed in range(6)
        for task in (modular_chain(3, seed=seed), nested_boolean(3, seed=seed))
    ]
    reasoning_total = len(reasoning_tasks)
    for index, task in enumerate(reasoning_tasks):
        prompt_sha256 = hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        cell_id = f"reasoning:{task.family}:{task.depth}:{task.seed}"
        row = resumed.get(cell_id)
        if not (
            isinstance(row, dict)
            and row.get("family") == task.family
            and row.get("depth") == task.depth
            and row.get("seed") == task.seed
            and row.get("prompt_sha256") == prompt_sha256
            and row.get("max_tokens") == 256
            and isinstance(row.get("answer"), str)
            and type(row.get("hit")) is bool
        ):
            answer = _greedy_decode(
                model,
                tokenizer,
                task.prompt,
                max_tokens=256,
                cognitive_mode="deliberate",
            )
            row = {
                "family": task.family,
                "depth": task.depth,
                "seed": task.seed,
                "prompt_sha256": prompt_sha256,
                "answer": answer[:4000],
                "hit": bool(task.verify(answer)),
                "max_tokens": 256,
            }
            _emit_battery_progress(
                progress_callback,
                label=label,
                phase="reasoning",
                cell_id=cell_id,
                completed=index + 1,
                total=reasoning_total,
                row=row,
            )
        reasoning_hits += int(bool(row["hit"]))
        reasoning_rows.append(dict(row))

    try:
        identity_snapshot = snapshot_probe_behavior(
            model, natural_stability_probes(tokenizer)
        )
        identity_digests = [row["digest"] for row in identity_snapshot]
        for index, row in enumerate(identity_snapshot):
            _emit_battery_progress(
                progress_callback,
                label=label,
                phase="identity",
                cell_id=f"identity:{index}",
                completed=index + 1,
                total=len(identity_snapshot),
                row={"digest": row["digest"]},
            )
    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
        record_degradation(
            "cortex_upgrade",
            exc,
            action="recorded capability battery without an identity snapshot",
        )
        identity_digests = []

    return {
        "schema": EVALUATION_SCHEMA,
        "label": label,
        "breadth_accuracy": round(breadth_hits / len(BREADTH_PROBES), 4),
        "breadth_hits": breadth_hits,
        "breadth_total": len(BREADTH_PROBES),
        "breadth_rows": breadth_rows,
        "reasoning_accuracy": round(reasoning_hits / max(1, reasoning_total), 4),
        "reasoning_hits": reasoning_hits,
        "reasoning_total": reasoning_total,
        "reasoning_rows": reasoning_rows,
        "identity_digests": identity_digests,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def compare_batteries(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_descriptor: dict[str, Any] | None = None,
    critical_gates: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return a Pareto upgrade verdict across breadth and reasoning.

    A strict breadth-win rule makes replacement impossible once the incumbent
    reaches the finite breadth battery's ceiling.  Promotion evidence instead
    requires exact non-regression on both measured axes and a strict gain on at
    least one.  Deployment remains separately blocked by every critical gate.
    """
    breadth_delta = candidate["breadth_accuracy"] - current["breadth_accuracy"]
    reasoning_delta = candidate["reasoning_accuracy"] - current["reasoning_accuracy"]
    identity_changed = current.get("identity_digests") != candidate.get(
        "identity_digests"
    )
    no_regression = breadth_delta >= 0.0 and reasoning_delta >= 0.0
    strict_gain = breadth_delta > 0.0 or reasoning_delta > 0.0
    verdict = "PASS" if no_regression and strict_gain else "FAIL"
    gates = dict(critical_gates or {})
    all_critical_gates_pass = (
        set(gates) == _REQUIRED_CRITICAL_GATES
        and all(value is True for value in gates.values())
    )
    result = {
        "schema": EVALUATION_SCHEMA,
        "current_label": current["label"],
        "candidate_label": candidate["label"],
        "breadth_delta": round(breadth_delta, 4),
        "reasoning_delta": round(reasoning_delta, 4),
        "identity_behavior_changed": identity_changed,
        "identity_note": (
            "a new generation ALWAYS changes identity behavior — migration "
            "(persona retrain + steering re-extraction) is what restores it; "
            "this field feeds the migration plan, it does not gate the verdict"
        ),
        "candidate_descriptor_sha256": str(
            (candidate_descriptor or {}).get("descriptor_sha256") or ""
        ),
        "critical_gates": gates,
        "promotion_eligible": verdict == "PASS" and all_critical_gates_pass,
        "verdict": verdict,
        "compared_at": time.time(),
    }
    result["evaluation_sha256"] = _receipt_digest(result)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _receipt_digest(value: dict[str, Any], *, digest_key: str = "evaluation_sha256") -> str:
    material = dict(value)
    material.pop(digest_key, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


class MemoryGuard:
    """Refuses candidate loads the host cannot afford beside live processes."""

    def __init__(
        self,
        *,
        overhead_factor: float = _LOAD_OVERHEAD_FACTOR,
        free_margin_gb: float = _FREE_MARGIN_GB,
    ) -> None:
        self.overhead_factor = float(overhead_factor)
        self.free_margin_gb = float(free_margin_gb)

    @staticmethod
    def _weights_bytes(model_dir: Path) -> int:
        total = 0
        for pattern in ("*.safetensors", "*.npz", "*.gguf"):
            for file in Path(model_dir).glob(pattern):
                total += file.stat().st_size
        return total

    @staticmethod
    def _resident_giants_gb(threshold_gb: float = 6.0) -> list[dict[str, Any]]:
        """Python processes holding model-scale RSS (live app, training runs)."""
        try:
            from core.runtime.resource_observation import get_resource_observer
        except ImportError:
            return []
        giants = []
        for proc in get_resource_observer().processes():
            rss_gb = proc.rss_bytes / 1024**3
            if rss_gb >= threshold_gb:
                giants.append(
                    {
                        "pid": proc.pid,
                        "name": proc.name,
                        "rss_gb": round(rss_gb, 1),
                    }
                )
        return giants

    @staticmethod
    def _available_gb() -> float:
        try:
            from core.runtime.resource_observation import get_resource_observer

            return get_resource_observer().memory(include_process_tree=False).available_bytes / 1024**3
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return 0.0

    def admit(self, model_dir: Path | str) -> dict[str, Any]:
        model_dir = Path(model_dir)
        weights_gb = self._weights_bytes(model_dir) / 1024**3
        projected_gb = weights_gb * self.overhead_factor
        available_gb = self._available_gb()
        giants = self._resident_giants_gb()
        admitted = (
            weights_gb > 0
            and available_gb - projected_gb >= self.free_margin_gb
        )
        receipt = {
            "model_dir": str(model_dir),
            "weights_gb": round(weights_gb, 2),
            "projected_load_gb": round(projected_gb, 2),
            "available_gb": round(available_gb, 2),
            "free_margin_gb": self.free_margin_gb,
            "resident_giants": giants,
            "admitted": admitted,
        }
        if not admitted:
            reason = (
                "no weight files found"
                if weights_gb == 0
                else "insufficient memory headroom beside resident processes"
            )
            receipt["refusal_reason"] = reason
        return receipt


@dataclass
class MigrationStep:
    name: str
    artifact: str
    exists: bool
    lane: str  # automatic | operator_training_run | operator_review
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact": self.artifact,
            "exists": self.exists,
            "lane": self.lane,
            "detail": self.detail,
        }


def build_migration_plan(
    *,
    fused_model_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Enumerate what a generation swap must carry across, honestly laned.

    A new base is NOT Aura until her identity artifacts are rebuilt on it:
    the persona/CRSM delta must be RETRAINED (deltas are basis-specific),
    steering vectors RE-EXTRACTED, and expert adapters retrained or retired.
    Each step names its lane; nothing here pretends migration is a copy.
    """
    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    if data_dir is None:
        from core.config import DATA_DIR

        data_dir = Path(DATA_DIR)
    fused_model_dir = Path(fused_model_dir)
    data_dir = Path(data_dir)

    pointer = fused_model_dir / "active.json"
    steering = data_dir / "steering_vectors"
    adapters = data_dir / "expert_adapters"
    steps = [
        MigrationStep(
            name="activation_pointer",
            artifact=str(pointer),
            exists=pointer.is_file(),
            lane="automatic",
            detail="staged/activated/rolled back by this pipeline",
        ),
        MigrationStep(
            name="persona_crsm_delta",
            artifact=str(fused_model_dir),
            exists=fused_model_dir.is_dir(),
            lane="operator_training_run",
            detail=(
                "retrain the CRSM/persona delta against the NEW base "
                "(training/train_and_fuse.py --crsm-delta) and fuse a new "
                "artifact; low-rank deltas are basis-specific and never copy "
                "across generations"
            ),
        ),
        MigrationStep(
            name="caa_steering_vectors",
            artifact=str(steering),
            exists=steering.is_dir(),
            lane="operator_training_run",
            detail=(
                "re-extract CAA steering vectors from the new fused model; "
                "directions from the old activation basis do not transfer"
            ),
        ),
        MigrationStep(
            name="expert_adapters",
            artifact=str(adapters),
            exists=adapters.is_dir(),
            lane="operator_review",
            detail=(
                "retrain or retire domain expert LoRAs; each adapter's "
                "capture data can retrain against the new base through the "
                "existing compounding lanes"
            ),
        ),
        MigrationStep(
            name="recurrence_native_adapter",
            artifact="artifacts/closeout/latent_cortex/",
            exists=True,
            lane="operator_training_run",
            detail=(
                "rerun the recurrence-native curriculum on the new base so "
                "the RLC's trained recurrent mode carries into the next "
                "generation"
            ),
        ),
    ]
    return {
        "schema": MIGRATION_PLAN_SCHEMA,
        "steps": [step.to_dict() for step in steps],
        "automatic_steps": [s.name for s in steps if s.lane == "automatic"],
        "operator_steps": [s.name for s in steps if s.lane != "automatic"],
        "built_at": time.time(),
    }


def build_migration_contract(
    descriptor: dict[str, object],
    *,
    components: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Bind every model-facing Aura artifact to one exact representation basis.

    Equal layer counts and hidden widths are not evidence of a shared basis.
    Persona deltas, steering directions, expert adapters, and recurrent tissue
    are therefore rebuilt, explicitly retired, or refused before promotion.
    """

    validate_model_artifact_descriptor(descriptor)
    if not isinstance(components, dict) or set(components) != _REQUIRED_MIGRATION_COMPONENTS:
        raise ValueError("migration_components_incomplete")
    descriptor_sha256 = str(descriptor["descriptor_sha256"])
    normalized: dict[str, dict[str, object]] = {}
    for name in sorted(_REQUIRED_MIGRATION_COMPONENTS):
        raw = components.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"migration_component_invalid:{name}")
        component = validate_component_authority(
            raw,
            component=name,
            descriptor_sha256=descriptor_sha256,
        )
        normalized[name] = component

    material: dict[str, object] = {
        "schema": MIGRATION_CONTRACT_SCHEMA,
        "model_descriptor_sha256": descriptor_sha256,
        "components": normalized,
        "built_at": time.time(),
    }
    material["migration_contract_sha256"] = _receipt_digest(
        material,
        digest_key="migration_contract_sha256",
    )
    return material


def _validate_evaluation(
    evaluation: dict[str, object],
    *,
    descriptor_sha256: str,
) -> dict[str, object]:
    required = {
        "schema",
        "current_label",
        "candidate_label",
        "breadth_delta",
        "reasoning_delta",
        "identity_behavior_changed",
        "identity_note",
        "candidate_descriptor_sha256",
        "critical_gates",
        "promotion_eligible",
        "verdict",
        "compared_at",
        "evaluation_sha256",
    }
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != required
        or evaluation.get("schema") != EVALUATION_SCHEMA
    ):
        raise ValueError("evaluation_schema_invalid")
    if evaluation.get("candidate_descriptor_sha256") != descriptor_sha256:
        raise ValueError("evaluation_model_identity_mismatch")
    gates = evaluation.get("critical_gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != _REQUIRED_CRITICAL_GATES
        or any(value is not True for value in gates.values())
        or evaluation.get("verdict") != "PASS"
        or evaluation.get("promotion_eligible") is not True
    ):
        raise ValueError("evaluation_not_promotion_eligible")
    claimed = evaluation.get("evaluation_sha256")
    if not _valid_sha256(claimed) or claimed != _receipt_digest(evaluation):
        raise ValueError("evaluation_digest_invalid")
    return evaluation


def validate_upgrade_evaluation(
    evaluation: dict[str, object],
    *,
    descriptor_sha256: str,
) -> dict[str, object]:
    """Public exact-receipt validator used by the central cortex registry."""
    return _validate_evaluation(
        evaluation,
        descriptor_sha256=descriptor_sha256,
    )


def _validate_migration_contract(
    contract: dict[str, object],
    descriptor: dict[str, object],
) -> dict[str, object]:
    descriptor_sha256 = str(descriptor.get("descriptor_sha256") or "")
    if not isinstance(contract, dict):
        raise ValueError("migration_contract_schema_invalid")
    components = contract.get("components")
    if not isinstance(components, dict) or set(components) != _REQUIRED_MIGRATION_COMPONENTS:
        raise ValueError("migration_components_incomplete")
    required = {
        "schema",
        "model_descriptor_sha256",
        "components",
        "built_at",
        "migration_contract_sha256",
    }
    if set(contract) != required or contract.get("schema") != MIGRATION_CONTRACT_SCHEMA:
        raise ValueError("migration_contract_schema_invalid")
    if contract.get("model_descriptor_sha256") != descriptor_sha256:
        raise ValueError("migration_contract_model_identity_mismatch")
    claimed = contract.get("migration_contract_sha256")
    if not _valid_sha256(claimed) or claimed != _receipt_digest(
        contract,
        digest_key="migration_contract_sha256",
    ):
        raise ValueError("migration_contract_digest_invalid")

    for name, raw in components.items():
        if not isinstance(raw, dict):
            raise ValueError(f"migration_component_invalid:{name}")
        validate_component_authority(
            raw,
            component=name,
            descriptor_sha256=descriptor_sha256,
        )
    return contract


def validate_migration_contract(
    contract: dict[str, object],
    descriptor: dict[str, object],
) -> dict[str, object]:
    """Public exact-basis validator used by the central cortex registry."""
    return _validate_migration_contract(contract, descriptor)


def _read_pointer(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "active_model_path" not in payload:
        raise ValueError(f"activation pointer at {path} is not schema v2")
    return payload


def _governed_write(path: Path, payload: bytes, *, source: str) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    with local_internal_governed_scope("cortex_generation_upgrade"):
        gateway.ensure_directory(path.parent, source=source)
        gateway.write_bytes(path, payload, source=source)


def normalize_active_pointer_identity(
    *,
    artifact_descriptor: dict[str, Any],
    fused_model_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Upgrade the current pointer to exact schema 3 without changing its model.

    This is an identity migration, not a cortex promotion.  The original bytes
    are retained once, the descriptor is fully re-hashed against the already
    active artifact, and every existing pointer field is preserved.
    """
    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    pointer_path = fused_model_dir / "active.json"
    current_bytes = pointer_path.read_bytes()
    current = _read_pointer(pointer_path)
    active = Path(str(current["active_model_path"])).expanduser().resolve(strict=True)
    validate_model_artifact_descriptor(
        artifact_descriptor,
        model_path=active,
        verify_full_hash=True,
    )

    backup_path = fused_model_dir / IDENTITY_BACKUP_POINTER_NAME
    if backup_path.exists():
        predecessor_bytes = backup_path.read_bytes()
        predecessor = json.loads(predecessor_bytes)
        if not isinstance(predecessor, dict):
            raise ValueError("active_pointer_identity_backup_invalid")
        stripped_current = dict(current)
        stripped_current.pop("artifact_descriptor", None)
        stripped_current.pop("identity_transition", None)
        stripped_current["schema_version"] = predecessor.get("schema_version")
        if stripped_current != predecessor:
            raise ValueError("active_pointer_identity_transition_not_narrow")
    else:
        predecessor_bytes = current_bytes
        predecessor = dict(current)

    predecessor_raw_path = str(predecessor.get("active_model_path") or "").strip()
    if not predecessor_raw_path:
        raise ValueError("active_pointer_identity_backup_model_missing")
    predecessor_active = Path(predecessor_raw_path).expanduser().resolve(strict=True)
    if predecessor_active != active:
        raise ValueError("active_pointer_identity_backup_model_mismatch")

    predecessor_sha256 = hashlib.sha256(predecessor_bytes).hexdigest()
    transition: dict[str, object] = {
        "schema": IDENTITY_TRANSITION_SCHEMA,
        "kind": "model_identity_normalization",
        "previous_pointer_sha256": predecessor_sha256,
        "active_model_path": str(active),
        "model_descriptor_sha256": artifact_descriptor["descriptor_sha256"],
    }
    transition["transition_sha256"] = _receipt_digest(
        transition,
        digest_key="transition_sha256",
    )

    existing = current.get("artifact_descriptor")
    if existing is not None and existing != artifact_descriptor:
        raise ValueError("active_pointer_descriptor_conflict")
    existing_transition = current.get("identity_transition")
    if existing_transition is not None and existing_transition != transition:
        raise ValueError("active_pointer_identity_transition_conflict")

    normalized = dict(predecessor)
    normalized["schema_version"] = 3
    normalized["artifact_descriptor"] = artifact_descriptor
    normalized["identity_transition"] = transition
    normalized_bytes = (
        json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    changed = normalized_bytes != current_bytes
    if changed:
        if not backup_path.exists():
            _governed_write(
                backup_path,
                predecessor_bytes,
                source="cortex_upgrade.normalize_identity_backup",
            )
        _governed_write(
            pointer_path,
            normalized_bytes,
            source="cortex_upgrade.normalize_identity",
        )

    return {
        "schema": IDENTITY_NORMALIZATION_SCHEMA,
        "active_model_path": str(active),
        "model_descriptor_sha256": artifact_descriptor["descriptor_sha256"],
        "predecessor_pointer_sha256": predecessor_sha256,
        "identity_transition_sha256": transition["transition_sha256"],
        "before_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "after_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "backup_path": str(backup_path),
        "changed": changed,
        "normalized_at": time.time(),
    }


def stage_upgrade(
    *,
    candidate_model_path: Path | str,
    base_model_path: Path | str,
    tag: str,
    fused_model_dir: Path | str | None = None,
    evaluation: dict[str, Any],
    serving_profile: dict[str, Any],
    migration_contract: dict[str, Any],
    artifact_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an identity-bound staged pointer and byte-exact rollback."""
    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    candidate = Path(candidate_model_path).expanduser()
    if not candidate.is_dir():
        raise ValueError(f"candidate model directory missing: {candidate}")
    candidate = candidate.resolve(strict=True)

    descriptor = artifact_descriptor or build_model_artifact_descriptor(candidate)
    validate_model_artifact_descriptor(
        descriptor,
        model_path=candidate,
        verify_full_hash=True,
    )
    _validate_evaluation(
        evaluation,
        descriptor_sha256=str(descriptor["descriptor_sha256"]),
    )
    validate_model_serving_profile(serving_profile, descriptor)
    _validate_migration_contract(migration_contract, descriptor)

    pointer_path = fused_model_dir / "active.json"
    current_bytes = pointer_path.read_bytes()
    current = _read_pointer(pointer_path)

    staged_payload = {
        "active_model_path": str(candidate),
        "base_model": str(base_model_path),
        "fused_at": int(time.time()),
        "schema_version": 3,
        "size": current.get("size", "32B"),
        "tag": str(tag),
        "artifact_descriptor": descriptor,
        "evaluation": evaluation,
        "serving_profile": serving_profile,
        "migration_contract": migration_contract,
    }
    staged_bytes = (
        json.dumps(staged_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _governed_write(
        fused_model_dir / ROLLBACK_POINTER_NAME,
        current_bytes,
        source="cortex_upgrade.stage",
    )
    _governed_write(
        fused_model_dir / STAGED_POINTER_NAME,
        staged_bytes,
        source="cortex_upgrade.stage",
    )
    receipt = {
        "schema": STAGING_SCHEMA,
        "staged_pointer": str(fused_model_dir / STAGED_POINTER_NAME),
        "rollback_pointer": str(fused_model_dir / ROLLBACK_POINTER_NAME),
        "current_active_model": current["active_model_path"],
        "staged_active_model": str(candidate),
        "staged_sha256": hashlib.sha256(staged_bytes).hexdigest(),
        "rollback_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "serving_profile_sha256": serving_profile["profile_sha256"],
        "migration_contract_sha256": migration_contract[
            "migration_contract_sha256"
        ],
        "evaluation_verdict": evaluation.get("verdict"),
        "staged_at": time.time(),
    }
    logger.info("🧬 Cortex upgrade STAGED: %s → %s", current["active_model_path"], candidate)
    return receipt


def activate_upgrade(
    *,
    fused_model_dir: Path | str | None = None,
    authorized_by: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Flip the activation pointer to the staged target. Boot-time effect only.

    Hard gates, no overrides: a real operator authorization string and a
    PASS comparison verdict. The running mind is never hot-swapped — the
    new cortex exists only after the operator restarts.
    """
    if not isinstance(authorized_by, str) or len(authorized_by.strip()) < 3:
        raise PermissionError(
            "cortex activation requires an explicit operator authorization string"
        )
    if not isinstance(evaluation, dict) or evaluation.get("verdict") != "PASS":
        raise PermissionError(
            "cortex activation requires a PASS capability-comparison verdict"
        )
    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    staged_path = fused_model_dir / STAGED_POINTER_NAME
    rollback_path = fused_model_dir / ROLLBACK_POINTER_NAME
    if not staged_path.is_file() or not rollback_path.is_file():
        raise ValueError("nothing staged: run stage_upgrade first")
    staged_bytes = staged_path.read_bytes()
    staged = _read_pointer(staged_path)
    if staged.get("schema_version") != 3:
        raise ValueError("staged_pointer_contract_missing")
    descriptor = staged.get("artifact_descriptor")
    serving_profile = staged.get("serving_profile")
    migration_contract = staged.get("migration_contract")
    staged_evaluation = staged.get("evaluation")
    if not all(
        isinstance(value, dict)
        for value in (descriptor, serving_profile, migration_contract, staged_evaluation)
    ):
        raise ValueError("staged_pointer_contract_missing")
    assert isinstance(descriptor, dict)
    assert isinstance(serving_profile, dict)
    assert isinstance(migration_contract, dict)
    assert isinstance(staged_evaluation, dict)
    descriptor_sha256 = str(descriptor.get("descriptor_sha256") or "")
    try:
        _validate_evaluation(evaluation, descriptor_sha256=descriptor_sha256)
    except ValueError as exc:
        raise PermissionError("activation requires the exact staged evaluation") from exc
    if evaluation.get("evaluation_sha256") != staged_evaluation.get(
        "evaluation_sha256"
    ):
        raise PermissionError("activation requires the exact staged evaluation")
    _validate_evaluation(staged_evaluation, descriptor_sha256=descriptor_sha256)

    candidate = Path(str(staged["active_model_path"])).expanduser()
    validate_model_artifact_descriptor(
        descriptor,
        model_path=candidate,
        verify_full_hash=True,
    )
    validate_model_serving_profile(serving_profile, descriptor)
    _validate_migration_contract(migration_contract, descriptor)
    _governed_write(
        fused_model_dir / "active.json", staged_bytes, source="cortex_upgrade.activate"
    )
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "activated_model": staged["active_model_path"],
        "active_sha256": hashlib.sha256(staged_bytes).hexdigest(),
        "model_descriptor_sha256": descriptor_sha256,
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "serving_profile_sha256": serving_profile["profile_sha256"],
        "migration_contract_sha256": migration_contract[
            "migration_contract_sha256"
        ],
        "authorized_by": authorized_by.strip(),
        "evaluation_verdict": "PASS",
        "effective": "next_boot",
        "activated_at": time.time(),
    }
    logger.info(
        "🧬 Cortex upgrade ACTIVATED by %s (effective at next boot)",
        authorized_by.strip(),
    )
    return receipt


def rollback_upgrade(*, fused_model_dir: Path | str | None = None) -> dict[str, Any]:
    """Restore the rollback pointer byte-exactly, verified by digest."""
    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    rollback_path = fused_model_dir / ROLLBACK_POINTER_NAME
    if not rollback_path.is_file():
        raise ValueError("no rollback pointer exists")
    rollback_bytes = rollback_path.read_bytes()
    _read_pointer(rollback_path)
    _governed_write(
        fused_model_dir / "active.json", rollback_bytes, source="cortex_upgrade.rollback"
    )
    restored = (fused_model_dir / "active.json").read_bytes()
    exact = restored == rollback_bytes
    if not exact:
        record_degradation(
            "cortex_upgrade",
            RuntimeError("rollback restore was not byte-exact"),
            action="flagged rollback receipt; operator must inspect the pointer",
            severity="critical",
        )
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "restored_model": _read_pointer(rollback_path)["active_model_path"],
        "byte_exact": exact,
        "restored_sha256": hashlib.sha256(restored).hexdigest(),
        "rolled_back_at": time.time(),
    }
    logger.info("🧬 Cortex upgrade ROLLED BACK (byte_exact=%s)", exact)
    return receipt


__all__ = [
    "ACTIVATION_SCHEMA",
    "BREADTH_PROBES",
    "EVALUATION_SCHEMA",
    "EVALUATION_PROGRESS_SCHEMA",
    "IDENTITY_BACKUP_POINTER_NAME",
    "IDENTITY_NORMALIZATION_SCHEMA",
    "IDENTITY_TRANSITION_SCHEMA",
    "MIGRATION_CONTRACT_SCHEMA",
    "MIGRATION_PLAN_SCHEMA",
    "MemoryGuard",
    "MigrationStep",
    "ROLLBACK_POINTER_NAME",
    "STAGED_POINTER_NAME",
    "STAGING_SCHEMA",
    "activate_upgrade",
    "build_migration_contract",
    "build_migration_plan",
    "capability_battery",
    "compare_batteries",
    "normalize_active_pointer_identity",
    "rollback_upgrade",
    "stage_upgrade",
    "validate_migration_contract",
    "validate_upgrade_evaluation",
]

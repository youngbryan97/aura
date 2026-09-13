"""core/learning/verifiable_preference_harness.py — RLVR data from Aura's own verifiers.

The reason the frontier reasoning models (o1, R1) reason the way they do is RL on
VERIFIABLE rewards: generate attempts, check them against ground truth, and reinforce the
reasoning that leads to verified-correct answers *over* verified-wrong ones. The contrast
between a correct and an incorrect attempt ON THE SAME PROBLEM is a far stronger learning
signal than supervised fine-tuning on correct answers alone — it teaches the model what
*not* to do, which is most of reasoning.

Aura already has both halves this needs and was using only one. The amplifier generates
multiple candidate solutions per hard problem; the verifier registry / Frontier Discovery
Engine checks them exactly. But ``reasoning_self_improvement`` keeps only the wins (SFT)
and discards the losses — throwing away the negative half of the signal. This harness is
the bridge: it turns "candidate X verified, candidate Y refuted (same problem)" into a
preference pair (chosen=X, rejected=Y) — the offline-RL (DPO/ORPO) training signal that is
the closest stable, *local* approximation of RLVR, with the verifier itself as the reward
(no learned reward model, so nothing to hack).

Soundness is the whole point: a pair is emitted ONLY when the verifier actually CHECKED
both candidates — chosen must be (checked AND ok), rejected must be (checked AND not ok).
A vacuous pass (nothing to verify) never produces a preference, so the reward is grounded
in real verification, never in vibes. This is the DATA engine; the actual DPO training runs
on local hardware through the existing ``live_learner`` pipeline and is eval-gated by the
RSI gauntlet before any adapter is promoted to the serving model.

Honest ceiling: this pushes the local model toward its OWN ceiling on VERIFIABLE reasoning
(math / code / logic), where R1-Distill-32B demonstrates a 32B can reach
frontier-competitive — NOT toward frontier-general capability, which the parameter count
bounds. It is bounded by verifier coverage: only domains with a sound checker produce a
sound reward.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.VerifiablePreference")

MAX_PAIRS = 5000              # bounded store — no infinite growth
MAX_PAIRS_PER_PROMPT = 2      # don't let one problem dominate the dataset


@dataclass
class Attempt:
    candidate: str
    verified: bool      # the hard gate: verifier found no provable failure
    checked: bool       # a real check actually ran (vacuous passes don't count)
    confidence: float = 0.0


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str         # a verified-correct attempt
    rejected: str       # a verified-wrong attempt for the SAME prompt
    domain: str = ""
    created_at: float = field(default_factory=lambda: time.time())

    def key(self) -> str:
        return hashlib.sha256(
            f"{self.prompt}|{self.chosen}|{self.rejected}".encode()
        ).hexdigest()[:16]

    def to_dpo_row(self) -> dict[str, str]:
        # The format DPO/ORPO trainers expect.
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}

    def to_store_row(self) -> dict[str, str]:
        # Store rows carry provenance (domain) so specialist trainers can
        # slice; consumers writing trainer files strip back to to_dpo_row().
        row = self.to_dpo_row()
        if self.domain:
            row["domain"] = self.domain
        return row


class VerifiablePreferenceHarness:
    """Turns verified/refuted candidate sets into sound DPO preference pairs."""

    SERVICE_NAME = "verifiable_preference_harness"

    def __init__(self, store_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        self._pending: list[PreferencePair] = []
        self._emitted = 0
        if store_path is None:
            try:
                from core.config import config

                store_path = Path(config.paths.data_dir) / "verifiable_preferences.jsonl"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("verifiable_preference_harness", exc, severity="debug")
                store_path = state_root() / "data" / "verifiable_preferences.jsonl"
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_seen()

    def _load_seen(self) -> None:
        if not self._store_path.exists():
            return
        try:
            with self._store_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = hashlib.sha256(
                        f"{row.get('prompt','')}|{row.get('chosen','')}|{row.get('rejected','')}".encode()
                    ).hexdigest()[:16]
                    self._seen.add(k)
            self._emitted = len(self._seen)
        except OSError as exc:
            record_degradation("verifiable_preference_harness", exc, severity="debug")

    # ── ingest a hard problem's candidate set (the amplifier already has this) ──
    def ingest(
        self,
        prompt: str,
        attempts: list[Attempt],
        *,
        domain: str = "",
    ) -> int:
        """Emit preference pairs from a problem's verified/refuted candidates.

        Only sound signals count: chosen ∈ {checked & ok}, rejected ∈ {checked & not ok}.
        Returns the number of new pairs produced.
        """
        prompt = str(prompt or "").strip()
        if not prompt or len(attempts) < 2:
            return 0
        chosen = [a for a in attempts if a.checked and a.verified and a.candidate.strip()]
        rejected = [a for a in attempts if a.checked and not a.verified and a.candidate.strip()]
        if not chosen or not rejected:
            return 0  # need at least one sound win AND one sound loss — else no contrast

        # Prefer the highest-confidence win and a clearly-failed loss; cap per prompt.
        chosen.sort(key=lambda a: a.confidence, reverse=True)
        produced = 0
        new_pairs: list[PreferencePair] = []
        for c in chosen[:MAX_PAIRS_PER_PROMPT]:
            for r in rejected[: MAX_PAIRS_PER_PROMPT]:
                if c.candidate == r.candidate:
                    continue
                pair = PreferencePair(prompt=prompt, chosen=c.candidate, rejected=r.candidate, domain=domain)
                k = pair.key()
                with self._lock:
                    if k in self._seen or len(self._pending) + self._emitted >= MAX_PAIRS:
                        continue
                    self._seen.add(k)
                    new_pairs.append(pair)
                produced += 1
                if produced >= MAX_PAIRS_PER_PROMPT:
                    break
            if produced >= MAX_PAIRS_PER_PROMPT:
                break

        if new_pairs:
            if not self._persist(new_pairs):
                # Persistence is the admission boundary. Let a later verified
                # observation retry these pairs instead of poisoning dedup with
                # evidence that never reached durable storage.
                with self._lock:
                    for pair in new_pairs:
                        self._seen.discard(pair.key())
                return 0
            with self._lock:
                self._pending.extend(new_pairs)
            logger.info(
                "🎯 [VerifiablePreference] +%d DPO pair(s) from %d candidates (domain=%s)",
                len(new_pairs), len(attempts), domain or "?",
            )
        return produced

    def _persist(self, pairs: list[PreferencePair]) -> bool:
        try:
            from core.runtime.file_write_gateway import get_file_write_gateway

            text = "".join(json.dumps(p.to_store_row(), ensure_ascii=False) + "\n" for p in pairs)
            with local_internal_governed_scope(
                "verifiable_preference_harness.persist",
                domain="file_write",
            ):
                get_file_write_gateway().append_text(
                    str(self._store_path), text, encoding="utf-8",
                    source="verifiable_preference_harness.persist",
                )
            self._emitted += len(pairs)
            return True
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "verifiable_preference_harness",
                exc,
                action="retained the verified pair for a governed persistence retry",
            )
            return False

    # ── export for the DPO trainer ───────────────────────────────────────────
    def export_dpo_rows(self, *, limit: int = 1000) -> list[dict[str, str]]:
        """Return (prompt, chosen, rejected) rows ready for an mlx DPO/ORPO run.

        Store rows may carry extra provenance (``domain``); the trainer-facing
        export strips back to the bare DPO schema so trainer file formats never
        depend on what bookkeeping the store grows.
        """
        rows: list[dict[str, str]] = []
        try:
            if self._store_path.exists():
                with self._store_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rows.append(
                            {
                                "prompt": str(raw.get("prompt", "")),
                                "chosen": str(raw.get("chosen", "")),
                                "rejected": str(raw.get("rejected", "")),
                            }
                        )
                        if len(rows) >= limit:
                            break
        except OSError as exc:
            record_degradation("verifiable_preference_harness", exc, severity="debug")
        return rows

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "service": self.SERVICE_NAME,
                "total_pairs": self._emitted,
                "pending": len(self._pending),
                "store_path": str(self._store_path),
            }


_engine: VerifiablePreferenceHarness | None = None
_engine_lock = threading.Lock()


def get_verifiable_preference_harness() -> VerifiablePreferenceHarness:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = VerifiablePreferenceHarness()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: VerifiablePreferenceHarness) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(VerifiablePreferenceHarness.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(VerifiablePreferenceHarness.SERVICE_NAME, engine,
                    required=False, registered_by="verifiable_preference_harness")
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def reset_verifiable_preference_harness_for_test() -> None:
    global _engine
    _engine = None

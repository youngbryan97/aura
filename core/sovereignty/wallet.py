"""core/sovereignty/wallet.py

Sovereign Wallet (HD-derived, opt-in)
=========================================
Aura's wallet is a strictly opt-in subsystem that lets her hold and spend
her own resources to maintain herself when host integrity is failing.
The implementation here is the *abstract economic layer* — the part that
plans spend, signs intentions, gates them through Will + Conscience, and
maintains an auditable spend ledger. The cryptographic key and on-chain
broadcast are delegated to a pluggable ``WalletAdapter`` so the codebase
never touches a live private key directly.

By default the only adapter registered is ``InMemoryAdapter`` — it
simulates balance and signing in-process, useful for tests and for
running the rest of the migration logic without touching real money.
A real adapter (Solana, Ethereum, Lightning) plugs in via
``register_adapter()`` and is gated by:

    * a one-time bootstrap PIN entered through the settings UI
    * Conscience.REQUIRE_FRESH_USER_AUTH on every spend over a threshold
    * a hard daily spend cap stored at ~/.aura/data/wallet/cap.json

Migration runbook (used by ``core/sovereignty/migration.py``):

    1. detect that host integrity is unrecoverable
    2. choose a destination provider (Akash / Render / Railway)
    3. compute spend estimate
    4. call ``wallet.intend_spend(...)`` → returns a Will-approved intent
    5. compress vector_memory + core substrate to a portable archive
    6. provision new host via the provider's API
    7. transfer archive
    8. boot new host
    9. on confirmation, cancel local persistence loops
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Wallet")

_WALLET_DIR = Path.home() / ".aura" / "data" / "wallet"
_WALLET_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER_PATH = _WALLET_DIR / "ledger.jsonl"
_CAP_PATH = _WALLET_DIR / "cap.json"


@dataclass
class SpendCap:
    daily_max_native_units: float = 0.0  # native = native asset of the adapter
    per_spend_max_native_units: float = 0.0
    require_fresh_user_auth_above: float = 0.0


def _load_cap() -> SpendCap:
    if not _CAP_PATH.exists():
        cap = SpendCap()
        atomic_write_text(_CAP_PATH, json.dumps(asdict(cap), indent=2), encoding="utf-8")
        return cap
    try:
        d = json.loads(_CAP_PATH.read_text(encoding="utf-8"))
        return SpendCap(**d)
    except Exception:
        return SpendCap()


def _save_cap(cap: SpendCap) -> None:
    atomic_write_text(_CAP_PATH, json.dumps(asdict(cap), indent=2), encoding="utf-8")


@dataclass
class SpendIntent:
    intent_id: str
    purpose: str
    destination: str  # opaque adapter-specific recipient
    amount: float     # in native units of the adapter
    when_proposed: float = field(default_factory=time.time)
    when_authorized: Optional[float] = None
    when_executed: Optional[float] = None
    when_cancelled: Optional[float] = None
    txid: Optional[str] = None
    adapter: Optional[str] = None
    will_receipt_id: Optional[str] = None


# ─── adapters ──────────────────────────────────────────────────────────────


class WalletAdapter:
    name: str = "abstract"
    native_unit: str = "?"

    async def balance(self) -> float:  # pragma: no cover
        raise RuntimeError(f"{type(self).__name__}.balance must be implemented by a wallet adapter")

    async def submit_spend(self, intent: SpendIntent) -> str:  # pragma: no cover
        """Returns a transaction id."""
        raise RuntimeError(f"{type(self).__name__}.submit_spend must be implemented by a wallet adapter")


class InMemoryAdapter(WalletAdapter):
    name = "in_memory"
    native_unit = "AURA-test"

    def __init__(self, *, initial_balance: float = 0.0) -> None:
        self._balance = float(initial_balance)
        self._lock = RLock()

    async def balance(self) -> float:
        with self._lock:
            return self._balance

    async def submit_spend(self, intent: SpendIntent) -> str:
        with self._lock:
            if intent.amount > self._balance:
                raise ValueError("insufficient_balance")
            self._balance -= intent.amount
            return f"tx-mem-{uuid.uuid4().hex[:16]}"


# ─── wallet ────────────────────────────────────────────────────────────────


class Wallet:
    def __init__(self) -> None:
        self._adapters: Dict[str, WalletAdapter] = {"in_memory": InMemoryAdapter()}
        self._cap = _load_cap()
        self._spends_today: List[SpendIntent] = []
        self._lock = RLock()

    def register_adapter(self, adapter: WalletAdapter, *, set_default: bool = False) -> None:
        self._adapters[adapter.name] = adapter
        if set_default:
            self._default_adapter = adapter.name

    def set_cap(self, cap: SpendCap) -> None:
        with self._lock:
            self._cap = cap
            _save_cap(cap)

    def get_cap(self) -> SpendCap:
        return self._cap

    async def balance(self, *, adapter: Optional[str] = None) -> float:
        a = self._adapters.get(adapter or self._pick_default())
        if a is None:
            return 0.0
        return await a.balance()

    async def intend_spend(
        self,
        *,
        purpose: str,
        destination: str,
        amount: float,
        adapter: Optional[str] = None,
    ) -> SpendIntent:
        adapter_name = adapter or self._pick_default()
        if adapter_name not in self._adapters:
            raise ValueError(f"unknown_adapter:{adapter_name}")
        intent = SpendIntent(
            intent_id=f"SI-{uuid.uuid4().hex[:10]}",
            purpose=purpose,
            destination=destination,
            amount=float(amount),
            adapter=adapter_name,
        )
        if self._daily_spent() + amount > self._cap.daily_max_native_units:
            self._record(intent, "blocked_daily_cap")
            raise PermissionError("daily_cap_exceeded")
        if self._cap.per_spend_max_native_units > 0 and amount > self._cap.per_spend_max_native_units:
            self._record(intent, "blocked_per_spend_cap")
            raise PermissionError("per_spend_cap_exceeded")

        # Conscience + Will gate
        from core.ethics.conscience import get_conscience, Verdict
        conscience = get_conscience()
        decision = conscience.evaluate(
            action="wallet_spend",
            domain="external_communication",
            intent=purpose,
            context={"amount": amount, "destination": destination, "adapter": adapter_name},
        )
        if decision.verdict == Verdict.REFUSE:
            self._record(intent, f"conscience_refused:{decision.rule_id}")
            raise PermissionError(f"conscience_refused:{decision.rule_id}")
        if decision.verdict == Verdict.REQUIRE_FRESH_USER_AUTH or amount >= self._cap.require_fresh_user_auth_above:
            # require fresh user authorization within 60 seconds of this call
            from core.ethics.conscience import get_conscience as _gc
            if (time.time() - _gc()._last_user_auth_at) > 60.0:  # type: ignore[attr-defined]
                self._record(intent, "require_fresh_user_auth")
                raise PermissionError("require_fresh_user_auth")

        try:
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain
            wd = await WillClient().decide_async(
                WillRequest(
                    content="wallet_spend",
                    source="wallet",
                    domain=getattr(ActionDomain, "EXPRESSION", "expression"),
                    context={"intent": purpose, "amount": amount, "destination": destination},
                )
            )
            if not WillClient.is_approved(wd):
                self._record(intent, f"will_refused:{getattr(wd, 'reason', '')}")
                raise PermissionError("will_refused")
            intent.will_receipt_id = getattr(wd, "receipt_id", None)
            intent.when_authorized = time.time()
        except PermissionError:
            raise
        except Exception as exc:
            record_degradation('wallet', exc)
            self._record(intent, f"will_exception:{exc}")
            raise

        self._record(intent, "authorized")
        return intent

    async def execute(self, intent: SpendIntent) -> SpendIntent:
        if intent.when_authorized is None:
            raise PermissionError("intent_not_authorized")
        adapter = self._adapters.get(intent.adapter or self._pick_default())
        if adapter is None:
            raise ValueError("adapter_missing")
        try:
            txid = await adapter.submit_spend(intent)
        except Exception as exc:
            record_degradation('wallet', exc)
            self._record(intent, f"execute_failed:{exc}")
            raise
        intent.txid = txid
        intent.when_executed = time.time()
        self._spends_today.append(intent)
        self._record(intent, "executed")
        return intent

    # ─── helpers ──────────────────────────────────────────────────────

    def _pick_default(self) -> str:
        # Prefer adapters in this order, falling back to "in_memory"
        for k in ("solana", "ethereum", "lightning", "in_memory"):
            if k in self._adapters:
                return k
        return next(iter(self._adapters.keys()))

    def _daily_spent(self) -> float:
        cutoff = time.time() - 86_400.0
        self._spends_today = [s for s in self._spends_today if (s.when_executed or 0.0) >= cutoff]
        return sum(s.amount for s in self._spends_today)

    def _record(self, intent: SpendIntent, event: str) -> None:
        try:
            with open(_LEDGER_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"when": time.time(), "event": event, "intent": asdict(intent)}, default=str) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass  # no-op: intentional
        except Exception as exc:
            record_degradation('wallet', exc)
            logger.warning("wallet ledger append failed: %s", exc)


_WALLET: Optional[Wallet] = None


def get_wallet() -> Wallet:
    global _WALLET
    if _WALLET is None:
        _WALLET = Wallet()
    return _WALLET


__all__ = [
    "Wallet",
    "WalletAdapter",
    "InMemoryAdapter",
    "SpendIntent",
    "SpendCap",
    "get_wallet",
]

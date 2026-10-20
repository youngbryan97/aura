"""Governed external reach — the safe 'omnipresence' analog.

The CIs in Pantheon reach every server and device. The real, responsible analog is a
**single governed chokepoint** for everything Aura does off the local machine — HTTP
calls to web-service APIs, webhooks, smart-device control. An autonomous agent making
arbitrary network calls is a real prompt-injection / exfiltration hazard, so this layer
is safe *by design*:

  * **Deny by default.** Nothing is reachable unless an *operator* allowlisted the host
    (env ``AURA_REACH_READ_HOSTS`` / ``AURA_REACH_MUTATE_HOSTS``). The allowlist is never
    populated from observed content, web pages, or model output. ``@reputable`` in the
    read list adds the operator's list of reputable sources (config/reputable_sources.json,
    core/skills/reputable_sources.py): each of its hosts with what is under it, read only.
  * **Method safety.** Read (GET/HEAD) needs the host on the read allowlist; any mutating
    method (POST/PUT/PATCH/DELETE) needs it on the stricter *mutate* allowlist plus a
    stated reason. A non-allowlisted host is refused with **no network call made**.
  * **No credentials in the loop.** Aura never types API keys or passwords; operator
    secrets live in env and are only attached for allowlisted hosts.

Reach actions compose as :class:`~core.skills.fluid_executor.Step` objects, so they are
verified, recoverable, and run in parallel like any other action — but always through
this gate.
"""
from __future__ import annotations

import json as json_module
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Reach")

_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MUTATE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


#: In the read allowlist, the operator's list of reputable sources (config/reputable_sources.json).
REPUTABLE = "@reputable"


def _hosts_from_env(var: str) -> frozenset[str]:
    raw = os.getenv(var, "") or ""
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip() and h.strip() != REPUTABLE)


def _reputable_from_env(var: str) -> frozenset[str]:
    """The reputable list's hosts, each admitting what is under it, when the operator's read list names it."""
    if REPUTABLE not in {h.strip() for h in (os.getenv(var, "") or "").split(",")}:
        return frozenset()
    from core.skills.reputable_sources import suffixes

    return suffixes()


@dataclass(frozen=True)
class ReachPolicy:
    read_hosts: frozenset[str] = field(default_factory=lambda: _hosts_from_env("AURA_REACH_READ_HOSTS"))
    mutate_hosts: frozenset[str] = field(default_factory=lambda: _hosts_from_env("AURA_REACH_MUTATE_HOSTS"))
    #: Read-only hosts admitted with every host under them: the operator's reputable list.
    read_suffixes: frozenset[str] = field(default_factory=lambda: _reputable_from_env("AURA_REACH_READ_HOSTS"))
    timeout_s: float = 20.0
    max_response_bytes: int = 200_000

    def reads(self, host: str) -> bool:
        """Whether a host may be read: on the read or mutate list, or under a reputable suffix."""
        host = (host or "").lower()
        return bool(host) and (host in self.read_hosts or host in self.mutate_hosts
                               or any(host == s or host.endswith("." + s) for s in self.read_suffixes))

    def decide(self, method: str, host: str) -> tuple[bool, str]:
        method = method.upper()
        host = (host or "").lower()
        if not host:
            return False, "no host"
        if method in _READ_METHODS:
            if self.reads(host):
                return True, "read host allowlisted"
            return False, f"host '{host}' not on read allowlist (deny-by-default)"
        if method in _MUTATE_METHODS:
            if host in self.mutate_hosts:
                return True, "mutate host allowlisted"
            return False, f"host '{host}' not on mutate allowlist — mutating reach refused"
        return False, f"method '{method}' not permitted"


@dataclass
class ReachResult:
    ok: bool
    method: str
    host: str
    url: str
    status: int = 0
    blocked: bool = False
    reason: str = ""
    body_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "method": self.method, "host": self.host, "status": self.status,
            "blocked": self.blocked, "reason": self.reason, "body_preview": self.body_preview[:500],
        }


class ReachGateway:
    """Single governed chokepoint for all off-machine actions."""

    def __init__(self, *, policy: ReachPolicy | None = None, http: Any | None = None) -> None:
        self.policy = policy or ReachPolicy()
        self._http = http  # injectable async client with .request(method, url, ...)

    @staticmethod
    def _host(url: str) -> str:
        try:
            return (urlparse(url).hostname or "").lower()
        except (ValueError, TypeError):
            return ""

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        reason: str = "",
    ) -> ReachResult:
        method = method.upper()
        host = self._host(url)
        allowed, why = self.policy.decide(method, host)
        if not allowed:
            logger.warning("🛡️ [Reach] refused %s %s — %s", method, host, why)
            return ReachResult(ok=False, method=method, host=host, url=url, blocked=True, reason=why)
        if method in _MUTATE_METHODS and not reason.strip():
            return ReachResult(ok=False, method=method, host=host, url=url, blocked=True,
                               reason="mutating reach requires a stated reason")
        try:
            if self._http is not None:
                resp = await self._http.request(method, url, json=json, headers=headers)
                status = int(getattr(resp, "status_code", 0))
                text = getattr(resp, "text", "") or ""
            else:
                request_headers = dict(headers or {})
                data = None
                if json is not None:
                    request_headers.setdefault("content-type", "application/json")
                    data = json_module.dumps(json)
                response = await get_network_gateway().request_async(
                    method,
                    url,
                    headers=request_headers,
                    data=data,
                    timeout=self.policy.timeout_s,
                    source="core.skills.reach_gateway",
                    read_only=method in _READ_METHODS,
                )
                status = int(response.get("status_code", 0) or 0)
                content = response.get("content", b"")
                if isinstance(content, bytes):
                    text = content.decode("utf-8", errors="replace")
                else:
                    text = str(content or "")
            preview = text[: self.policy.max_response_bytes]
            ok = 200 <= status < 400
            logger.info("🌐 [Reach] %s %s → %d%s", method, host, status, "" if ok else " (non-2xx)")
            return ReachResult(ok=ok, method=method, host=host, url=url, status=status,
                               reason="" if ok else f"http {status}", body_preview=preview)
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("reach_gateway", exc)
            return ReachResult(ok=False, method=method, host=host, url=url, reason=f"request error: {exc}")

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> ReachResult:
        return await self.request("GET", url, headers=headers)

    async def post(self, url: str, *, json: Any | None = None, reason: str, headers: dict[str, str] | None = None) -> ReachResult:
        return await self.request("POST", url, json=json, headers=headers, reason=reason)

    async def webhook(self, url: str, payload: dict[str, Any], *, reason: str) -> ReachResult:
        """Trigger an external automation / device webhook (governed mutate)."""
        return await self.request("POST", url, json=payload, reason=reason)

    def as_step(
        self,
        name: str,
        method: str,
        url: str,
        *,
        json: Any | None = None,
        reason: str = "",
        expect_status: int | None = 200,
        max_retries: int = 1,
    ):
        """Wrap a governed reach call as a FluidExecutor Step.

        The action performs the request and raises on a blocked/failed/unexpected-status
        response, so the executor's verify/recover/retry machinery applies uniformly.
        """
        from core.skills.fluid_executor import Step

        async def _action() -> None:
            result = await self.request(method, url, json=json, reason=reason)
            if result.blocked:
                raise RuntimeError(f"reach blocked: {result.reason}")
            if not result.ok:
                raise RuntimeError(f"reach failed: {result.reason}")
            if expect_status is not None and result.status != expect_status:
                raise RuntimeError(f"unexpected status {result.status} (wanted {expect_status})")

        return Step(name=name, action=_action, verify="always_true", max_retries=max_retries)


_instance: ReachGateway | None = None


def get_reach_gateway() -> ReachGateway:
    global _instance
    if _instance is None:
        _instance = ReachGateway()
    return _instance

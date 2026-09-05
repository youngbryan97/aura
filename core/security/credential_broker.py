"""core/security/credential_broker.py — a credential scoped to one act.

Aura's governance decides what may happen; below that decision the process
holds whatever ambient credentials it was started with. So a compromised child
process, a confused skill, or a prompt injection that reaches a tool call has
the same reach as the whole agent. Governance being right is doing all the work,
and it is the wrong layer to be doing it alone.

A brokered credential is bounded on four axes and every one of them is checked
at use rather than at grant:

* **Purpose.** It works for the operation it was issued for and no other.
* **Scope.** A host, a path, a resource. Wildcards are refused - a scope of
  ``*`` is an ambient credential with a ceremony.
* **Uses.** A count, usually one. A credential that can be used again after the
  act it was for is a credential that outlives its reason.
* **Time.** A lease, short. Expiry is checked against an injected clock so the
  test for it is not a sleep.

The secret is not here
----------------------
The broker holds a handle and a policy; the value is fetched from the vault at
use, by the broker, and never returned to the caller. A caller that could read
the value could keep it, and keeping it is exactly what the lease exists to
prevent. :meth:`CredentialBroker.use` takes the operation as a callback and
passes the value into it.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["Lease", "CredentialBroker", "CredentialRefused"]


class CredentialRefused(PermissionError):
    """A credential was asked to do something it was not scoped for."""


@dataclass
class Lease:
    """A bounded right to use one secret, for one purpose, a few times."""

    lease_id: str
    handle: str
    purpose: str
    scopes: frozenset[str]
    uses_remaining: int
    expires_at: float
    issued_to: str = ""
    used: int = 0
    revoked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id, "handle": self.handle, "purpose": self.purpose,
            "scopes": sorted(self.scopes), "uses_remaining": self.uses_remaining,
            "used": self.used, "revoked": self.revoked, "issued_to": self.issued_to,
        }


class CredentialBroker:
    """Issues leases, checks them at use, and never hands out the value."""

    def __init__(
        self,
        vault: Mapping[str, str],
        *,
        clock: Callable[[], float] = time.time,
        default_ttl: float = 60.0,
    ) -> None:
        self._lock = checked_lock("core.security.credential_broker.CredentialBroker", reentrant=True)
        self._vault = dict(vault)
        self._clock = clock
        self._default_ttl = float(default_ttl)
        self._leases: dict[str, Lease] = {}
        self._refusals: list[dict[str, str]] = []

    def issue(
        self,
        handle: str,
        *,
        purpose: str,
        scopes: Sequence[str],
        uses: int = 1,
        ttl: float | None = None,
        issued_to: str = "",
    ) -> Lease:
        """Issue a bounded lease. A wildcard scope is refused."""
        if handle not in self._vault:
            raise KeyError(f"no credential named {handle!r}")
        if not scopes or any(s.strip() in ("*", "") for s in scopes):
            raise CredentialRefused(
                "a wildcard scope is an ambient credential with a ceremony; name the "
                "hosts, paths or resources this lease is for"
            )
        if uses < 1:
            raise CredentialRefused("a lease with no uses cannot do anything")
        with self._lock:
            lease = Lease(
                lease_id=secrets.token_hex(8), handle=handle, purpose=purpose,
                scopes=frozenset(scopes), uses_remaining=int(uses),
                expires_at=self._clock() + (self._default_ttl if ttl is None else ttl),
                issued_to=issued_to,
            )
            self._leases[lease.lease_id] = lease
            return lease

    def use(
        self,
        lease_id: str,
        *,
        purpose: str,
        scope: str,
        operation: Callable[[str], Any],
    ) -> Any:
        """Run ``operation`` with the secret. The caller never sees the value."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                self._refuse(lease_id, "no such lease")
            if lease.revoked:
                self._refuse(lease_id, "revoked")
            if self._clock() >= lease.expires_at:
                self._refuse(lease_id, "expired")
            if lease.uses_remaining <= 0:
                self._refuse(lease_id, "no uses remaining")
            if purpose != lease.purpose:
                self._refuse(lease_id, f"issued for {lease.purpose!r}, used for {purpose!r}")
            if scope not in lease.scopes:
                self._refuse(lease_id, f"{scope!r} is not in {sorted(lease.scopes)}")
            lease.uses_remaining -= 1
            lease.used += 1
            value = self._vault[lease.handle]
        return operation(value)

    def revoke(self, lease_id: str) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            lease.revoked = True
            return True

    def _refuse(self, lease_id: str, reason: str) -> None:
        self._refusals.append({"lease_id": lease_id, "reason": reason})
        raise CredentialRefused(f"lease {lease_id}: {reason}")

    def report(self) -> dict[str, Any]:
        with self._lock:
            leases = list(self._leases.values())
            refusals = list(self._refusals)
        now = self._clock()
        return {
            "leases": len(leases),
            "live": sum(
                1 for l in leases
                if not l.revoked and l.uses_remaining > 0 and now < l.expires_at
            ),
            "spent": sum(1 for l in leases if l.uses_remaining <= 0),
            "revoked": sum(1 for l in leases if l.revoked),
            "refusals": refusals[-20:],
            "handles": sorted(self._vault),
            "values_returned_to_callers": 0,
        }

"""Who may drive the browser, where, and to do what.

``PhantomBrowser`` exposes browse, click, type, scroll, screenshot and
content extraction as plain methods returning booleans. CP126
``a66d2e59`` found there was no principal, no scoped authority, no
site-or-action policy, no approval lease and no executor receipt behind
any of them — so anything that could reach the object could click a
purchase button, type into a login form, or upload a file, and the caller
got back ``True``.

``8bf8d32e`` found the destination side of the same gap: ``browse``
checked whether the string started with the letters "http" and otherwise
prefixed "https". No parse, no scheme restriction, no credential
rejection, no private-address exclusion, no DNS-rebinding defence, no
port policy. ``core/runtime/url_policy.py`` already implements all of
that for HTTP fetches; the browser simply never called it.

This module is the boundary. Two rules give it its shape:

* **Reading is not acting.** Navigating and reading a public page is
  ordinary and needs a named principal and a destination that passes
  policy. Clicking and typing change someone else's system, so they need
  a lease that names the action and the origin it applies to.
* **A lease is spent.** It authorizes the interactions it was issued for
  and then it is gone, so one approval cannot become standing consent.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "BrowserAction",
    "BrowserLease",
    "BrowserVerdict",
    "authorize_browser_action",
    "issue_browser_lease",
    "revoke_browser_lease",
]


class BrowserAction(str, enum.Enum):
    """What a caller wants to do. Ordered by what it can change."""

    NAVIGATE = "navigate"
    READ = "read"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    CLICK = "click"
    TYPE = "type"

    @property
    def is_effectful(self) -> bool:
        """Whether this can change state on the far side of the network."""
        return self in {BrowserAction.CLICK, BrowserAction.TYPE}


#: Effectful leases expire. An approval that outlives the task it was
#: granted for is standing consent nobody gave.
DEFAULT_LEASE_TTL_S = 300.0
#: An unbounded lease is the same defect in slower motion.
MAX_LEASE_INTERACTIONS = 50


@dataclass
class BrowserLease:
    """Permission to interact with one origin, for a while, a few times."""

    lease_id: str
    principal: str
    origin: str
    actions: frozenset[BrowserAction]
    issued_at: float
    expires_at: float
    remaining: int
    purpose: str = ""

    def covers(self, action: BrowserAction, origin: str) -> tuple[bool, str]:
        if self.remaining <= 0:
            return False, "lease is spent"
        if time.time() > self.expires_at:
            return False, "lease has expired"
        if action not in self.actions:
            return False, f"lease does not cover {action.value}"
        if origin != self.origin:
            return False, f"lease covers {self.origin}, not {origin}"
        return True, ""


@dataclass(frozen=True)
class BrowserVerdict:
    """The decision, and everything a receipt needs to record it."""

    allowed: bool
    reason: str
    action: BrowserAction
    principal: str
    url: str = ""
    origin: str = ""
    lease_id: str = ""
    at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.capabilities.browser_authority.verdict.v1",
            "allowed": self.allowed,
            "reason": self.reason,
            "action": self.action.value,
            "principal": self.principal,
            "url": self.url,
            "origin": self.origin,
            "lease_id": self.lease_id,
            "at": self.at,
        }


_leases: dict[str, BrowserLease] = {}


def issue_browser_lease(
    *,
    principal: str,
    origin: str,
    actions: set[BrowserAction] | frozenset[BrowserAction],
    ttl_s: float = DEFAULT_LEASE_TTL_S,
    interactions: int = 10,
    purpose: str = "",
) -> BrowserLease:
    """Grant bounded permission to interact with one origin."""
    lease = BrowserLease(
        lease_id=uuid.uuid4().hex,
        principal=str(principal or "anonymous"),
        origin=str(origin or ""),
        actions=frozenset(actions),
        issued_at=time.time(),
        expires_at=time.time() + max(1.0, min(float(ttl_s), 3600.0)),
        remaining=max(1, min(int(interactions), MAX_LEASE_INTERACTIONS)),
        purpose=str(purpose or "")[:200],
    )
    _leases[lease.lease_id] = lease
    return lease


def revoke_browser_lease(lease_id: str) -> bool:
    return _leases.pop(str(lease_id), None) is not None


def origin_of(url: str) -> str:
    import urllib.parse

    parsed = urllib.parse.urlparse(str(url or ""))
    if not parsed.scheme or not parsed.hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def authorize_browser_action(
    action: BrowserAction,
    *,
    principal: str,
    url: str = "",
    lease_id: str = "",
    target: str = "",
) -> BrowserVerdict:
    """Decide one browser action. Refuses by default.

    A named principal is required for everything: an anonymous caller
    driving a real browser at a real site is the shape of the finding.
    Destinations go through :mod:`core.runtime.url_policy`, which already
    rejects non-https schemes, credentials in the URL, disallowed ports,
    hosts outside the allowlist and names that resolve to private or
    loopback addresses.
    """
    who = str(principal or "").strip()
    origin = origin_of(url) if url else ""
    if not who or who == "anonymous":
        return BrowserVerdict(
            False, "no principal named for a browser action", action, who, url, origin
        )

    if url:
        try:
            # The BROWSER policy, not the fetch policy: every SSRF check
            # (scheme, credentials, port, resolved-address classification,
            # DNS-rebinding defence) and no domain allowlist, because a
            # browser is meant to reach arbitrary public sites.
            from core.runtime.url_policy import validate_browser_url

            validated, error = validate_browser_url(url)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "browser_authority",
                exc,
                severity="warning",
                action="refused the navigation because URL policy could not be consulted",
            )
            return BrowserVerdict(False, "url policy unavailable", action, who, url, origin)
        if validated is None:
            return BrowserVerdict(False, f"url refused: {error}", action, who, url, origin)

    prohibited, why = _standing_prohibition(action, url, target)
    if prohibited:
        return BrowserVerdict(False, f"standing directive: {why}", action, who, url, origin)

    if not action.is_effectful:
        return BrowserVerdict(True, "read-only browser action", action, who, url, origin)

    lease = _leases.get(str(lease_id))
    if lease is None:
        return BrowserVerdict(
            False,
            f"{action.value} changes state on the far side and needs a lease",
            action,
            who,
            url,
            origin,
        )
    if lease.principal != who:
        return BrowserVerdict(False, "lease belongs to another principal", action, who, url, origin)
    covered, refusal = lease.covers(action, origin)
    if not covered:
        return BrowserVerdict(False, refusal, action, who, url, origin, lease.lease_id)

    lease.remaining -= 1
    return BrowserVerdict(True, "lease authorizes this interaction", action, who, url, origin, lease.lease_id)


def _standing_prohibition(action: BrowserAction, url: str, target: str) -> tuple[bool, str]:
    """Whether a standing directive covers this action. Fails CLOSED."""
    try:
        from core.governance.standing_directives import get_standing_directives

        match, _loaded = get_standing_directives().check(
            tool_name=f"browser.{action.value}",
            args={"url": str(url or ""), "target": str(target or "")},
            effect_scope="read_only" if not action.is_effectful else "network_write",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "browser_authority",
            exc,
            severity="warning",
            action="refused the browser action; standing directives could not be read",
        )
        return True, "directives unreadable"
    if match is not None:
        return True, str(match.matched_on)
    return False, ""

"""Reading a verdict out of a receipt without inventing one.

There is a recurring shape in this codebase, and it is always written the
same way::

    if not receipt.get("allowed", True):
        refuse()

That reads "refuse only if the receipt explicitly said no". A receipt that
said *nothing* — because the validator errored early, returned a partial
dict, was a stub, or answered a question it did not understand — is
therefore treated as approval. The absence of a check, reported as a passed
check.

The distinction that matters is three-valued, and a boolean cannot hold it:

``allow``    the receipt states a verdict and the verdict is yes
``deny``     the receipt states a verdict and the verdict is no
``unstated`` the receipt has no opinion — which is not a yes

A ``.get(key, True)`` collapses ``unstated`` into ``allow``; a
``.get(key, False)`` collapses it into ``deny``, which is safe but tells the
operator nothing about WHY something was refused, so the refusal looks like
a policy decision instead of a broken validator. Both lose the fact worth
knowing.

This module keeps the three apart, so a caller can fail closed and still say
which of the two reasons it failed on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ALLOW = "allow"
DENY = "deny"
UNSTATED = "unstated"

#: Keys that carry a verdict, in the order a receipt is asked for one.
#: A receipt may say "allowed": True or "denied": True; both are read.
_POSITIVE_KEYS = ("allowed", "approved", "permitted", "granted", "authorized", "ok")
_NEGATIVE_KEYS = ("denied", "blocked", "refused", "rejected")


@dataclass(frozen=True)
class Verdict:
    """What a receipt actually said, and why the caller may act on it."""

    state: str
    reason: str = ""
    key: str = ""

    @property
    def allows(self) -> bool:
        """True only for an explicit yes.

        This is the accessor a gate wants. ``unstated`` answers False, so a
        receipt that never formed an opinion cannot authorize anything.
        """
        return self.state == ALLOW

    @property
    def denies(self) -> bool:
        """True only for an explicit no — distinct from an absent verdict."""
        return self.state == DENY

    @property
    def is_stated(self) -> bool:
        return self.state in (ALLOW, DENY)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "key": self.key}


def read_verdict(
    receipt: Any,
    *,
    positive_keys: tuple[str, ...] = _POSITIVE_KEYS,
    negative_keys: tuple[str, ...] = _NEGATIVE_KEYS,
    reason_keys: tuple[str, ...] = ("reason", "error", "detail", "message"),
) -> Verdict:
    """Read the verdict a receipt states, or report that it stated none.

    Never infers. A receipt that is not a mapping, is empty, or carries none
    of the verdict keys comes back ``unstated`` — which is the honest answer
    and, at a gate, the safe one.
    """
    if not isinstance(receipt, Mapping):
        return Verdict(UNSTATED, f"receipt is {type(receipt).__name__}, not a mapping")

    def _reason() -> str:
        for key in reason_keys:
            value = receipt.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
        return ""

    # An explicit denial wins over an explicit approval: a receipt carrying
    # both is self-contradictory, and the safe reading of a contradiction at
    # a gate is the restrictive one.
    for key in negative_keys:
        if key in receipt and _is_definite(receipt[key]):
            if _truthy(receipt[key]):
                return Verdict(DENY, _reason() or f"receipt set {key}", key)
    for key in positive_keys:
        if key in receipt and _is_definite(receipt[key]):
            if _truthy(receipt[key]):
                return Verdict(ALLOW, "", key)
            return Verdict(DENY, _reason() or f"receipt set {key} false", key)
    return Verdict(UNSTATED, _reason() or "receipt stated no verdict")


def _is_definite(value: Any) -> bool:
    """None means "not determined", which is not a verdict either."""
    return value is not None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "allow", "allowed"}
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def receipt_allows(receipt: Any, **kwargs: Any) -> bool:
    """Shorthand for ``read_verdict(receipt).allows`` — explicit yes only."""
    return read_verdict(receipt, **kwargs).allows


__all__ = [
    "ALLOW",
    "DENY",
    "UNSTATED",
    "Verdict",
    "read_verdict",
    "receipt_allows",
]

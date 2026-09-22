"""What actually leaves this machine, read before it leaves.

``core/runtime/network_gateway.py`` had a complete outbound boundary except
for one thing: it never looked at the body. The preflight
(``validate_outbound_network``) is handed ``method``, ``url``,
``data_length`` and ``source`` — a length, not the content. Governance
decides whether *a* request may be made; nothing decided what was inside it.

So an outbound request carrying sensitive context could go out unread. The
redactor in :mod:`core.security.structural_redaction` already knew how to
find an API key or an email address in nested data, and every audit path in
the system used it — but it ran on the *record* of a call, never on the call
itself. The transcript in the database was clean and the copy sent to a
third party was not.

This module is the missing read. Two tiers, because the two kinds of
sensitive content answer to different authorities:

**Credentials never leave, anywhere.** An API key, a JWT, a private key
block in an outbound body is a leak whatever the destination and whatever
the tool intended. :data:`~core.security.structural_redaction.CREDENTIAL_PATTERNS`
applies to every inspected destination.

**Personal identifiers leave when the destination is not a stranger.** An
email address in the body of an email tool is the payload, not the leak —
stripping it would break the tool and teach someone to route around this
module. The personal tier therefore applies where the *destination* makes
the content wrong, which today means a third-party model provider: the one
place where a turn's context becomes somebody else's log line.

Three properties matter more than the patterns:

**Local is not egress.** Loopback, this machine's own LAN, and ``.local``
names are where Aura's own runtime, its MLX worker, and its paired devices
live. Filtering there would corrupt Aura talking to herself while
protecting nobody. Those destinations are passed through untouched, and the
receipt says so rather than implying an inspection happened.

**Structure, not the wire bytes.** Redaction runs over parsed JSON strings
and never over the serialized document. The phone-number pattern matches
``1234.5678``, so a regex sweep across raw JSON would silently turn an
embedding vector into ``[PHONE_REDACTED]`` and the request would fail far
from here, looking like a provider bug. Numbers are not strings and are
never touched. Bodies that parse to no strings come back byte-identical.
"Every string" means keys too: a dict key is the only place in JSON where a
string is not a value, and reading values alone let ``{"sk-live-…": 1}``
through a boundary whose whole claim is that credentials never leave.

**Fail closed means closed.** A body that cannot be decoded cannot be
inspected, and a model-provider destination therefore does not get it. The
receipt carries ``inspected`` as a fact, so no reader has to infer from
"allowed" that a check ran — the failure this codebase keeps rediscovering
is the absence of a check reported as a passed check.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.security.structural_redaction import (
    CREDENTIAL_PATTERNS,
    PERSONAL_PATTERNS,
    redact_text,
)

logger = logging.getLogger("Aura.EgressPrivacy")

#: Compatibility prefix for old model-provider egress receipts. No remote
#: model endpoint can be registered, but retaining the stronger privacy tier
#: keeps archived or replayed receipts conservative.
MODEL_PROVIDER_SOURCE_PREFIX = "llm_provider:"

#: Hostname suffixes that name this machine or its local network.
_LOCAL_HOST_SUFFIXES = (".local", ".localdomain", ".internal")
_LOCAL_HOST_NAMES = frozenset({"localhost", "localhost.localdomain", ""})

_DECODE_ERRORS = (AttributeError, LookupError, UnicodeDecodeError, ValueError)

_TELEMETRY_CHANNELS = (
    dict(
        identifier=0x0601,
        name="egress.bodies_redacted",
        type="int",
        unit="count",
        description="outbound bodies that had content stripped before sending",
        owner="core/security/egress_privacy.py",
        group="privacy",
        yellow_high=1,
        stale_after_s=3600.0,
    ),
    dict(
        identifier=0x0602,
        name="egress.bodies_refused",
        type="int",
        unit="count",
        description="outbound bodies refused because they could not be inspected",
        owner="core/security/egress_privacy.py",
        group="privacy",
        yellow_high=1,
        red_high=5,
        stale_after_s=3600.0,
    ),
)
_declared_channels = False
_redacted_total = 0
_refused_total = 0


class Tier:
    """Which patterns apply at a destination."""

    #: This machine or its LAN — nothing applies, nothing is claimed.
    LOCAL = "local"
    #: A third party doing a job for us: credentials still never leave.
    CREDENTIALS = "credentials"
    #: A third-party model: the turn's context becomes their log line.
    FULL = "full"


@dataclass(frozen=True)
class EgressFilterResult:
    """What the boundary did, stated rather than implied."""

    allowed: bool
    body: bytes | None
    tier: str
    inspected: bool
    redactions: int = 0
    reason: str = ""
    kinds: tuple[str, ...] = field(default_factory=tuple)
    #: Set instead of ``body`` when the caller handed us a prompt rather than
    #: a request body — an SDK that builds its own HTTP. Same boundary, same
    #: tiers; only the shape of what is being inspected differs.
    text: str | None = None

    @property
    def modified(self) -> bool:
        return self.redactions > 0

    def to_dict(self) -> dict[str, Any]:
        # Deliberately no body and no sample of what was found: a receipt
        # that quotes the secret it caught has moved the secret into the
        # audit log. Kinds are the diagnostic; values are the thing being
        # protected.
        return {
            "allowed": self.allowed,
            "tier": self.tier,
            "inspected": self.inspected,
            "redactions": self.redactions,
            "kinds": list(self.kinds),
            "reason": self.reason,
        }


def destination_is_local(url: str) -> bool:
    """Is this host this machine, or the network this machine sits on?

    Private LAN counts as local because that is where a paired phone and the
    MLX worker live. It is a deliberate trust boundary, not an oversight:
    egress privacy is about content reaching a stranger, and the devices the
    owner paired are not strangers.
    """
    try:
        host = (urllib.parse.urlsplit(str(url or "")).hostname or "").strip().lower()
    # not a failure: the comment below says it: an unparseable URL is not evidence of
    # a local destination.
    except ValueError:
        # An unparseable URL is not evidence of a local destination.
        return False
    if host in _LOCAL_HOST_NAMES:
        return True
    if host.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    # not a failure: a host that is not an IP address is not a local one, which is
    # the refusing direction here.
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    )


def tier_for(url: str, source: str) -> str:
    """The tier this destination earns."""
    if destination_is_local(url):
        return Tier.LOCAL
    if str(source or "").startswith(MODEL_PROVIDER_SOURCE_PREFIX):
        return Tier.FULL if _personal_redaction_enabled() else Tier.CREDENTIALS
    return Tier.CREDENTIALS


def _personal_redaction_enabled() -> bool:
    """Does the operator want personal identifiers held back from models?

    Defaults to True. The setting exists because it is the owner's call, not
    because the answer is unclear: a config read that fails answers True, so
    a broken config cannot quietly open the boundary.
    """
    try:
        from core.config import get_config

        return bool(
            getattr(
                get_config().security,
                "redact_personal_data_to_model_providers",
                True,
            )
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "egress_privacy",
            exc,
            severity="debug",
            action="kept personal-identifier redaction on after config read failed",
            enforce_failure_policy=False,
        )
        return True


def _patterns_for(tier: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    if tier == Tier.FULL:
        # Credentials first — an email inside a userinfo URL is already gone
        # by the time the personal patterns run.
        return CREDENTIAL_PATTERNS + PERSONAL_PATTERNS
    return CREDENTIAL_PATTERNS


def _redact_strings_in_place(
    root: Any, patterns: tuple[tuple[re.Pattern[str], str], ...]
) -> tuple[Any, int, set[str]]:
    """Redact every string in a parsed JSON document — keys as well as values.

    Iterative rather than recursive: ``json.loads`` already proved the
    document is finite and acyclic, so the only thing recursion would add is
    a stack limit that a deeply nested but legitimate payload could hit.

    A dict key is the one place in JSON where a string sits and is not a
    value, and walking only the values left exactly that hole open:
    ``{"sk-live-…": "quota"}`` went out intact while the identical secret one
    field to the right did not. The tier decides *which* patterns apply; it
    does not get to decide that half the strings in the document are exempt
    from them. Provider schemas key on fixed names, so this costs nothing on
    the traffic that exists today and closes the boundary for the traffic
    that does not exist yet.
    """
    changes = 0
    kinds: set[str] = set()
    stack: list[Any] = [root]

    def _scrub(text: str) -> str:
        nonlocal changes
        redacted, changed = redact_text(text, patterns=patterns)
        if changed:
            changes += 1
            kinds.update(_kinds_in(redacted, patterns))
        return redacted

    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            renamed: dict[str, str] = {}
            # Snapshot the items: the value assignments below only rebind
            # existing keys, but the key rewrite that follows resizes the dict.
            for key, value in list(node.items()):
                if isinstance(value, str):
                    node[key] = _scrub(value)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
                if isinstance(key, str):
                    scrubbed_key = _scrub(key)
                    if scrubbed_key != key:
                        renamed[key] = scrubbed_key
            if renamed:
                _rewrite_keys(node, renamed)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, str):
                    node[index] = _scrub(value)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, str):
            # A bare JSON string document.
            return _scrub(node), changes, kinds

    return root, changes, kinds


def _rewrite_keys(node: dict[Any, Any], renamed: dict[str, str]) -> None:
    """Swap redacted keys in without dropping an entry.

    Two distinct keys can redact to the same placeholder — an object keyed by
    API key does exactly that — and plain reassignment would silently discard
    one of the values. Quietly sending less than the caller asked for is its
    own defect and would surface as an unreproducible provider error, so a
    collision gets a positional discriminator instead of a deletion.

    The dict is rebuilt rather than mutated key-by-key so that insertion
    order survives; a reordered request body would make this module the first
    suspect the day a provider changes behaviour, for no benefit.
    """
    rebuilt: dict[Any, Any] = {}
    for key, value in node.items():
        new_key = renamed.get(key, key) if isinstance(key, str) else key
        if new_key in rebuilt:
            discriminator = 2
            while f"{new_key}~{discriminator}" in rebuilt:
                discriminator += 1
            new_key = f"{new_key}~{discriminator}"
        rebuilt[new_key] = value
    node.clear()
    node.update(rebuilt)


def filter_outbound_body(
    *,
    url: str,
    body: bytes | None,
    source: str,
    publish_evidence: bool = True,
) -> EgressFilterResult:
    """Read an outbound body and return what may actually be sent.

    Never raises: the caller is a network path, and an exception here would
    turn a privacy control into an outage. A failure inside the filter is a
    refusal for model providers and a recorded pass-through elsewhere, which
    is the honest reading of "we did not manage to look".
    """
    tier = tier_for(url, source)
    if tier == Tier.LOCAL:
        return EgressFilterResult(
            allowed=True,
            body=body,
            tier=tier,
            inspected=False,
            reason="destination is this machine or its local network",
        )
    if not body:
        return EgressFilterResult(
            allowed=True, body=body, tier=tier, inspected=True, reason="empty body"
        )

    try:
        text = body.decode("utf-8")
    except _DECODE_ERRORS:
        return _uninspectable(
            tier,
            body,
            "body is not UTF-8 text and cannot be read before sending",
            publish_evidence=publish_evidence,
        )

    patterns = _patterns_for(tier)
    try:
        document = json.loads(text)
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except (json.JSONDecodeError, RecursionError, ValueError):
        document = None

    try:
        if isinstance(document, (dict, list, str)):
            redacted_doc, changes, kinds = _redact_strings_in_place(document, patterns)
            new_body = (
                json.dumps(redacted_doc).encode("utf-8") if changes else body
            )
        else:
            # Not JSON, or a bare number/bool: treat the whole body as text.
            redacted_text, changed = redact_text(text, patterns=patterns)
            changes = 1 if changed else 0
            kinds = _kinds_in(redacted_text, patterns) if changed else set()
            new_body = redacted_text.encode("utf-8") if changed else body
    except (AttributeError, RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
        record_degradation(
            "egress_privacy",
            exc,
            severity="warning",
            action="refused or passed the body per tier after redaction failed",
            enforce_failure_policy=False,
        )
        return _uninspectable(
            tier,
            body,
            f"redaction failed: {exc}",
            publish_evidence=publish_evidence,
        )

    if not changes:
        # Byte-identical when nothing was found. Re-serializing a clean body
        # would change bytes for no reason and make this module the first
        # suspect the day a provider rejects a request.
        return EgressFilterResult(
            allowed=True,
            body=body,
            tier=tier,
            inspected=True,
            reason="nothing sensitive found",
        )

    if publish_evidence:
        _count_redaction()
        logger.info(
            "Egress privacy: stripped %d string(s) [%s] from a %s body to %s",
            changes,
            ",".join(sorted(kinds)) or "unclassified",
            tier,
            urllib.parse.urlsplit(str(url or "")).hostname or "unknown-host",
        )
    return EgressFilterResult(
        allowed=True,
        body=new_body,
        tier=tier,
        inspected=True,
        redactions=changes,
        kinds=tuple(sorted(kinds)),
        reason="sensitive values stripped before sending",
    )


#: The ``[TOKEN]`` a replacement leaves behind. Pulled out rather than
#: assumed at the start of the string, because the userinfo replacement is
#: ``\1[REDACTED_USERINFO]@`` — a prefix test would classify a caught
#: credential-in-URL as "unclassified" and quietly under-report the one
#: pattern most likely to fire on a real request.
_KIND_MARKER = re.compile(r"\[([A-Z_]+)\]")


def _kinds_in(
    text: str, patterns: tuple[tuple[re.Pattern[str], str], ...]
) -> set[str]:
    kinds: set[str] = set()
    for _pattern, replacement in patterns:
        marker = _KIND_MARKER.search(replacement)
        if marker is not None and marker.group(0) in text:
            kinds.add(marker.group(1).lower())
    return kinds


def _uninspectable(
    tier: str,
    body: bytes | None,
    reason: str,
    *,
    publish_evidence: bool = True,
) -> EgressFilterResult:
    """A body we could not read. Model providers do not get it."""
    if tier == Tier.FULL:
        if publish_evidence:
            _count_refusal()
            logger.warning("Egress privacy: refused an uninspectable body — %s", reason)
        return EgressFilterResult(
            allowed=False,
            body=None,
            tier=tier,
            inspected=False,
            reason=reason,
        )
    # A binary upload to a tool endpoint is normal traffic and refusing it
    # would break working capability to protect nothing this filter can see.
    # It goes, and the receipt states that nothing was inspected.
    return EgressFilterResult(
        allowed=True,
        body=body,
        tier=tier,
        inspected=False,
        reason=reason,
    )


def _count_redaction() -> None:
    global _redacted_total
    _redacted_total += 1
    _write_channel("egress.bodies_redacted", _redacted_total)


def _count_refusal() -> None:
    global _refused_total
    _refused_total += 1
    _write_channel("egress.bodies_refused", _refused_total)


def _write_channel(name: str, value: int) -> None:
    global _declared_channels
    try:
        from core.fsw.telemetry_dictionary import ChannelType, channel, write

        if not _declared_channels:
            for spec in _TELEMETRY_CHANNELS:
                channel(**{**spec, "type": ChannelType.INT})
            _declared_channels = True
        write(name, value)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        record_degradation(
            "egress_privacy",
            exc,
            severity="debug",
            action=f"egress channel {name} not written",
            enforce_failure_policy=False,
        )


def egress_privacy_counters() -> dict[str, int]:
    """Totals since process start, for health reporting."""
    return {"bodies_redacted": _redacted_total, "bodies_refused": _refused_total}


def reset_egress_privacy_counters_for_test() -> None:
    global _redacted_total, _refused_total
    _redacted_total = 0
    _refused_total = 0


__all__ = [
    "MODEL_PROVIDER_SOURCE_PREFIX",
    "EgressFilterResult",
    "Tier",
    "destination_is_local",
    "egress_privacy_counters",
    "filter_outbound_body",
    "reset_egress_privacy_counters_for_test",
    "tier_for",
]

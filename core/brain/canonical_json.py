"""Pure canonical JSON encoding shared by signed brain evidence.

Keep this module free of runtime state, filesystem access, and model imports so
symbolic verifier identities can include it without acquiring those powers.
"""

from __future__ import annotations

import json
from typing import Any


#: The canonicalization contract these signatures are made under.
#:
#: sort_keys and compact separators are a Python recipe, not a cross-language
#: standard: RFC 8785 pins number formatting (shortest round-trip ECMAScript
#: representation) and string escaping in ways ``json.dumps`` does not promise
#: across implementations or versions. A signature verified in another language
#: — or by a future Python whose float repr differs — could disagree about the
#: bytes without either side being wrong.
#:
#: Naming the profile is what makes that checkable. Anything relying on these
#: digests can require this exact identifier, and a move to RFC 8785 or
#: canonical CBOR becomes a version bump rather than a silent divergence.
CANONICAL_JSON_PROFILE = "aura.canonical_json.python_sorted_compact.v1"

#: What the profile guarantees, and what it does not.
CANONICAL_JSON_CONTRACT: dict[str, Any] = {
    "profile": CANONICAL_JSON_PROFILE,
    "key_order": "lexicographic_by_unicode_codepoint",
    "separators": "compact",
    "string_escaping": "python_json_ensure_ascii",
    "number_formatting": "python_repr",
    "non_finite_numbers": "rejected",
    "cross_language_standard": None,
    "known_divergences": [
        "number formatting is Python's repr, not RFC 8785 shortest round-trip",
        "string escaping follows Python's ensure_ascii rather than RFC 8785",
    ],
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value deterministically for hashes/signatures.

    The encoding is pinned by ``CANONICAL_JSON_PROFILE``. It is deterministic
    within this profile; it is not RFC 8785, and the contract says so rather
    than leaving a verifier in another language to discover it.
    """

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "CANONICAL_JSON_CONTRACT",
    "CANONICAL_JSON_PROFILE",
    "canonical_json_bytes",
]

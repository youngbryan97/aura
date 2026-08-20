"""Parts of a message that are addresses, not words.

A path, a URL, a UUID and a long hyphenated identifier are opaque: they name
something, and the characters inside them do not mean what they would mean in
a sentence. Reading intent out of them produces confident nonsense.

Twice on 2026-08-19:

* "there's a python project at /private/tmp/claude-501/-Users-bryan--aura-
  live-source/7a6cdc9e-da7f-47f7-8c38-.../ledger ... work out why" was
  answered "30." — the arithmetic matcher found "7-8" inside the UUID
  `47f7-8c38` and "work out" satisfied its intent gate.
* The same path routed a request to debug a Python file into the
  browser-dialogue skill for talking to another AI, because the directory is
  called `claude-501`.

Both matchers were correct about the strings they saw. What neither could see
is that those characters were part of an address the person pasted, not
something they said. Stripping the opaque spans first gives every matcher the
sentence the person actually wrote.

Deliberately conservative: it removes spans that are unambiguously addresses,
because a matcher that loses a real word is worse than one that keeps a stray
identifier.
"""

from __future__ import annotations

import re

__all__ = ["OPAQUE_SPAN_RE", "without_opaque_spans"]

#: What an address looks like, in the order that matters.
#:
#: URLs first, because they contain paths. Then filesystem paths, which are
#: the common case. Then bare UUIDs and hash-like runs, which appear in logs
#: and ids without a surrounding path.
OPAQUE_SPAN_RE = re.compile(
    r"""(?:
        [a-z][a-z0-9+.\-]*://\S+                      # a URL of any scheme
      | (?<![\w])~?/[\w.\-~]+(?:/[\w.\-~]+)+/?        # an absolute-ish path
      | (?<![\w])[\w.\-]+/[\w.\-]+(?:/[\w.\-]+)*      # a relative path
      | \b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b   # a UUID
      | \b[0-9a-f]{16,}\b                             # a long hex digest
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def without_opaque_spans(text: object, *, replacement: str = " ") -> str:
    """The message with its addresses removed.

    The replacement is a space rather than nothing, so words either side of a
    stripped path do not fuse into one that was never written.
    """
    body = str(text or "")
    if not body:
        return ""
    return OPAQUE_SPAN_RE.sub(replacement, body)

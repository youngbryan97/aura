"""Effect-ownership debt is two different claims, and only one is a bypass.

The headline number was one undifferentiated 1,973, which invites the reading
that Aura contains 1,973 exploitable governance bypasses. It does not, and the
split is not close:

    1,000  routed through a declared gateway, not ActionExecutor-owned
      973  raw ungoverned primitives

A call to subprocess_gateway, file_write_gateway or atomic_writer is following
the convention CLAUDE.md documents — "all consequential file writes go through
core/runtime/file_write_gateway.py". It is migration debt only in the narrow
sense that ActionExecutor is not its canonical owner. The top offender by raw
count, core/self_modification/safe_modification.py, is 26 calls of which every
single one goes through a gateway.

A raw primitive is a different thing: nothing governs it. That tier is pinned
separately here so it can be driven down on its own, rather than hiding inside
a total that barely moves when it does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "aura_effect_ownership_baseline.json"

#: Measured at the time of splitting. Both may only fall.
#: RAW was 973 and is 905 after removing 69 `X.parent.mkdir(...)` calls that
#: the atomic writer already performs — dead code that also registered as
#: ungoverned effect debt.
#: 1,000/905 before the orphan retirement removed 112 unreachable modules,
#: which took their effect call sites with them.
#: The governed tier is deliberately NOT ratcheted downward, and that is not
#: laxity. Paying raw debt MOVES a call into it — migrating atomic_storage's
#: rename/unlink to durable_replace/durable_unlink took raw 874 -> 869 and
#: governed 974 -> 979. A "governed only falls" rule would fail on exactly the
#: work it is meant to encourage.
#:
#: The two invariants that do hold: total debt only falls, and the raw tier
#: only falls. Together they permit migration and forbid regression.
#: 1848/866 before twelve `X.parent.mkdir(...)` calls came out — each one
#: immediately followed by a gateway or atomic write of that same path, which
#: creates the parent itself. Same finding as the 69 before them, in the files
#: that were written after.
TOTAL_CEILING = 1840
RAW_CEILING = 854


def _split() -> tuple[int, int]:
    from tools.lint_governance_cli import GATEWAY_CATEGORIES

    buckets = json.loads(BASELINE.read_text(encoding="utf-8"))["buckets"]
    governed = raw = 0
    for bucket in buckets:
        if bucket["canonical_owner"]:
            continue
        if bucket["category"] in GATEWAY_CATEGORIES:
            governed += bucket["count"]
        else:
            raw += bucket["count"]
    return governed, raw


def test_the_raw_tier_only_falls():
    """The safety-relevant half. Nothing governs these call sites."""
    _governed, raw = _split()
    assert raw <= RAW_CEILING, (
        f"ungoverned raw effect primitives rose to {raw} (ceiling {RAW_CEILING})"
    )


def test_total_debt_only_falls():
    """Governed may rise as raw falls; the sum may not rise either way."""
    governed, raw = _split()
    assert governed + raw <= TOTAL_CEILING


def test_the_two_tiers_account_for_all_the_debt():
    """A category added later must land in one tier or the other, not vanish."""
    governed, raw = _split()
    buckets = json.loads(BASELINE.read_text(encoding="utf-8"))["buckets"]
    total = sum(b["count"] for b in buckets if not b["canonical_owner"])
    assert governed + raw == total


def test_gateway_categories_name_real_gateways():
    """A category listed as 'governed' must correspond to an actual gateway.

    Without this the tiering is just a way to make the number look smaller:
    anything inconvenient could be declared a gateway category.
    """
    from tools.lint_governance_cli import GATEWAY_CATEGORIES

    expected_modules = {
        "file_write_gateway": "core/runtime/file_write_gateway.py",
        "direct_atomic_file_write": "core/runtime/atomic_writer.py",
        "subprocess_gateway": "core/runtime/subprocess_gateway.py",
        "network_gateway": "core/runtime/network_gateway.py",
    }
    for category, module in expected_modules.items():
        assert category in GATEWAY_CATEGORIES
        assert (ROOT / module).is_file(), f"{category} names a gateway that is missing"


def test_the_gateways_themselves_are_canonical_owners():
    """A gateway cannot route through itself; counting it as debt is incoherent."""
    buckets = json.loads(BASELINE.read_text(encoding="utf-8"))["buckets"]
    for path in ("core/runtime/atomic_writer.py", "core/runtime/file_write_gateway.py"):
        rows = [b for b in buckets if b["path"] == path]
        assert rows, f"{path} has no recorded effect calls at all"
        raw_debt = [
            b for b in rows
            if not b["canonical_owner"] and b["category"] == "raw_file_mutation"
        ]
        assert not raw_debt, (
            f"{path} is the write gateway and its own writes are counted as debt"
        )


# --------------------------------------------------------------------------
# The payable third of the raw tier
# --------------------------------------------------------------------------

#: unlink/remove/rename/replace calls, which durable_unlink and durable_replace
#: could take ownership of. Measured at the time of splitting; only falls.
PAYABLE_CEILING = 179


def _payable() -> int:
    from tools.lint_governance_cli import GATEWAY_CATEGORIES, PAYABLE_PRIMITIVES

    buckets = json.loads(BASELINE.read_text(encoding="utf-8"))["buckets"]
    return sum(
        b["count"]
        for b in buckets
        if not b["canonical_owner"]
        and b["category"] not in GATEWAY_CATEGORIES
        and any(token in b["callee"] for token in PAYABLE_PRIMITIVES)
    )


def test_the_payable_raw_debt_only_falls():
    assert _payable() <= PAYABLE_CEILING


def test_every_payable_primitive_has_a_governed_equivalent():
    """The split is only honest if "payable" means a migration target exists.

    Otherwise it is a way to make the number look smaller by declaring the
    inconvenient part unpayable.
    """
    writer = (ROOT / "core" / "runtime" / "atomic_writer.py").read_text(encoding="utf-8")
    assert "def durable_unlink(" in writer
    assert "def durable_replace(" in writer


def test_mkdir_is_not_counted_as_payable():
    """It is over half the raw tier and has no matching governed primitive.

    ensure_directory creates a PRIVATE 0o700 directory and requires an active
    governance scope, so substituting it for `path.mkdir(parents=True,
    exist_ok=True)` would tighten permissions at 450+ sites and fail wherever
    no governed scope is open. Counting those as migratable would imply a
    migration that breaks the system.
    """
    from tools.lint_governance_cli import PAYABLE_PRIMITIVES

    assert not any("mkdir" in token for token in PAYABLE_PRIMITIVES)
    gateway = (
        ROOT / "core" / "runtime" / "file_write_gateway.py"
    ).read_text(encoding="utf-8")
    assert "ensure_private_directory" in gateway, (
        "ensure_directory no longer delegates to the private-directory creator; "
        "re-check whether mkdir has become migratable"
    )

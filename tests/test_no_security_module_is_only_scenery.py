"""Security-shaped code that reads as live and is not.

``core/security/consent_kernel.py`` defined ``audit_and_verify_action``,
documented as "Runs the complete safety verification chain" — secret
leakage, network policy, risk classification, approval, audit. Nothing
called it. The only reference anywhere outside its own cluster was a
docstring in ``sandbox.py``, and two tests that invoked it directly and
watched it refuse. It refused correctly. It never ran.

That is worse than an absent control. Someone auditing this tree reads
the chain, sees the tests pass, and believes it runs. The absence of a
check reported as a passed check is the failure this codebase keeps
rediscovering, and this is its shape at the level of a whole module.

Every guarantee it advertised has a live owner, and each of those is
stricter than what the chain did:

* secret leakage — the boundary that matters is egress, not a local
  write: ``egress_privacy.filter_outbound_body`` runs on every request
  through ``network_gateway``, with counters on the health surface.
  ``SecretGuard`` tested six spellings and would miss an AWS key, a
  GitHub token, a JWT and a PEM body.
* network policy — ``NetworkGateway`` holds the allowed domains and reads
  an authorization verdict that fails CLOSED when the verdict is unstated.
  ``NetworkPolicyEngine`` was ``host in allowed``, with no subdomain, no
  scheme and no port.
* egress recording — ``egress_privacy_counters`` on the health surface.
* approval and risk — ``authorization_receipt.read_verdict``, and for
  source changes ``mutation_constitution.admit_mutation``.
  ``ActionRiskClassifier`` was three substring tests returning 8, 4, 7
  or 1, and ``ApprovalRequiredChecker`` was ``risk >= 7``.
* audit — ``core/runtime/receipts.py`` and the audit chain.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECURITY = ROOT / "core" / "security"

#: Where a live consumer could be. Not core/security itself: a cluster
#: that only imports itself is exactly the shape being ruled out.
CONSUMERS = ("core", "interface", "skills", "executors", "tools", "tests")

DELETED_CHAIN = (
    "consent_kernel",
    "secret_guard",
    "network_policy",
    "action_risk_classifier",
    "approval_required",
    "egress_monitor",
)


def _sources() -> dict[pathlib.Path, str]:
    found: dict[pathlib.Path, str] = {}
    for top in CONSUMERS:
        for path in (ROOT / top).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            if SECURITY in path.parents:
                continue
            try:
                found[path] = path.read_text(encoding="utf-8")
            except OSError:
                continue
    return found


def test_every_security_module_has_a_consumer_outside_the_package():
    sources = _sources()
    orphans = []
    for module in sorted(SECURITY.glob("*.py")):
        if module.name == "__init__.py":
            continue
        pattern = re.compile(rf"\b{re.escape(module.stem)}\b")
        if not any(pattern.search(text) for text in sources.values()):
            orphans.append(module.stem)
    assert not orphans, (
        "security modules nothing outside core/security reaches for: "
        f"{orphans}. A safety control with no caller is scenery, and it reads "
        "as protection to anyone auditing this tree."
    )


def test_the_chain_that_advertised_itself_is_gone():
    for stem in DELETED_CHAIN:
        assert not (SECURITY / f"{stem}.py").exists(), (
            f"core/security/{stem}.py is back. It was deleted because nothing "
            "called it and everything it claimed has a stricter live owner; "
            "if it returns it needs a caller, not a test that invokes it "
            "directly and watches it refuse."
        )


def test_nothing_imports_the_deleted_chain():
    sources = _sources()
    offenders = []
    for path, text in sources.items():
        for stem in DELETED_CHAIN:
            if re.search(rf"core\.security\.{re.escape(stem)}\b", text):
                offenders.append(f"{path.relative_to(ROOT)} -> {stem}")
    assert not offenders, offenders


def test_the_live_owners_are_still_wired():
    """The guarantees have to be somewhere, and this says where.

    Deleting a dead control is only right while the live one is real, so
    this fails if the replacement stops being reachable.
    """

    from core.runtime.network_gateway import _filter_outbound_body  # noqa: F401
    from core.runtime.authorization_receipt import read_verdict  # noqa: F401
    from core.security.egress_privacy import (  # noqa: F401
        egress_privacy_counters,
        filter_outbound_body,
    )
    from core.self_modification.mutation_constitution import (  # noqa: F401
        admit_mutation,
    )

    gateway = (ROOT / "core" / "runtime" / "network_gateway.py").read_text("utf-8")
    assert "filter_outbound_body" in gateway, (
        "outbound redaction is no longer called from the network gateway"
    )

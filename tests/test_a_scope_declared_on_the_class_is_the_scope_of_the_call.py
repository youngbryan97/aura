"""A skill that declares its effect scope on its class is scoped by that declaration when it is called.

catalog_policy asks new skills to declare ``effect_scope`` beside their code, and registration refuses one whose code
reaches past it. The resolver consulted at the moment of a call read only the per-action table and the older name
table, so a skill declared the new way resolved to "unknown", which is rated critical.

LIVE, 9 November 2026: her own goal pass was refused read_source and find_in_source, both declared read_only, with
"Requires user confirmation: Typed execution contract: scope=unknown, risk=critical". Nobody is there to confirm a
pass at a goal she set herself, so it ended having read nothing. Eight skills were declared that way.
"""

from __future__ import annotations

import pytest

from core.capabilities.permission_model import PermissionRiskModel
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope
from core.skills.action_scope import _declared_classes, _import_class
from core.skills.catalog_policy import VALID_EFFECT_SCOPES


def _declared_on_class() -> dict[str, str]:
    found = {}
    for name, (module, cls) in _declared_classes().items():
        scope = str(getattr(_import_class(module, cls), "effect_scope", "") or "").strip().lower()
        if scope in VALID_EFFECT_SCOPES:
            found[name] = scope
    return found


def test_no_skill_with_a_valid_declaration_on_its_class_resolves_to_unknown():
    unknown = sorted(name for name in _declared_on_class() if resolve_execution_effect_scope(name, {}) == "unknown")
    assert not unknown, f"declared on the class and still resolved as unknown: {unknown}"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("read_source", {"path": "core/memory/cold_store.py"}),
        ("find_in_source", {"pattern": "TODO", "where": "core/memory"}),
    ],
)
def test_her_goal_pass_may_read_and_search_her_source_without_a_confirmation(name, args):
    scope = resolve_execution_effect_scope(name, args)
    risk = classify_execution_risk(name, args, effect_scope=scope)
    decision = PermissionRiskModel().check_permission(
        name, "", {"origin": "autonomous"}, effect_scope=scope, execution_risk=risk
    )
    assert scope == "read_only"
    assert risk == "low"
    assert decision.approved and not decision.requires_confirmation


def test_a_declaration_that_is_not_a_scope_is_not_believed(monkeypatch):
    class Boasting:
        effect_scope = "harmless"

    monkeypatch.setattr("core.skills.action_scope.skill_class_named", lambda name: Boasting)
    monkeypatch.setattr("core.executive.execution_policy._skill_meta_for", lambda name: None)
    assert resolve_execution_effect_scope("a_skill_nobody_listed", {}) == "unknown"

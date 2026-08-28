"""A capability is refused for what it can do, not for what the request says.

LIVE, 2026-08-28: "post a 25000-cent consulting invoice, debit Accounts
Receivable, credit Revenue" was refused with "Modality 'network_write' is
disabled". The pattern for network_write is the bare word "post", and posting
an entry to a ledger is the older sense of the word — the one accountants have
used since before there were networks to post to.

The tool was code_repl, whose effect scope is sandboxed_compute, declared where
it is defined to "calculate anything and change nothing outside its own
sandbox". It could not have reached a network if it had tried. The refusal was
about a word.

The narrowing here is exact: a modality the scope cannot reach is not detected
from prose. Every modality the scope CAN reach is still detected, and still
refused if it is disabled.
"""

from __future__ import annotations

from core.capabilities.permission_model import PermissionRiskModel


def test_posting_a_ledger_entry_is_not_posting_to_a_network() -> None:
    model = PermissionRiskModel()
    said = "post a 25000-cent consulting invoice, debit Accounts Receivable"
    assert (
        model._detect_modality(said, "", effect_scope="sandboxed_compute")
        != "network_write"
    )


def test_the_same_words_are_still_refused_where_they_could_be_true() -> None:
    """The word is not banned. The impossible reading of it is."""

    model = PermissionRiskModel()
    said = "post a 25000-cent consulting invoice, debit Accounts Receivable"
    assert (
        model._detect_modality(said, "", effect_scope="external_io")
        == "network_write"
    )
    assert not model._check_modality("network_write")


def test_genuine_effects_are_untouched() -> None:
    model = PermissionRiskModel()
    for said, scope, wanted in (
        ("send an email to the team", "external_io", "network_write"),
        ("upload the file to the server", "external_io", "network_write"),
        ("delete every file in the folder", "privileged_mutation", "file_delete"),
        ("search the web for prices", "read_only", "network_read"),
    ):
        assert model._detect_modality(said, "", effect_scope=scope) == wanted


def test_an_unknown_scope_changes_nothing() -> None:
    """Only scopes whose reach is actually known narrow anything.

    A scope this table does not list must behave exactly as before, or the fix
    becomes a quiet permission grant for everything nobody remembered to name.
    """

    model = PermissionRiskModel()
    said = "post a 25000-cent consulting invoice"
    assert model._detect_modality(said, "") == "network_write"
    assert (
        model._detect_modality(said, "", effect_scope="something_not_listed")
        == "network_write"
    )


def test_nothing_above_sandboxed_compute_was_widened() -> None:
    """The table only names scopes that genuinely cannot reach."""

    from core.capabilities.permission_model import PermissionRiskModel as Model

    listed = set(Model._REACHABLE_BY_SCOPE)
    assert listed <= {
        "status",
        "read_only",
        "pure_compute",
        "sandboxed_compute",
        "read_write_artifacts",
    }
    for scope in listed:
        reach = Model._REACHABLE_BY_SCOPE[scope]
        assert "network_write" not in reach
        assert "email" not in reach
        assert "file_delete" not in reach
        assert "cloud_write" not in reach

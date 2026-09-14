"""Nominal receiver types take precedence over path-like variable spelling."""

import ast

from tools.lint_governance import EffectVisitor


def _calls(source):
    visitor = EffectVisitor(relative_path="core/example.py")
    visitor.visit(ast.parse(source))
    return [(kind, callee) for kind, callee, _ in visitor.calls]


def test_typed_receipt_open_is_not_a_file_open():
    assert _calls('''
from core.cognition.outcome_ledger import OutcomeLedger as Ledger
def execute(*, ledger: Ledger):
    ledger.open(action, score, context={})
''') == []


def test_path_type_is_detected_even_with_a_non_path_name():
    assert _calls('''
from pathlib import Path as LocalPath
def execute(account: LocalPath):
    account.open("wb")
''') == [("raw_file_mutation", "account.open")]


def test_unknown_ledger_and_rebound_parameter_remain_conservative():
    source = '''
from core.cognition.outcome_ledger import OutcomeLedger
def execute(ledger: OutcomeLedger):
    ledger.open(action, score)
    ledger = unknown_source()
    ledger.open(mode)
def other(ledger):
    ledger.open(mode)
'''
    assert _calls(source) == [("raw_file_mutation", "ledger.open")] * 2


def test_nested_parameter_does_not_inherit_outer_receiver_type():
    assert _calls('''
from core.cognition.outcome_ledger import OutcomeLedger
def execute(ledger: OutcomeLedger):
    def nested(ledger):
        ledger.open("wb")
    ledger.open(action, score)
''') == [("raw_file_mutation", "ledger.open")]


def test_loop_binding_invalidates_the_parameter_type():
    assert _calls('''
from core.cognition.outcome_ledger import OutcomeLedger
def execute(ledger: OutcomeLedger):
    for ledger in unknown_paths:
        ledger.open("wb")
''') == [("raw_file_mutation", "ledger.open")]


def test_unknown_annotation_does_not_grant_an_exemption():
    assert _calls('''
from external.paths import CustomPath
def execute(ledger: CustomPath):
    ledger.open("wb")
''') == [("raw_file_mutation", "ledger.open")]


def test_variadic_parameters_shadow_outer_receiver_types():
    assert _calls('''
from core.cognition.outcome_ledger import OutcomeLedger
def execute(ledger: OutcomeLedger):
    def positional(*ledger):
        ledger.open("wb")
    def keywords(**ledger):
        ledger.open("wb")
''') == [("raw_file_mutation", "ledger.open")] * 2


def test_imports_replace_receiver_annotations():
    assert _calls('''
from core.cognition.outcome_ledger import OutcomeLedger
def execute(ledger: OutcomeLedger):
    from pathlib import Path as ledger
    ledger.open("wb")
''') == [("raw_file_mutation", "pathlib.Path.open")]

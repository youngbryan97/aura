"""The completion tracker reads its expressions; it does not run them.

288 expressions live in `config/isc_completion_evidence.json` and used to be
handed to `eval`, which made a JSON file executable by anyone who could edit
it. The reader offers the grammar those 288 use — subscripts, comparisons,
boolean operators, comprehensions, and a fixed list of pure builtins — and
refuses everything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.report_expression import (
    CALLABLES,
    METHODS,
    ExpressionRefused,
    evaluate_report_expression,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "config" / "isc_completion_evidence.json"

REPORT = {
    "synergy": [{"null_draws": 1000, "interaction_gain": 0.2},
                {"null_draws": 2000, "interaction_gain": 0.1}],
    "lesion": {"intact": {"spread": 2.0}, "lesioned": {"spread": 1.0}, "deficit": True},
    "recording": {"frames": 240, "flat_column_names": ["a", "b"]},
    "campaign": {"mind": "deterministic kind", "commit": "0" * 40, "tree_hash": "abc"},
    "per_condition": {"per_condition": {"a": {"one_component": True, "edges": 3},
                                        "b": {"one_component": False, "edges": 4}}},
}


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("float(report['recording']['frames']) > 0", True),
        ("len(report['recording']['flat_column_names']) > 0", True),
        ("all(int(r['null_draws']) >= 1000 for r in report['synergy'])", True),
        ("all(int(r['null_draws']) >= 2000 for r in report['synergy'])", False),
        ("any('interaction_gain' in r for r in report['synergy'])", True),
        ("'kind' in report['campaign']['mind'] and 'deterministic' in report['campaign']['mind']", True),
        ("len(report['campaign']['commit']) == 40 and len(report['campaign']['tree_hash']) > 0", True),
        ("float(report['lesion']['lesioned']['spread']) < float(report['lesion']['intact']['spread'])", True),
        ("bool(report['lesion']['deficit'])", True),
        ("report['lesion'].get('missing') is None", True),
        ("len([c for c in report['per_condition']['per_condition'].values() if c['one_component']]) >= 1", True),
        ("len(set(c['edges'] for c in report['per_condition']['per_condition'].values())) > 1", True),
        ("not report['lesion']['deficit']", False),
        ("report['recording']['frames'] - 40 == 200", True),
        ("'ok' if report['lesion']['deficit'] else 'no'", "ok"),
        ("{k for k, v in report['per_condition']['per_condition'].items() if v['edges'] > 3} == {'b'}", True),
    ],
)
def test_the_grammar_the_evidence_file_uses(expr, expected):
    assert evaluate_report_expression(expr, REPORT) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "report.__class__",
        "[x for x in ().__class__.__base__.__subclasses__()]",
        "eval('1+1')",
        "exec('a=1')",
        "getattr(report, 'keys')()",
        "report['lesion'] if False else __import__('sys')",
        "len(report, x=1)",
        "lambda: 1",
    ],
)
def test_anything_outside_the_grammar_is_refused(expr):
    with pytest.raises(ExpressionRefused):
        evaluate_report_expression(expr, REPORT)


def test_a_missing_key_is_a_failed_check_not_a_refusal():
    """The difference matters: one is an item not done, the other a broken check."""
    with pytest.raises(KeyError):
        evaluate_report_expression("report['nothing']['here']", REPORT)


def test_nothing_it_can_call_can_reach_the_interpreter():
    for name, function in CALLABLES.items():
        assert getattr(function, "__module__", "builtins") == "builtins", name
    assert "__import__" not in CALLABLES
    assert not any(name.startswith("_") for name in CALLABLES)
    assert not any(name.startswith("_") for name in METHODS)


def test_every_expression_in_the_evidence_file_is_inside_the_grammar():
    """The reader has to cover what is actually stored, not a subset of it."""
    evidence = json.loads(EVIDENCE.read_text())
    exprs = [
        check["expr"]
        for item in evidence.values()
        if isinstance(item, dict)
        for check in (item.get("checks") or [])
        if check.get("kind") == "report"
    ]
    assert exprs, "the evidence file holds no report checks to cover"
    refused = []
    for expr in exprs:
        try:
            evaluate_report_expression(expr, REPORT)
        except ExpressionRefused as exc:
            refused.append(f"{expr}: {exc}")
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            pass  # the sample report does not hold it; the expression still read
    assert not refused, refused

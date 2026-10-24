"""Equal arity does not preserve keyword calls, methods or exported bindings."""
from __future__ import annotations

import pytest

from core.self_modification.formal_verifier import verify_mutation
from core.self_modification.public_contract import public_contract_violations
from core.self_modification.safe_pipeline import SafePipeline


@pytest.mark.parametrize(("before", "after"), [
    ("def f(*, option=1): pass", "def f(*, renamed=1): pass"),
    ("def f(x, /): pass", "def f(*, x): pass"),
    ("def f(x=1): pass", "def f(x): pass"),
    ("def f(**options): pass", "def f(*options): pass"),
    ("async def f(x): pass", "def f(x): pass"),
    ("class C:\n def run(self, x): pass", "class C:\n def run(self, *, x): pass"),
    ("class C:\n def __init__(self, x): pass", "class C:\n def __init__(self): pass"),
    ("class C:\n def __getitem__(self, key): pass", "class C:\n def __getitem__(self): pass"),
    ("class C(metaclass=Meta): pass", "class C: pass"),
    ("__all__=['VERSION']\nVERSION=1", "__all__=[]"),
    ("__all__: list[str]=['VERSION']\nVERSION=1", "__all__: list[str]=[]"),
    ("def f(x): pass", "__all__=['f']\nclass f: pass"),
    ("class C:\n @property\n def value(self): return 1\n @value.setter\n def value(self, x): pass",
     "class C:\n @property\n def value(self, extra): return 1\n @value.setter\n def value(self, x): pass"),
])
def test_generated_and_formal_checks_reject_interface_breaks(before, after):
    assert public_contract_violations(before, after)
    with pytest.raises(AssertionError):
        exec(compile(SafePipeline._generate_tests("example.py", before, after), "generated", "exec"), {})
    assert not verify_mutation(file_path="example.py", before_source=before, after_source=after).ok


def test_a_nested_same_named_function_cannot_hide_an_exported_signature_change():
    before = "def f(x): pass\ndef other():\n def f(x): pass"
    after = "def f(*, x): pass\ndef other():\n def f(x): pass"
    assert public_contract_violations(before, after) == ["interface_changed:f"]


def test_implementation_changes_and_additional_names_keep_the_contract():
    before = "def f(x=1): return x + 1"
    after = "def f(x=1): return x + 2\ndef added(): return 3"
    assert public_contract_violations(before, after) == []
    exec(compile(SafePipeline._generate_tests("example.py", before, after), "generated", "exec"), {})


def test_unresolvable_exports_do_not_pass_static_validation():
    assert public_contract_violations("__all__ = names()", "pass")

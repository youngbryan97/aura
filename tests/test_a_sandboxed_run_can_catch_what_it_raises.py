"""Code run for her can catch the exceptions it raises.

LIVE 17 November 2026: asked whether 2**61 - 1 is prime, her code tried sympy and fell back on ImportError; the
runner's builtins had no ImportError, and the run ended in NameError("name 'ImportError' is not defined").
"""
from __future__ import annotations

import pytest

from core.sandbox.runner import run_untrusted


@pytest.mark.parametrize("caught", ["ImportError", "ValueError", "ZeroDivisionError", "KeyError", "OverflowError"])
def test_each_builtin_exception_can_be_caught(caught):
    raised = {"ImportError": "import sympy", "ValueError": "int('x')", "ZeroDivisionError": "1 / 0",
              "KeyError": "{}['k']", "OverflowError": "2.0 ** 100000"}[caught]
    result = run_untrusted(f"try:\n    {raised}\nexcept {caught}:\n    print('caught')\n", timeout=20)
    assert result.get("status") == "ok" and result.get("stdout") == "caught\n", result


def test_the_fallback_she_wrote_computes():
    code = ("try:\n    import sympy\n    print(sympy.isprime(2**61 - 1))\nexcept ImportError:\n    n = 2**61 - 1\n"
            "    print(pow(3, n - 1, n) == 1)\n")
    result = run_untrusted(code, timeout=20)
    assert result.get("status") == "ok" and result.get("stdout") == "True\n", result

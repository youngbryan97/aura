"""
Pytest plugin that captures and prints the actual measured values from every test.

When run with: pytest --capture-results
Each test's key measurements are printed after the test name.
"""
import pytest
import json
import time
from pathlib import Path

# Global results collector
_captured_results = {}


def pytest_addoption(parser):
    parser.addoption("--capture-results", action="store_true", default=False,
                     help="Capture and print measured values from every test")


@pytest.fixture(autouse=True)
def _result_capture(request):
    """Fixture that provides a `report` function tests can use to log measured values."""
    measurements = {}

    def report(**kwargs):
        measurements.update(kwargs)

    request.node._measurements = measurements
    request.node.report = report
    yield
    if measurements and request.config.getoption("--capture-results", default=False):
        test_name = request.node.nodeid
        _captured_results[test_name] = measurements
        vals = "  ".join(f"{k}={v}" for k, v in measurements.items())
        print(f"\n    📊 {vals}")


def pytest_terminal_summary(terminalreporter, config):
    if not config.getoption("--capture-results", default=False):
        return
    if not _captured_results:
        return

    output_path = Path("tests/MEASURED_VALUES.json")
    with open(output_path, "w") as f:
        json.dump(_captured_results, f, indent=2, default=str)
    terminalreporter.write_sep("=", f"Measured values written to {output_path}")

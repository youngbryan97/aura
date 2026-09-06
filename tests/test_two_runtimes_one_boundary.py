"""Aura has two runtimes. They must answer the same six questions.

Before this, ``AuraKernel`` and ``RobustOrchestrator`` shared exactly one
public method — ``stop`` — and it did not mean the same thing on both: on the
kernel it clears a flag and cancels background tasks, on the orchestrator it
brings the process down. Nothing could substitute one for the other, and no
test could ask either of them what a runtime is for.
"""
from __future__ import annotations

import inspect

import pytest

from core.runtime.what_a_runtime_is import (
    THE_OPERATIONS,
    AuraRuntime,
    the_services_it_can_address,
    what_is_missing_from,
)

RUNTIMES = ("core.kernel.aura_kernel:AuraKernel",
            "core.orchestrator.main:RobustOrchestrator")


def _load(address: str):
    module_name, class_name = address.split(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


@pytest.mark.parametrize("address", RUNTIMES)
def test_both_runtimes_offer_the_whole_boundary(address):
    runtime = _load(address)
    missing = what_is_missing_from(runtime)
    assert missing == [], f"{address} does not offer: {missing}"


@pytest.mark.parametrize("address", RUNTIMES)
def test_both_runtimes_satisfy_the_protocol(address):
    """The isinstance check, which is what substitutability actually means."""
    runtime = object.__new__(_load(address))
    assert isinstance(runtime, AuraRuntime)


@pytest.mark.parametrize("address", RUNTIMES)
def test_the_lifecycle_and_message_operations_are_awaitable(address):
    runtime = _load(address)
    for operation in ("runtime_start", "runtime_stop", "runtime_deliver",
                      "runtime_watch"):
        assert inspect.iscoroutinefunction(getattr(runtime, operation)), (
            f"{address}.{operation} must be awaitable"
        )


@pytest.mark.parametrize("address", RUNTIMES)
def test_the_read_operations_are_not_awaitable(address):
    """Reading state or resolving a service must not need a loop.

    Health reporting and shutdown paths call both from synchronous code.
    """
    runtime = _load(address)
    for operation in ("runtime_state", "runtime_service"):
        assert not inspect.iscoroutinefunction(getattr(runtime, operation))


@pytest.mark.parametrize("address", RUNTIMES)
def test_deliver_takes_an_origin_by_keyword(address):
    """Where a message came from decides how it is treated, on both."""
    signature = inspect.signature(_load(address).runtime_deliver)
    origin = signature.parameters["origin"]
    assert origin.kind is inspect.Parameter.KEYWORD_ONLY
    assert origin.default == "user"


def test_the_two_runtimes_now_share_more_than_stop():
    """The finding, pinned. One shared method is not a boundary."""
    kernel, orchestrator = (_load(a) for a in RUNTIMES)
    shared = {
        name
        for name in dir(kernel)
        if not name.startswith("_")
        and callable(getattr(kernel, name, None))
        and callable(getattr(orchestrator, name, None))
    }
    assert set(THE_OPERATIONS) <= shared
    assert len(shared) > 1


def test_every_operation_says_what_it_is_for():
    """A protocol whose operations are unexplained is a naming convention."""
    for name, purpose in THE_OPERATIONS.items():
        assert name.startswith("runtime_")
        assert ":" in purpose, f"{name} does not name its category"


def test_the_missing_report_names_what_is_missing():
    class NotARuntime:
        def runtime_state(self):
            return None

    missing = what_is_missing_from(NotARuntime())
    assert "runtime_state" not in missing
    assert "runtime_deliver" in missing
    assert len(missing) == len(THE_OPERATIONS) - 1


# ------------------------------------------------------------- ownership


def test_the_ownership_report_names_the_spine():
    """Every declared service name is either addressable or named as debt."""
    class Nothing:
        def runtime_service(self, name, default=None):
            return default

    report = the_services_it_can_address(Nothing())
    assert report["declared"] > 0
    assert report["addressable"] == 0
    assert len(report["unaddressable"]) == report["declared"]


def test_a_runtime_that_resolves_everything_reports_no_debt():
    class Everything:
        def runtime_service(self, name, default=None):
            return object()

    report = the_services_it_can_address(Everything())
    assert report["unaddressable"] == []
    assert report["addressable"] == report["declared"]


def test_a_runtime_without_the_operation_is_not_silently_zero():
    """Absent is a different answer from "resolved nothing"."""
    report = the_services_it_can_address(object())
    assert report["error"] == "runtime has no runtime_service"


def test_a_resolver_that_raises_counts_the_name_as_unaddressable():
    class Angry:
        def runtime_service(self, name, default=None):
            raise RuntimeError("no")

    report = the_services_it_can_address(Angry())
    assert report["addressable"] == 0

"""Internal PCI: LZ76 contracts, probe governance, and service integration."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

import core.consciousness.whole_system_phi_service as phi_service_mod
import core.governance.will as will_mod
from core.consciousness.perturbational_probe import (
    PerturbationalProbe,
    ProbeCampaignReport,
    ReversiblePerturbation,
    lz76_complexity,
    normalized_lz,
    pci_from_windows,
)
from core.consciousness.whole_system_phi_service import WholeSystemPhiService

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# LZ76
# ─────────────────────────────────────────────────────────────────────────────

def test_lz76_known_values():
    assert lz76_complexity("") == 0
    assert lz76_complexity("0000000000") <= 2       # constant → minimal phrases
    assert lz76_complexity("0101010101") <= 4       # periodic → small
    rng = np.random.default_rng(5)
    rand = "".join(str(int(b)) for b in rng.integers(0, 2, 400))
    assert lz76_complexity(rand) > lz76_complexity("01" * 200)


def test_normalized_lz_ranges():
    rng = np.random.default_rng(9)
    quiet = np.zeros((60, 8), dtype=bool)
    rich = rng.integers(0, 2, (60, 8)).astype(bool)
    assert normalized_lz(quiet) == 0.0
    assert 0.4 < normalized_lz(rich) <= 1.6
    assert normalized_lz(rich) > normalized_lz(quiet)


def _chaotic_response(T: int, n: int, seed: int) -> np.ndarray:
    """A large, aperiodic, spatiotemporally differentiated transient — a
    faithful synthetic of an integrated network's evoked response (each
    channel driven by a chaotic logistic map, so it is neither constant nor
    periodic)."""
    rng = np.random.default_rng(seed)
    out = 0.1 * rng.standard_normal((T, n))
    x = np.linspace(0.1, 0.9, n)
    for t in range(T):
        x = 3.9 * x * (1.0 - x)
        out[t] += 3.0 * (x - 0.5)
    return out


def test_pci_separates_response_from_quiet_baseline():
    rng = np.random.default_rng(13)
    baseline = 0.1 * rng.standard_normal((90, 10))
    response = _chaotic_response(90, 10, seed=1)
    out = pci_from_windows(baseline, response)
    # a genuinely complex evoked response beats the thresholded-baseline
    # control on BOTH the per-bit LZc and the robust engaged-complexity metric
    assert out["pci"] > out["pci_baseline_control"]
    assert out["evoked_complexity"] > 5.0
    assert out["active_fraction"] > 0.2
    assert out["estimator"].startswith("internal_pci")


def test_pci_scores_stereotyped_max_response_low():
    """The seizure/anesthesia signature: a huge but STEREOTYPED response is
    low-complexity — the measurement must not mistake amplitude for
    integration."""
    rng = np.random.default_rng(17)
    baseline = 0.1 * rng.standard_normal((90, 10))
    stereotyped = np.full((90, 10), 5.0)          # everything on, held constant
    differentiated = _chaotic_response(90, 10, seed=2)
    stereo = pci_from_windows(baseline, stereotyped)
    diff = pci_from_windows(baseline, differentiated)
    # stereotyped is maximally active yet minimally complex — low on both
    assert stereo["active_fraction"] >= diff["active_fraction"]
    assert stereo["pci"] < diff["pci"]
    assert stereo["evoked_complexity"] < diff["evoked_complexity"]


# ─────────────────────────────────────────────────────────────────────────────
# The governed probe
# ─────────────────────────────────────────────────────────────────────────────

class _SyntheticOrganism:
    """Quiet noise until perturbed; then a DIFFERENTIATED transient — each
    channel rings at its own frequency (a richly connected network's evoked
    response), which is what registers as high complexity."""

    def __init__(self, n: int = 8, seed: int = 3):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.echo = 0.0
        self._x = np.linspace(0.1, 0.9, n)

    def sample(self) -> dict[str, float]:
        base = 0.05 * self.rng.standard_normal(self.n)
        if self.echo > 0.01:
            self._x = 3.9 * self._x * (1.0 - self._x)   # chaotic, differentiated
            base += self.echo * (self._x - 0.5)
            self.echo *= 0.95
        return {f"c{i}": float(v) for i, v in enumerate(base)}

    def perturb(self) -> bool:
        self.echo = 3.0
        self._x = np.linspace(0.1, 0.9, self.n)
        return True


def _approving_will(monkeypatch, *, approve: bool = True, reason: str = "ok"):
    class _Decision:
        receipt_id = "probe-receipt-1"

        def __init__(self):
            self.reason = reason

        def is_approved(self):
            return approve

    monkeypatch.setattr(will_mod, "get_will",
                        lambda: type("W", (), {"decide": lambda self, **k: _Decision()})())


def test_probe_runs_and_detects_propagation(monkeypatch):
    _approving_will(monkeypatch)
    org = _SyntheticOrganism()
    probe = PerturbationalProbe(sampler=org.sample, perturb=org.perturb,
                                sleep=lambda s: None)
    report = probe.run(n_baseline=60, n_response=60, interval_s=0.0)
    assert report.ran
    assert report.will_receipt_id == "probe-receipt-1"
    assert report.pci["evoked_complexity"] > report.sham_pci["evoked_complexity"]
    assert report.pci["active_fraction"] > report.sham_pci["active_fraction"]
    assert len(report.transitions) == 1
    assert len(report.transitions[0][0]) == 8


def test_probe_respects_will_refusal(monkeypatch):
    _approving_will(monkeypatch, approve=False, reason="covenant holds")
    org = _SyntheticOrganism()
    called = {"n": 0}

    def perturb():
        called["n"] += 1
        return True

    probe = PerturbationalProbe(sampler=org.sample, perturb=perturb,
                                sleep=lambda s: None)
    report = probe.run(interval_s=0.0)
    assert not report.ran
    assert "refused" in report.reason
    assert called["n"] == 0  # a refused probe never touches the organism


def test_probe_campaign_requires_repeatable_effect_and_recovery(monkeypatch):
    _approving_will(monkeypatch)
    org = _SyntheticOrganism()

    def perturb():
        org.perturb()

        def restore():
            org.echo = 0.0
            return True

        return ReversiblePerturbation(True, restore, "synthetic.reversible")

    probe = PerturbationalProbe(
        sampler=org.sample,
        perturb=perturb,
        sleep=lambda seconds: None,
    )
    campaign = probe.run_campaign(
        trials=3,
        n_baseline=60,
        n_response=60,
        n_recovery=40,
        interval_s=0.0,
    )
    assert campaign.trials_completed == 3
    assert len(campaign.transitions) == 3
    assert campaign.channel_names == tuple(f"c{index}" for index in range(8))
    assert campaign.aggregate["evoked_complexity_delta_ci_5"] > 0.0
    assert campaign.aggregate["recovered_trials"] == 3
    assert campaign.aggregate["causal_response_established"]
    assert all(report.recovery["restore_succeeded"] for report in campaign.reports)


def test_probe_fails_closed_without_a_will():
    org = _SyntheticOrganism()
    probe = PerturbationalProbe(sampler=org.sample, perturb=org.perturb,
                                sleep=lambda s: None)
    # No monkeypatch: the detached test Will refuses consequential state
    # mutation or is unavailable — either way the probe must not run wild.
    report = probe.run(interval_s=0.0)
    if report.ran:  # a permissive dev-mode Will is acceptable…
        assert report.pci  # …but only with a complete report
    else:
        assert "refused" in report.reason or "unavailable" in report.reason


# ─────────────────────────────────────────────────────────────────────────────
# Service integration
# ─────────────────────────────────────────────────────────────────────────────

def _feed_ring(service: WholeSystemPhiService, T: int, n: int = 6,
               seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    X = np.zeros((T, n))
    A = 0.2 * np.eye(n)
    for i in range(n):
        A[i, (i + 1) % n] = 0.55
    for t in range(1, T):
        X[t] = A @ X[t - 1] + rng.standard_normal(n)
    for t in range(T):
        service.observe({f"ch{i}": float(X[t, i]) for i in range(n)})


def test_service_estimates_after_window_fills(monkeypatch):
    monkeypatch.setenv("AURA_WSPHI_MIN_SAMPLES", "300")
    monkeypatch.setenv("AURA_WSPHI_ESTIMATE_EVERY", "300")
    # A claim needs a null that can resolve it. At the service's cheap default
    # the p-floor is 1/(N+1) ≈ 0.048 — too coarse to establish anything, which
    # is now enforced rather than assumed. This test asserts a POSITIVE verdict
    # on a genuinely coupled ring, so it must pay for the resolution to make one.
    monkeypatch.setenv("AURA_WSPHI_SURROGATES", "120")
    monkeypatch.setenv("AURA_WSPHI_BOOTSTRAPS", "120")
    service = WholeSystemPhiService()
    _feed_ring(service, 320)
    assert service.ready()
    est = service.maybe_estimate()
    assert est is not None
    assert est.surrogate_resolution_sufficient()
    assert est.integration_established()
    status = service.status()
    assert status["estimates_done"] == 1
    assert status["latest"]["claim"]
    # not due again until another window's worth arrives
    assert service.maybe_estimate() is None


def test_service_ignores_junk_observations():
    service = WholeSystemPhiService()
    service.observe({"a": float("nan"), "b": 1.0})     # <2 clean channels
    service.observe({})
    assert service.status()["window"] == 0


def test_service_interpolates_sparse_samples_instead_of_injecting_zero():
    service = WholeSystemPhiService()
    for index in range(10):
        row = {
            "a": float(index),
            "b": float((index + 1) * 10),
            "c": float(index * 2),
        }
        if index == 5:
            row.pop("b")
        service.observe(row)
    matrix, names, diagnostics = service._matrix()
    b_index = names.index("b")
    assert matrix[5, b_index] == pytest.approx(60.0)
    assert diagnostics["missing_values_interpolated"] == 1
    assert diagnostics["missing_by_channel"]["b"] == 1


@pytest.mark.asyncio
async def test_service_schedules_estimation_without_blocking_the_caller(monkeypatch):
    monkeypatch.setenv("AURA_WSPHI_MIN_SAMPLES", "16")
    monkeypatch.setenv("AURA_WSPHI_ESTIMATE_EVERY", "16")
    service = WholeSystemPhiService()
    for index in range(20):
        service.observe({"a": index, "b": index + 1, "c": index + 2, "d": index + 3})

    class _Decision:
        admitted = True
        lease_id = "phi-lease"
        reason = "capacity_available"

    class _Admission:
        def __init__(self):
            self.released = False

        async def acquire(self, request):
            return _Decision()

        async def release(self, lease_id, *, reason):
            self.released = lease_id == "phi-lease"

    admission = _Admission()
    monkeypatch.setattr(
        phi_service_mod,
        "optional_service",
        lambda name, *aliases, default=None: admission
        if name == "resource_admission" else default,
    )

    def deliberately_slow_estimate():
        time.sleep(0.20)
        return None

    monkeypatch.setattr(service, "maybe_estimate", deliberately_slow_estimate)
    started = time.perf_counter()
    assert service.schedule_if_due()
    elapsed = time.perf_counter() - started
    task = service._estimate_task
    assert elapsed < 0.05
    assert task is not None
    await asyncio.sleep(0.02)
    assert not task.done()
    await task
    assert admission.released
    assert not service.status()["estimate_task_active"]


@pytest.mark.asyncio
async def test_service_runs_live_probe_campaign_and_keeps_named_rows(monkeypatch):
    monkeypatch.setenv("AURA_WSPHI_PROBE_ENABLED", "true")
    monkeypatch.setenv("AURA_WSPHI_PROBE_INITIAL_DELAY_S", "0")
    service = WholeSystemPhiService()
    service._estimates_done = 1
    service._created_at = time.time() - 10.0

    class _Decision:
        admitted = True
        lease_id = "probe-lease"
        reason = "capacity_available"

    class _Admission:
        def __init__(self):
            self.released = False

        async def acquire(self, request):
            return _Decision()

        async def release(self, lease_id, *, reason):
            self.released = lease_id == "probe-lease"

    admission = _Admission()
    monkeypatch.setattr(
        phi_service_mod,
        "optional_service",
        lambda name, *aliases, default=None: admission
        if name == "resource_admission" else default,
    )
    names = ("a", "b", "c", "d")
    campaign = ProbeCampaignReport(
        trials_requested=3,
        trials_completed=3,
        reports=[],
        aggregate={"causal_response_established": True},
        channel_names=names,
        transitions=[((0, 0, 0, 0), (1, 1, 1, 1))] * 3,
    )
    monkeypatch.setattr(
        PerturbationalProbe,
        "run_campaign",
        lambda self, **kwargs: campaign,
    )

    assert service.schedule_probe_if_due()
    task = service._probe_task
    assert task is not None
    await task

    status = service.status()
    assert admission.released
    assert status["interventional_rows"] == 3
    assert status["latest_probe"]["aggregate"]["causal_response_established"]
    assert service._interventional[0].channel_names == names


def test_service_carries_interventional_rows_into_estimates(monkeypatch):
    """Named interventional rows must actually REACH the estimator.

    This previously asserted ``n_interventional_transitions >= 0``, which is
    true of every possible input — including the real failure, where all rows
    were rejected at projection and the count was 0. It also hid behind
    ``if est.exact_macro:``, so it could skip entirely. A test that cannot fail
    is how a whole probe campaign contributed zero interventions to an artifact
    named "estimate_with_interventions" without anyone noticing.
    """
    monkeypatch.setenv("AURA_WSPHI_MIN_SAMPLES", "300")
    monkeypatch.setenv("AURA_WSPHI_ESTIMATE_EVERY", "300")
    service = WholeSystemPhiService()
    names = tuple(f"ch{i}" for i in range(6))
    service.add_interventional_transitions(
        [((0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0))],
        probe_report={"pci": 0.4},
        channel_names=names,
    )
    _feed_ring(service, 320)
    est = service.maybe_estimate()
    assert est is not None
    assert est.exact_macro, "no discrete estimate was produced"
    assert est.exact_macro["n_interventional_transitions"] > 0, (
        "interventional rows never reached the estimator — they were rejected "
        f"at projection ({est.exact_macro.get('n_projection_rejected_transitions')} "
        "rejected)"
    )
    assert est.exact_macro["n_projection_rejected_transitions"] == 0
    assert service.status()["latest_probe"]["pci"] == 0.4


def test_anonymous_interventional_rows_are_refused(monkeypatch):
    """Unnamed rows must be dropped at the door, not misread downstream.

    In the checked-in campaign 13-bit anonymous rows met 8 retained channels and
    were rejected at projection. But when the arity coincidentally matches (as
    here: 6 rows, 6 live channels) an unnamed row is NOT rejected — it is read
    positionally against whatever channels survived, silently attributing an
    intervention on one channel to another. That is fabricated causal evidence,
    so the rows are refused outright.
    """
    monkeypatch.setenv("AURA_WSPHI_MIN_SAMPLES", "300")
    monkeypatch.setenv("AURA_WSPHI_ESTIMATE_EVERY", "300")
    service = WholeSystemPhiService()
    service.add_interventional_transitions(
        [((0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0))],
        probe_report={"pci": 0.4},
    )
    assert service.status()["interventional_rows"] == 0, (
        "an unnamed interventional row was accepted; if its arity matches the "
        "retained channels it will be misaligned rather than rejected"
    )

    _feed_ring(service, 320)
    est = service.maybe_estimate()
    assert est is not None
    assert est.exact_macro
    assert est.exact_macro["n_interventional_transitions"] == 0
    # The probe report itself is still recorded — refusing the rows must not
    # lose the trial's other evidence.
    assert service.status()["latest_probe"]["pci"] == 0.4


def test_measure_tool_passes_channel_names_to_the_estimator():
    """The one-line wiring defect that emptied the campaign.

    tools/measure_whole_system_phi.py called add_interventional_transitions
    without channel_names, so all five probe trials were projection-rejected.
    """
    import inspect

    from tools import measure_whole_system_phi

    src = inspect.getsource(measure_whole_system_phi)
    call_start = src.find("host.service.add_interventional_transitions")
    assert call_start != -1, "call site not found — update this test"
    call = src[call_start : call_start + 400]
    assert "channel_names=" in call, (
        "measure_whole_system_phi calls add_interventional_transitions without "
        "channel_names — every interventional row will be rejected at projection"
    )


@pytest.mark.asyncio
async def test_service_persists_report_through_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_WSPHI_MIN_SAMPLES", "300")
    monkeypatch.setenv("AURA_WSPHI_ESTIMATE_EVERY", "300")
    monkeypatch.setenv("AURA_PHI_DIR", str(tmp_path / "phi"))
    service = WholeSystemPhiService()
    _feed_ring(service, 320)
    assert service.maybe_estimate() is not None
    path = await service.persist_latest()
    assert path.endswith("whole_system_latest.json")
    import json

    payload = json.loads((tmp_path / "phi" / "whole_system_latest.json"
                          ).read_text())["payload"]
    assert payload["estimate"]["estimator"].startswith(
        "gaussian_stochastic_interaction")
    assert "not a consciousness meter" in payload["estimate"]["claim"]


@pytest.mark.asyncio
async def test_report_persistence_renews_its_grant_after_directory_io(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    monkeypatch.setenv("AURA_PHI_DIR", str(tmp_path / "phi"))
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    service = WholeSystemPhiService()
    service._latest = SimpleNamespace(to_dict=lambda: {"phi": None, "claim": "synthetic persistence fixture"})
    service._latest_probe = {"status": "unmeasured"}
    gateway = get_file_write_gateway()
    ensure = gateway.ensure_directory_async
    expired = []

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        result = await ensure(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        expired.append(token.lease_id)
        return result

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    path = await service.persist_latest()
    assert path == str(tmp_path / "phi" / "whole_system_latest.json")
    payload = json.loads((tmp_path / "phi" / "whole_system_latest.json").read_text())["payload"]
    assert expired and payload == {"estimate": service._latest.to_dict(), "probe": {"status": "unmeasured"}}


def test_sample_runtime_channels_survives_a_bare_container():
    service = WholeSystemPhiService()
    channels = service.sample_runtime_channels()
    # in a bare test process only the body channels are guaranteed
    assert isinstance(channels, dict)
    for value in channels.values():
        assert value == value  # no NaNs

"""A shed that protects the foreground must not unload the foreground's lane.

The 2026-07-25 probe loaded models **55 times across 30 turns** and shed the
fallback ladder five times, each shed logged as
``unloading Qwen2.5-7B-Instruct-4bit to protect the foreground lane
(protected_foreground_shed)``.

The sequence: a protected turn arrives, the cortex begins a 20GB load, free
memory dips under the shed threshold *because of that load*, and the shed then
unloads the Brainstem and Reflex — the only lanes that can answer while the
cortex warms. Shedding the parachute to make the plane lighter.

Every reload then pays full load latency and contends with the cortex for the
single GPU slot, which is most of a 72s p50.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit


class FakeClient:
    """A worker that actually stops when it is rebooted.

    It used to set ``rebooted`` and stay alive. A shed only counts once
    ``_worker_is_unloaded`` confirms the worker went away — "the absence of
    the check is not the check" — so this fake modelled a reboot that
    reclaimed nothing, and a test asking for the completed-action warning was
    asking for it after an incomplete action.

    ``unloads`` is here for the case that needs the other behaviour: a reboot
    that returns without stopping anything.
    """

    def __init__(self, path, *, alive=True, unloads=True):
        self.model_path = path
        self.alive = alive
        self.unloads = unloads
        self.rebooted = False

    def is_alive(self):
        return self.alive

    async def reboot_worker(self, reason="", mark_failed=False):
        self.rebooted = True
        if self.unloads:
            self.alive = False


@pytest.fixture()
def gate(monkeypatch):
    g = InferenceGate.__new__(InferenceGate)
    g._mlx_client = object()
    g._last_background_memory_shed_at = 0.0
    g._brainstem_client = FakeClient("/models/Qwen2.5-7B-Instruct-4bit")
    g._reflex_client = FakeClient("/models/Qwen2.5-1.5B-Instruct-4bit")
    # Tight enough that the shed runs (below the 24+8GB abundance guard), but
    # not so tight that the cortex itself cannot load (>= its 24GB reserve).
    # That window is precisely where preserving the ladder is the right call.
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: type("S", (), {"available_gb": 28.0})(),
        raising=False,
    )
    return g


def _run_shed(gate, clients, reason):
    import core.brain.llm.mlx_client as mlx_client

    original = dict(getattr(mlx_client, "_CLIENTS", {}))
    mlx_client._CLIENTS.clear()
    mlx_client._CLIENTS.update(clients)
    try:
        asyncio.run(
            gate._shed_background_workers_for_memory_pressure(force=True, reason=reason)
        )
    finally:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(original)


class TestTheLadderSurvivesAProtectedShed:
    def test_the_fallback_lanes_are_kept(self, gate):
        brainstem = gate._brainstem_client
        reflex = gate._reflex_client
        background = FakeClient("/models/some-background-model")

        _run_shed(
            gate,
            {
                brainstem.model_path: brainstem,
                reflex.model_path: reflex,
                background.model_path: background,
            },
            "protected_foreground_shed",
        )

        assert not brainstem.rebooted, "the Brainstem answers this very turn"
        assert not reflex.rebooted, "the Reflex is the last rung"

    def test_genuine_background_workers_are_still_shed(self, gate):
        """The shed must still do its job — this is not a disable."""
        background = FakeClient("/models/some-background-model")

        _run_shed(
            gate,
            {background.model_path: background},
            "protected_foreground_shed",
        )

        assert background.rebooted

    def test_other_shed_reasons_are_unchanged(self, gate):
        """Only foreground protection preserves the ladder; OOM relief cannot."""
        brainstem = gate._brainstem_client

        _run_shed(gate, {brainstem.model_path: brainstem}, "memory_pressure_critical")

        assert brainstem.rebooted, (
            "a real memory emergency must still be able to shed everything"
        )


class TestLadderDiscovery:
    def test_ladder_paths_come_from_the_live_clients(self, gate):
        paths = gate._fallback_ladder_paths()
        assert "/models/Qwen2.5-7B-Instruct-4bit" in paths
        assert "/models/Qwen2.5-1.5B-Instruct-4bit" in paths

    def test_env_configured_lanes_are_included(self, gate, monkeypatch):
        monkeypatch.setenv("AURA_MLX_BRAINSTEM_MODEL", "/models/custom-brainstem")
        assert "/models/custom-brainstem" in gate._fallback_ladder_paths()

    def test_a_gate_with_no_ladder_returns_empty(self):
        bare = InferenceGate.__new__(InferenceGate)
        assert bare._fallback_ladder_paths() == frozenset() or isinstance(
            bare._fallback_ladder_paths(), frozenset
        )


class TestTheLadderYieldsWhenItBlocksTheCortex:
    """Keeping the parachute is right until it is what keeps the engine off.

    Live 2026-07-25: the cortex died six times in one run against

        foreground_warmup_deferred:memory_pressure:75.6%/15.6GB
        (need <72.0% and >=20.0GB)

    on a 64GB host. Preserving the ladder unconditionally — the previous
    version of this very fix — kept several GB resident that the 20GB cortex
    load needed, so the lane the shed exists to protect could never return.

    One turn served by a lower lane costs a turn. A cortex that can never load
    costs the session.
    """

    def test_the_ladder_is_shed_when_the_cortex_cannot_load(self, gate, monkeypatch):
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 15.6})(),
            raising=False,
        )
        brainstem = gate._brainstem_client

        _run_shed(
            gate, {brainstem.model_path: brainstem}, "protected_foreground_shed"
        )

        assert brainstem.rebooted, (
            "with 15.6GB free the 20GB cortex cannot load; the ladder is the "
            "thing to spend"
        )

    def test_the_ladder_is_kept_when_the_cortex_could_load(self, gate, monkeypatch):
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 40.0})(),
            raising=False,
        )
        brainstem = gate._brainstem_client

        _run_shed(
            gate, {brainstem.model_path: brainstem}, "protected_foreground_shed"
        )

        assert not brainstem.rebooted, (
            "with headroom to spare the fallback stays resident"
        )

    def test_an_unreadable_probe_keeps_the_ladder(self, gate, monkeypatch):
        """Shedding on a guess is the failure this branch exists to prevent."""
        def _boom():
            raise RuntimeError("probe unavailable")

        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            _boom,
            raising=False,
        )
        assert gate._memory_blocks_primary_load() is False

    def test_ready_primary_never_sheds_or_warns(self, gate, monkeypatch, caplog):
        primary = FakeClient("/models/Aura-32B-cortex", alive=True)
        fallback = gate._brainstem_client
        gate._mlx_client = primary
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 10.0})(),
            raising=False,
        )

        with caplog.at_level("WARNING", logger="Aura.InferenceGate"):
            _run_shed(
                gate,
                {
                    primary.model_path: primary,
                    fallback.model_path: fallback,
                },
                "protected_foreground_shed",
            )

        assert not fallback.rebooted
        assert "memory was too short" not in caplog.text
        assert "unloading" not in caplog.text

    def test_cold_primary_uses_model_derived_threshold(self, gate, monkeypatch):
        gate._mlx_client = FakeClient("/models/Aura-32B-cortex", alive=False)
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 19.0})(),
            raising=False,
        )
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._model_load_min_available_gb",
            lambda _path: 18.0,
        )

        assert gate._memory_blocks_primary_load() is False

        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 17.0})(),
            raising=False,
        )
        assert gate._memory_blocks_primary_load() is True

    def test_no_action_warning_without_an_eligible_live_worker(
        self, gate, monkeypatch, caplog
    ):
        gate._mlx_client = FakeClient("/models/Aura-32B-cortex", alive=False)
        dead_fallback = FakeClient(
            "/models/Qwen2.5-7B-Instruct-4bit", alive=False
        )
        gate._brainstem_client = dead_fallback
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 10.0})(),
            raising=False,
        )

        with caplog.at_level("WARNING", logger="Aura.InferenceGate"):
            _run_shed(
                gate,
                {dead_fallback.model_path: dead_fallback},
                "protected_foreground_shed",
            )

        assert "memory was too short" not in caplog.text
        assert "unloading" not in caplog.text

    def test_successful_ladder_unload_emits_completed_action_warning(
        self, gate, monkeypatch, caplog
    ):
        gate._mlx_client = FakeClient("/models/Aura-32B-cortex", alive=False)
        fallback = gate._brainstem_client
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 10.0})(),
            raising=False,
        )

        with caplog.at_level("WARNING", logger="Aura.InferenceGate"):
            _run_shed(
                gate,
                {fallback.model_path: fallback},
                "protected_foreground_shed",
            )

        assert fallback.rebooted
        assert "memory was too short" in caplog.text
        assert "shed 1 live fallback worker" in caplog.text


class TestAShedOnlyCountsWhenTheWorkerWentAway:
    """"The absence of the check is not the check."""

    def test_a_reboot_that_reclaims_nothing_is_not_counted(
        self, gate, monkeypatch, caplog
    ):
        stubborn = FakeClient("/models/Qwen2.5-7B-Instruct-4bit", unloads=False)
        gate._brainstem_client = stubborn
        gate._mlx_client = FakeClient("/models/Aura-32B-cortex", alive=False)
        monkeypatch.setattr(
            "core.utils.memory_monitor.get_memory_pressure_snapshot",
            lambda: type("S", (), {"available_gb": 10.0})(),
            raising=False,
        )

        with caplog.at_level("WARNING", logger="Aura.InferenceGate"):
            _run_shed(gate, {stubborn.model_path: stubborn}, "protected_foreground_shed")

        assert stubborn.rebooted
        assert "did not report unloaded" in caplog.text
        assert "shed 1 live fallback worker" not in caplog.text

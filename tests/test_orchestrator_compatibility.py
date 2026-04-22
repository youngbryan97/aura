import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.orchestrator.mixins.boot.boot_cognitive import BootCognitiveMixin
from core.orchestrator.mixins.boot.boot_resilience import BootResilienceMixin
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin


class _BootProbe(BootCognitiveMixin):
    cognition = None


class _ResilienceProbe(BootResilienceMixin):
    _actor_bus = None
    actor_bus = None
    supervisor = None


class _OutputProbe(OutputFormatterMixin):
    def __init__(self, emitter):
        self.cognitive_engine = SimpleNamespace(_emit_thought=emitter)


@pytest.mark.asyncio
async def test_init_cognitive_core_awaits_async_setup(monkeypatch):
    setup = AsyncMock()
    cognitive_engine = SimpleNamespace(setup=setup)
    capability_engine = object()

    def _get(name, default=None):
        if name == "cognitive_engine":
            return cognitive_engine
        if name == "capability_engine":
            return capability_engine
        return default

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_cognitive.ServiceContainer.get",
        staticmethod(_get),
    )

    await _BootProbe()._init_cognitive_core()

    setup.assert_awaited_once_with(registry=capability_engine, router=capability_engine)


@pytest.mark.asyncio
async def test_emit_thought_stream_schedules_async_emitters():
    emitter = AsyncMock()
    probe = _OutputProbe(emitter)

    probe._emit_thought_stream("hello")
    await asyncio.sleep(0)

    emitter.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_start_state_vault_actor_uses_actor_bus_request_for_handshake(monkeypatch):
    started = object()

    class _Supervisor:
        def __init__(self):
            self.specs = []

        def add_actor(self, spec):
            self.specs.append(spec)

        def start_actor(self, name):
            assert name == "state_vault"
            return started

    class _Bus:
        def __init__(self):
            self.actors = {}
            self.requests = []

        def has_actor(self, name):
            return name in self.actors

        def add_actor(self, name, connection):
            self.actors[name] = connection

        async def request(self, actor, msg_type, payload, timeout=0):
            self.requests.append((actor, msg_type, payload, timeout))
            return {"type": "pong", "ts": 123.0}

    bus = _Bus()
    supervisor = _Supervisor()

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda name, default=None: bus if name == "actor_bus" else default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = supervisor

    await probe._start_state_vault_actor()

    assert bus.actors["state_vault"] is started
    assert bus.requests == [
        ("state_vault", "ping", {"source": "boot_resilience", "attempt": 1}, 2.0)
    ]


@pytest.mark.asyncio
async def test_start_state_vault_actor_fallback_ping_uses_request_wire_format(monkeypatch):
    class _Pipe:
        def __init__(self):
            self.sent = []
            self.last_request_id = None

        def send(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            self.last_request_id = payload["request_id"]

        def poll(self, timeout):
            return True

        def recv(self):
            return json.dumps(
                {
                    "response_to": self.last_request_id,
                    "payload": {"type": "pong", "ts": 456.0},
                }
            )

    pipe = _Pipe()

    class _Supervisor:
        def add_actor(self, spec):
            self.spec = spec

        def start_actor(self, name):
            assert name == "state_vault"
            return pipe

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = _Supervisor()

    await probe._start_state_vault_actor()

    assert pipe.sent
    assert pipe.sent[0]["type"] == "ping"
    assert pipe.sent[0]["is_request"] is True
    assert pipe.sent[0]["payload"]["source"] == "boot_resilience"

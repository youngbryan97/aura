from __future__ import annotations

from types import SimpleNamespace

from core.runtime.model_layers import require_model_layers, resolve_model_layers


def _layers(count: int = 3):
    return [SimpleNamespace(name=f"layer-{index}") for index in range(count)]


def test_direct_model_layers_are_resolved_with_model_as_owner():
    model = SimpleNamespace(layers=_layers())

    view = require_model_layers(model)

    assert view.owner is model
    assert view.layers is model.layers
    assert view.path == "layers"


def test_wrapped_model_layers_resolve_the_forward_owner():
    inner = SimpleNamespace(layers=_layers())
    model = SimpleNamespace(model=inner)

    view = require_model_layers(model)

    assert view.owner is inner
    assert view.layers is inner.layers
    assert view.path == "model.layers"


def test_forwarding_layers_property_does_not_hide_the_true_forward_owner():
    inner = SimpleNamespace(layers=_layers(), embed_tokens=object(), norm=object())

    class Wrapper:
        model = inner

        @property
        def layers(self):
            return self.model.layers

    model = Wrapper()

    view = require_model_layers(model)

    assert view.owner is inner
    assert view.layers is inner.layers
    assert view.path == "model.layers"


def test_transformer_backed_layout_is_supported_without_recursive_guessing():
    transformer = SimpleNamespace(layers=_layers())
    model = SimpleNamespace(transformer=transformer, vision_tower=SimpleNamespace(layers=_layers(9)))

    view = require_model_layers(model)

    assert view.owner is transformer
    assert view.layers is transformer.layers
    assert view.path == "transformer.layers"


def test_unsupported_or_empty_layout_is_explicitly_unavailable():
    assert resolve_model_layers(SimpleNamespace()) is None
    assert resolve_model_layers(SimpleNamespace(layers=[])) is None


def test_latent_bridge_attaches_to_direct_model_layout(monkeypatch):
    from core.consciousness.latent_bridge import LatentBridge, LatentReadoutHook

    vector = SimpleNamespace()

    class Library:
        vectors = {"valence": vector}

        @staticmethod
        def get_vectors_for_layer(_index):
            return {"valence": vector}

    engine = SimpleNamespace(
        _model_attached=True,
        _library=Library(),
        _model_info={"target_layers": [0, 2]},
    )
    model = SimpleNamespace(layers=_layers())
    monkeypatch.setattr(LatentReadoutHook, "install", lambda self: setattr(self, "_installed", True))

    bridge = LatentBridge(engine)

    assert bridge.attach(model) is True
    assert [hook._layer_idx for hook in bridge._readout_hooks] == [0, 2]
    assert bridge.get_status()["layer_path"] == "layers"


def test_latent_bridge_declines_unknown_layout_without_claiming_attachment():
    vector = SimpleNamespace()

    class Library:
        vectors = {"valence": vector}

        @staticmethod
        def get_vectors_for_layer(_index):
            return {"valence": vector}

    from core.consciousness.latent_bridge import LatentBridge

    bridge = LatentBridge(
        SimpleNamespace(
            _model_attached=True,
            _library=Library(),
            _model_info={"target_layers": [0]},
        )
    )

    assert bridge.attach(SimpleNamespace()) is False
    assert bridge.get_status()["attached"] is False
    assert bridge.get_status()["attachment_error"].startswith("unsupported_model_layer_topology:")


def test_neutral_steering_is_attached_but_not_injected(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.consciousness import affective_steering

    hook = SimpleNamespace(_phi_residual_channel=None)

    class Engine:
        _model_attached = True
        _hooks = [hook]
        _alpha = 0.0

        def attach(self, _model, _tokenizer):
            return None

        def start_substrate_sync(self, shared_state=None):
            return None

        def is_active(self):
            return False

    engine = Engine()
    monkeypatch.setattr(affective_steering, "get_steering_engine", lambda: engine)
    monkeypatch.setattr(mlx_worker, "record_degradation", lambda *args, **kwargs: None)
    flag = SimpleNamespace(value=True)

    attached, active = mlx_worker._attach_affective_steering(
        SimpleNamespace(), SimpleNamespace(), None, None, flag
    )

    assert attached is engine
    assert active is False
    assert flag.value is False


def test_failed_steering_attach_keeps_resident_model_available(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.consciousness import affective_steering

    class Engine:
        def attach(self, _model, _tokenizer):
            raise RuntimeError("synthetic attach failure")

    monkeypatch.setattr(affective_steering, "get_steering_engine", lambda: Engine())
    monkeypatch.setattr(mlx_worker, "record_degradation", lambda *args, **kwargs: None)
    flag = SimpleNamespace(value=True)

    attached, active = mlx_worker._attach_affective_steering(
        SimpleNamespace(), SimpleNamespace(), None, None, flag
    )

    assert attached is None
    assert active is False
    assert flag.value is False


def test_deferred_steering_does_not_start_sync_without_hooks(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.consciousness import affective_steering

    sync_calls = []

    class Engine:
        _model_attached = False
        _hooks = []
        _model_info = {"attachment_error": "steering_generation_deferred"}

        def attach(self, _model, _tokenizer, **_kwargs):
            return False

        def start_substrate_sync(self, shared_state=None):
            sync_calls.append(shared_state)

        def is_active(self):
            return False

    engine = Engine()
    monkeypatch.setattr(affective_steering, "get_steering_engine", lambda: engine)

    attached, active = mlx_worker._attach_affective_steering(
        SimpleNamespace(),
        SimpleNamespace(),
        object(),
        None,
        SimpleNamespace(value=True),
        model_path="/models/deferred",
    )

    assert attached is engine
    assert active is False
    assert sync_calls == []


def test_zero_surface_control_needs_no_steering_engine():
    from core.brain.llm.mlx_worker import (
        _apply_surface_generation_controls,
        _enforce_surface_controls_or_fail,
    )

    state = _apply_surface_generation_controls(
        None,
        SimpleNamespace(layers=_layers()),
        {"clean_user_surface_contract": True},
    )

    assert state["surface_alpha_requested"] == 0.0
    assert state["surface_alpha_applied"] == 0.0
    assert state["apply_errors"] == []
    _enforce_surface_controls_or_fail(
        {"clean_user_surface_contract": True}, state
    )


def test_positive_surface_control_requires_real_steering_engine():
    from core.brain.llm.mlx_worker import _apply_surface_generation_controls

    state = _apply_surface_generation_controls(
        None,
        SimpleNamespace(layers=_layers()),
        {
            "clean_user_surface_contract": True,
            "clean_user_surface_steering_alpha": 0.2,
        },
    )

    assert state["surface_alpha_requested"] == 0.2
    assert "surface_alpha_applied" not in state
    assert state["apply_errors"] == ["steering_unavailable"]

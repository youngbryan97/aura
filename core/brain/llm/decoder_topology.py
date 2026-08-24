"""What a decoder is made of, read from the model rather than assumed.

Every compute estimate, budget, and layer-window decision in the latent cortex
was written when a decoder meant one thing: N identical blocks, each with
Q/K/V/O attention over a growing K/V cache and a SwiGLU feed-forward. Qwen3.5
hybrids break three of those assumptions at once and none of them announce it.

* The wrapper moved. ``Qwen3_5ForConditionalGeneration`` exposes
  ``language_model``, not ``model``, so the usual ``model.model.args`` walk
  finds nothing and the caller reports the model as having no profile at all.
* ``hidden_size == num_attention_heads * head_dim`` is false. 24 heads of 256
  is 6144 against a 5120 residual stream, which is legal and normal, and an
  invariant asserting otherwise refuses the checkpoint outright.
* Three layers in four are not attention layers. A ``GatedDeltaNet`` block
  holds a fixed-size recurrent state, has different projections, and never
  populates a K/V cache. Counting it as an attention layer overstates both the
  FLOPs and the cache footprint, and quietly answers "yes" to mechanisms that
  need a K/V cache to exist.

So this module reports the layer inventory as a fact about the loaded object,
and gives the mechanisms that genuinely require full-layer attention a place to
refuse honestly rather than a number that happens to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

DECODER_TOPOLOGY_SCHEMA: Final = "aura.decoder_topology.v1"

FULL_ATTENTION: Final = "full_attention"
LINEAR_ATTENTION: Final = "linear_attention"

#: Attribute names a wrapper may use for the causal language model inside it.
#: ``language_model`` is the multimodal convention; ``model`` is the plain one.
_LANGUAGE_MODEL_ATTRS: Final = ("language_model", "model")


class DecoderTopologyError(ValueError):
    """The model does not describe a decoder this contract can read."""


@dataclass(frozen=True, slots=True)
class LayerTopology:
    """One layer, and whether it keeps a K/V cache.

    ``holds_kv`` is the load-bearing field. A linear-attention layer carries a
    recurrent state of fixed size instead, so anything sizing a cache, rewinding
    one, or persisting slot keys into one has to know the difference.
    """

    index: int
    kind: str
    holds_kv: bool


@dataclass(frozen=True, slots=True)
class DecoderTopology:
    """The layer inventory of one loaded decoder."""

    model_type: str
    num_hidden_layers: int
    hidden_size: int
    layers: tuple[LayerTopology, ...]

    @property
    def is_hybrid(self) -> bool:
        return any(not layer.holds_kv for layer in self.layers)

    @property
    def attention_layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if layer.holds_kv)

    @property
    def linear_layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if not layer.holds_kv)

    @property
    def kv_layer_count(self) -> int:
        return len(self.attention_layer_indices)

    def kind_at(self, index: int) -> str:
        for layer in self.layers:
            if layer.index == index:
                return layer.kind
        raise DecoderTopologyError(f"layer {index} is outside this decoder")

    def require_full_layer_kv(self, mechanism: str) -> None:
        """Refuse a mechanism that needs every layer to keep a K/V cache.

        Persisting thought-slot keys so every generated token attends to them
        *at every layer* is such a mechanism. On a hybrid decoder it holds at
        the attention layers and is simply not expressible at the others, and
        the honest answer is to say so rather than to run it on a quarter of
        the stack and report the old experiment.
        """
        if not self.is_hybrid:
            return
        raise DecoderTopologyError(
            f"{mechanism} requires full-layer K/V attention; this decoder keeps "
            f"a cache at {self.kv_layer_count} of {self.num_hidden_layers} "
            f"layers ({len(self.linear_layer_indices)} carry a recurrent state)"
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": DECODER_TOPOLOGY_SCHEMA,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "is_hybrid": self.is_hybrid,
            "attention_layer_indices": list(self.attention_layer_indices),
            "linear_attention_layer_indices": list(self.linear_layer_indices),
        }


def resolve_language_model(model: Any) -> Any:
    """The causal language model inside whatever wrapper was handed over.

    A multimodal wrapper holds it at ``language_model``; a plain one is already
    it. Walking to the object that owns ``args`` and ``layers`` is what makes
    the rest of this module independent of how the checkpoint was packaged.
    """
    seen: list[int] = []
    current = model
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.append(id(current))
        args = getattr(current, "args", None)
        if args is not None and getattr(args, "hidden_size", None) is not None:
            return current
        for name in _LANGUAGE_MODEL_ATTRS:
            candidate = getattr(current, name, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    raise DecoderTopologyError("model does not expose a decoder compute profile")


def decoder_layers(model: Any) -> list[Any]:
    """The transformer blocks, wherever the wrapper keeps them."""
    language = resolve_language_model(model)
    for holder in (language, getattr(language, "model", None)):
        layers = getattr(holder, "layers", None)
        if layers:
            return list(layers)
    raise DecoderTopologyError("model does not expose decoder layers")


def _linear_signal(layer: Any) -> bool | None:
    """The model's own answer, or None when this layer does not give one.

    ``is_linear`` is set per layer at construction on hybrid models. A block
    holding ``linear_attn`` says the same thing structurally, and one holding
    ``self_attn`` says the opposite. Anything else is silent, and silence is
    not evidence of either.
    """
    is_linear = getattr(layer, "is_linear", None)
    if isinstance(is_linear, bool):
        return is_linear
    if getattr(layer, "linear_attn", None) is not None:
        return True
    if getattr(layer, "self_attn", None) is not None:
        return False
    return None


def _layer_kv_flags(layers: list[Any]) -> list[bool]:
    """Which layers hold a K/V cache, using the model's signal where it has one.

    A decoder where no layer describes itself is read as dense, which is the
    reading every checkpoint had before hybrids existed and the one a bare test
    double intends. A decoder where *any* layer says it is linear is
    self-describing, and the silent layers in it are attention layers -- the
    hybrid convention, where only the linear blocks carry the marker.
    """
    signals = [_linear_signal(layer) for layer in layers]
    if not any(signal is True for signal in signals):
        return [True] * len(layers)
    return [signal is not True for signal in signals]


def topology_from_model(model: Any) -> DecoderTopology:
    language = resolve_language_model(model)
    args = getattr(language, "args", None)
    layers = decoder_layers(model)
    flags = _layer_kv_flags(layers)
    inventory = tuple(
        LayerTopology(
            index=index,
            kind=FULL_ATTENTION if holds_kv else LINEAR_ATTENTION,
            holds_kv=holds_kv,
        )
        for index, holds_kv in enumerate(flags)
    )
    if not any(entry.holds_kv for entry in inventory):
        raise DecoderTopologyError("decoder exposes no attention layers")
    return DecoderTopology(
        model_type=str(getattr(args, "model_type", type(model).__name__)),
        num_hidden_layers=len(inventory),
        hidden_size=int(getattr(args, "hidden_size", 0)),
        layers=inventory,
    )


def topology_from_config(config: Any) -> DecoderTopology:
    """The same inventory, from a config, for decisions made before a load."""
    if not isinstance(config, dict):
        raise DecoderTopologyError("config must be a mapping")
    text = config.get("text_config")
    source = text if isinstance(text, dict) else config
    layers = source.get("num_hidden_layers")
    if not isinstance(layers, int) or layers < 1:
        raise DecoderTopologyError("config declares no positive num_hidden_layers")

    declared = source.get("layer_types")
    if isinstance(declared, list) and len(declared) == layers:
        kinds = [
            FULL_ATTENTION if str(kind) == FULL_ATTENTION else LINEAR_ATTENTION
            for kind in declared
        ]
    else:
        interval = source.get("full_attention_interval")
        if isinstance(interval, int) and interval > 0:
            kinds = [
                FULL_ATTENTION if (index + 1) % interval == 0 else LINEAR_ATTENTION
                for index in range(layers)
            ]
        else:
            kinds = [FULL_ATTENTION] * layers

    return DecoderTopology(
        model_type=str(config.get("model_type") or source.get("model_type") or ""),
        num_hidden_layers=layers,
        hidden_size=int(source.get("hidden_size") or 0),
        layers=tuple(
            LayerTopology(index=index, kind=kind, holds_kv=kind == FULL_ATTENTION)
            for index, kind in enumerate(kinds)
        ),
    )


def topology_disagreements(
    from_model: DecoderTopology, from_config: DecoderTopology
) -> list[str]:
    """Where the loaded object and its config describe different decoders.

    Checked because they can disagree: a config edited after a fuse, or a
    loader that drops or reorders blocks, produces a model whose receipts cite
    geometry it does not have. Reported field by field so the mismatch names
    itself instead of arriving as a digest difference.
    """
    problems: list[str] = []
    if from_model.num_hidden_layers != from_config.num_hidden_layers:
        problems.append(
            "num_hidden_layers: model "
            f"{from_model.num_hidden_layers} vs config "
            f"{from_config.num_hidden_layers}"
        )
    if (
        from_model.hidden_size
        and from_config.hidden_size
        and from_model.hidden_size != from_config.hidden_size
    ):
        problems.append(
            f"hidden_size: model {from_model.hidden_size} vs config "
            f"{from_config.hidden_size}"
        )
    if from_model.attention_layer_indices != from_config.attention_layer_indices:
        problems.append(
            "attention layer placement differs between the model and its config"
        )
    return problems

"""Load a Prism Hadamard-folded 2-bit pack onto a stock MLX Qwen3.5 model.

Ternary Bonsai 2 27B is Qwen3.5 27B at two bits, group 128, in a 8.6GB
MLX pack. Two bits alone destroys a model of this size: the damage is
concentrated in a few activation channels whose magnitude dwarfs the
rest, and a group scale sized for the outlier quantises everything else
to nothing. A Hadamard rotation spreads each channel's magnitude across
its whole block before quantising, so no group is sized by one outlier,
and the inverse rotation is folded into the weights on the other side.
The arithmetic is exact and MLX has both primitives already —
``mx.hadamard_transform`` and a 2-bit ``quantized_matmul``.

The published pack is schema 2 and the loader published beside it reads
schema 1: it refuses the config it ships with, and it looks for the sign
vectors among the tensors, where schema 2 no longer keeps them. Schema 2
puts the whole transform manifest in ``hadamard.json`` — block size, the
401 folded tensors, the one inverse-folded embedding, and the sign
vectors by width. This reads that.

The two modules below are the mechanism, kept here rather than imported
from the model directory: a checkpoint is data, and a checkpoint that
ships the code to read itself is a checkpoint that runs. Derived from
the runtime published with the pack (Apache 2.0, Prism ML).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

logger = logging.getLogger("Aura.PrismHadamard")

__all__ = [
    "PACK_MODEL_TYPE",
    "Packed",
    "is_prism_hadamard_pack",
    "load_prism_hadamard_pack",
    "load_prism_hadamard_text_model",
]

PACK_MODEL_TYPE = "prism_hadamard_qwen35"

#: Block sizes the transform has been validated at, from the published
#: runtime. A block the pack was not folded at reconstructs noise.
_VALIDATED_BLOCKS = (512, 1024, 2048, 4096)

#: The pack stores two-bit groups of 128. Both are contracts with the
#: stored tensors, not settings.
_BITS = 2
_GROUP = 128


def fwht(x: mx.array, block: int, signs: mx.array, *, inverse: bool = False) -> mx.array:
    """Signed fast Walsh-Hadamard transform over the last dimension."""

    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(
        x.reshape(-1, block), scale=1 / math.sqrt(block)
    ).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A two-bit linear or embedding whose activations are rotated first."""

    def __init__(
        self,
        arrays: tuple[mx.array, mx.array, mx.array],
        block: int = 0,
        signs: mx.array | None = None,
        embedding: bool = False,
        dtype: Any = mx.float16,
    ) -> None:
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        self.block = block
        self.signs = signs
        self.embedding = embedding
        self.dtype = dtype

    def __call__(self, x: mx.array) -> mx.array:
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=_GROUP,
                    bits=_BITS,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return fwht(out, self.block, self.signs, inverse=True) if self.block else out
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=_GROUP,
            bits=_BITS,
        )


def is_prism_hadamard_pack(directory: str | Path) -> bool:
    """True when this directory holds a pack this loader owns."""

    config = Path(directory) / "config.json"
    if not config.is_file():
        return False
    try:
        return (
            json.loads(config.read_text(encoding="utf-8")).get("model_type")
            == PACK_MODEL_TYPE
        )
    except (OSError, ValueError):
        return False


def _signs_by_width(manifest: dict[str, Any]) -> dict[int, mx.array]:
    if manifest.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError("Only explicit sign vectors are supported")
    widths = list(manifest["prism.hadamard.sign_widths"])
    values = list(manifest["prism.hadamard.sign_values"])
    signs: dict[int, mx.array] = {}
    offset = 0
    for width in widths:
        chunk = values[offset : offset + int(width)]
        if len(chunk) != int(width) or any(value not in (-1, 1) for value in chunk):
            raise ValueError(f"Invalid sign vector for width {width}")
        signs[int(width)] = mx.array(chunk, dtype=mx.float32)
        offset += int(width)
    if offset != len(values):
        raise ValueError("Trailing sign values in the transform manifest")
    return signs


def _resolve(model: Any, path: str) -> tuple[Any, str]:
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _validate(original: Any, arrays: list[mx.array], signs: mx.array | None) -> None:
    if not isinstance(original, (nn.Linear, nn.Embedding)):
        raise ValueError("Unsupported packed module target")
    rows, width = original.weight.shape
    if width % _GROUP:
        raise ValueError("Invalid packed width")
    expected = [(rows, width // 16), (rows, width // _GROUP), (rows, width // _GROUP)]
    if [tuple(a.shape) for a in arrays] != expected or arrays[0].dtype != mx.uint32:
        raise ValueError("Invalid packed tensor shapes or storage dtype")
    for array in arrays[1:]:
        if array.dtype not in (mx.float16, mx.float32, mx.bfloat16):
            raise ValueError("Invalid affine dtype")
        if not mx.all(mx.isfinite(array)).item():
            raise ValueError("Non-finite affine parameters")
    if signs is not None and tuple(signs.shape) != (width,):
        raise ValueError("Sign vector does not match the packed width")


def load_prism_hadamard_text_model(directory: str | Path) -> tuple[Any, dict[str, Any]]:
    """Return the text model and its config, or raise saying what was wrong."""

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != PACK_MODEL_TYPE:
        raise ValueError(f"Not a {PACK_MODEL_TYPE} pack: {config.get('model_type')!r}")
    quantization = config.get("quantization") or {}
    if (quantization.get("bits"), quantization.get("group_size")) != (_BITS, _GROUP):
        raise ValueError(f"Unsupported quantization {quantization!r}")

    manifest = json.loads(
        (directory / str(config.get("hadamard_config") or "hadamard.json")).read_text(
            encoding="utf-8"
        )
    )
    block = int(manifest["prism.hadamard.block_size"])
    if block not in _VALIDATED_BLOCKS:
        raise ValueError(f"Unvalidated Hadamard block size {block}")
    signs = _signs_by_width(manifest)

    model = TextModel(TextModelArgs.from_dict(config["text_config"]))
    weights = mx.load(str(directory / "model.safetensors"))
    # The pack is written in the vision-language namespace, where the text
    # tower hangs off `language_model` and the vision tower sits beside it.
    # The pack ships both — its README says vision is not included, which
    # is true of the runtime published with it and not of the weights — so
    # everything outside the text tower is left on disk rather than loaded
    # into a text model that has nowhere to put it.
    prefix = "language_model."
    if any(name.startswith(prefix) for name in weights):
        weights = {
            name[len(prefix) :]: value
            for name, value in weights.items()
            if name.startswith(prefix)
        }

    seen: set[str] = set()
    for record in config["modules"]:
        path = str(record["path"])
        if path in seen:
            raise ValueError(f"Duplicate packed module {path!r}")
        seen.add(path)
        if str(record.get("dtype")) != "float16":
            raise ValueError(f"Unsupported activation dtype for {path!r}")
        arrays = [weights[f"{path}.{suffix}"] for suffix in ("weight", "scales", "biases")]
        parent, leaf = _resolve(model, path)
        original = getattr(parent, leaf)
        rows, width = original.weight.shape
        folded = int(record.get("block") or 0)
        if folded and folded != block:
            raise ValueError(f"{path!r} is folded at {folded}, the pack at {block}")
        vector = signs.get(width) if folded else None
        if folded and vector is None:
            raise ValueError(f"No sign vector of width {width} for {path!r}")
        _validate(original, arrays, vector)
        setattr(
            parent,
            leaf,
            Packed(arrays, folded, vector, bool(record.get("embedding")), mx.float16),
        )

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    logger.info(
        "🌳 Prism Hadamard pack loaded: %d packed modules, block %d, %d sign width(s)",
        len(seen),
        block,
        len(signs),
    )
    return model, config


def load_prism_hadamard_pack(directory: str | Path) -> tuple[Any, Any]:
    """Return ``(model, tokenizer)``, the shape ``mlx_lm.load`` returns.

    ``TextModel`` already answers everything mlx_lm's generation asks of a
    model — ``__call__`` to logits, ``make_cache``, ``layers`` — so the
    pack drops into the ordinary decode path once it is loaded. Only the
    loading is different, and only because the published loader reads a
    schema the published weights are not written in.
    """

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    directory = Path(directory)
    model, _config = load_prism_hadamard_text_model(directory)
    tokenizer = load_tokenizer(directory, eos_token_ids=_eos_ids(directory))
    return model, tokenizer


def _eos_ids(directory: Path) -> list[int] | None:
    """Stop ids from the pack's generation config, when it names any."""

    path = directory / "generation_config.json"
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    eos = config.get("eos_token_id")
    if isinstance(eos, int):
        return [eos]
    if isinstance(eos, list):
        return [int(value) for value in eos if isinstance(value, int)]
    return None

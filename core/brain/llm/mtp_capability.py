"""Whether this checkpoint's multi-token-prediction head can actually be used.

Qwen3.5 configs declare ``mtp_num_hidden_layers``, which reads like a decode
accelerator waiting to be switched on. For the checkpoints installed here it is
a declaration with nothing behind it, and the detection below says so in three
independent ways rather than one, because any single check could be repaired
without the capability becoming real:

1. The weight index carries no ``mtp.*`` tensor. Neither the mlx-community base
   nor the persona fuse has one.
2. ``mlx_lm.models.qwen3_5.sanitize`` drops every ``mtp.*`` key on load, by
   design. Even a checkpoint that shipped them would not keep them.
3. The supported speculative path in ``mlx_lm`` takes a separate draft model,
   not an internal head, so there is no API through which an MTP head would be
   reached.

Reaching around any of those means patching a third-party module, which is not
a capability -- it is an unreviewed fork of somebody else's decode loop running
in the serving path.

What *is* supported is ordinary draft-model speculation, and that has a real
prerequisite this module reports: the draft must share the target's vocabulary
exactly. The 1.5B and 7B rigs do not (152k against 248k); ``Qwen3.5-9B-4bit``
does. Whether a 9B draft in front of a 27B target is worth 5.5 GB and its
acceptance rate is a measurement, not a property, and this module never guesses
it.

Detection is static: it reads configs and weight indices, and loads nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

MTP_CAPABILITY_SCHEMA: Final = "aura.mtp_capability.v1"

#: Every reason a native head cannot be used. Reported together, because a
#: reader who fixes one and expects the capability to appear is owed the rest.
NATIVE_BLOCKERS: Final = (
    "checkpoint_carries_no_mtp_tensors",
    "loader_discards_mtp_tensors",
    "no_supported_api_reaches_an_internal_head",
)


@dataclass(frozen=True)
class MTPCapability:
    """A typed verdict. ``supported`` false is a result, not an error."""

    schema: str
    checkpoint: str
    declares_mtp_layers: int
    mtp_tensor_count: int
    loader_discards_mtp: bool
    native_supported: bool
    native_blockers: tuple[str, ...]
    draft_speculation_supported: bool
    compatible_draft_models: tuple[dict[str, Any], ...] = ()
    unmeasured: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MTPAcceptanceTelemetry:
    """Acceptance accounting for a draft-model run, if one is ever admitted.

    Speculative decoding is only a win when drafts are accepted often enough to
    pay for the draft forward passes. A lane that reports latency without
    acceptance cannot tell a speedup from a slowdown, so both are recorded and
    the rate is derived rather than supplied.
    """

    drafted_tokens: int = 0
    accepted_tokens: int = 0
    draft_forward_passes: int = 0
    target_forward_passes: int = 0
    rejections: int = 0
    samples: list[float] = field(default_factory=list)

    def record(self, *, drafted: int, accepted: int) -> None:
        if drafted < 0 or accepted < 0 or accepted > drafted:
            raise ValueError("accepted tokens must lie inside the drafted count")
        self.drafted_tokens += drafted
        self.accepted_tokens += accepted
        self.rejections += drafted - accepted
        self.draft_forward_passes += 1
        self.target_forward_passes += 1
        if drafted:
            self.samples.append(accepted / drafted)

    @property
    def acceptance_rate(self) -> float | None:
        """None until something was drafted. Never a default of zero.

        An unmeasured rate reported as 0.0 is indistinguishable from a draft
        model that never gets anything right, and the two call for opposite
        actions.
        """
        if not self.drafted_tokens:
            return None
        return self.accepted_tokens / self.drafted_tokens

    def receipt(self) -> dict[str, Any]:
        return {
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejections": self.rejections,
            "draft_forward_passes": self.draft_forward_passes,
            "target_forward_passes": self.target_forward_passes,
            "acceptance_rate": self.acceptance_rate,
            "measured": bool(self.drafted_tokens),
        }


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    inner = config.get("text_config")
    return inner if isinstance(inner, dict) else config


def _mtp_tensor_count(model_dir: Path) -> int:
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        return 0
    try:
        weights = json.loads(index.read_text()).get("weight_map") or {}
    except (OSError, ValueError):
        return 0
    return sum(1 for key in weights if "mtp." in key)


def loader_discards_mtp(model_type: str) -> bool:
    """Does the installed mlx_lm drop ``mtp.*`` for this architecture?

    Read from the loader rather than assumed, so a future mlx_lm that keeps the
    weights flips this without anybody editing a constant here.
    """
    try:
        import importlib
        import inspect

        module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except (ImportError, ValueError):
        return False
    # Model.sanitize delegates to the inner text model's, so inspecting only
    # the outer class reports False while the keys are still being dropped one
    # frame down. Every class in the module is asked instead.
    for attribute in vars(module).values():
        sanitize = getattr(attribute, "sanitize", None)
        if sanitize is None:
            continue
        try:
            source = inspect.getsource(sanitize)
        except (OSError, TypeError):
            continue
        if '"mtp." not in k' in source or "'mtp.' not in k" in source:
            return True
    return False


def compatible_draft_models(
    target_vocab: int, target_type: str, candidates: dict[str, Path]
) -> tuple[dict[str, Any], ...]:
    """Installed models a draft lane could legally use.

    The vocabulary has to match exactly. A draft that tokenizes differently
    proposes ids the target reads as other words, and the rejection sampler
    cannot repair that -- it would silently degrade to a slow, wrong decode.
    """
    found: list[dict[str, Any]] = []
    for name, directory in sorted(candidates.items()):
        config_path = Path(directory) / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError):
            continue
        text = _text_config(config)
        vocab = text.get("vocab_size")
        family = config.get("model_type") or text.get("model_type")
        if vocab != target_vocab:
            continue
        found.append(
            {
                "name": name,
                "path": str(directory),
                "vocab_size": vocab,
                "model_type": family,
                "num_hidden_layers": text.get("num_hidden_layers"),
                "same_family": family == target_type,
            }
        )
    return tuple(found)


def detect(
    model_dir: Path | str, draft_candidates: dict[str, Path] | None = None
) -> MTPCapability:
    """Static capability detection. Loads no weights."""
    directory = Path(model_dir)
    config = json.loads((directory / "config.json").read_text())
    text = _text_config(config)
    model_type = str(config.get("model_type") or text.get("model_type") or "")
    # qwen3_5_text names the inner stack; the loader module is qwen3_5.
    loader_type = model_type.removesuffix("_text")

    declared = text.get("mtp_num_hidden_layers")
    declared = int(declared) if isinstance(declared, int) else 0
    tensors = _mtp_tensor_count(directory)
    discards = loader_discards_mtp(loader_type)

    blockers: list[str] = []
    if tensors == 0:
        blockers.append("checkpoint_carries_no_mtp_tensors")
    if discards:
        blockers.append("loader_discards_mtp_tensors")
    blockers.append("no_supported_api_reaches_an_internal_head")

    drafts = compatible_draft_models(
        int(text.get("vocab_size") or 0), loader_type, draft_candidates or {}
    )
    return MTPCapability(
        schema=MTP_CAPABILITY_SCHEMA,
        checkpoint=str(directory),
        declares_mtp_layers=declared,
        mtp_tensor_count=tensors,
        loader_discards_mtp=discards,
        native_supported=False,
        native_blockers=tuple(blockers),
        draft_speculation_supported=bool(drafts),
        compatible_draft_models=drafts,
        unmeasured=(
            "draft_acceptance_rate",
            "end_to_end_speedup",
            "output_equivalence_under_speculation",
        ),
        notes=(
            "The declared MTP head has no weights in this checkpoint and the "
            "loader discards the keys regardless. Draft-model speculation is "
            "the only supported route and its benefit is unmeasured."
        ),
    )

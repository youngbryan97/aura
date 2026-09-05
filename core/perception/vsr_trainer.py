"""core/perception/vsr_trainer.py
──────────────────────────────
A trainable open-vocabulary VSR model — the free, license-clean path.

The auto_avsr frontier weights are gated by the BBC LRS3 data license
(someone else's data term, not ours to waive). This module is the
alternative that is entirely yours: a from-scratch CTC lip-reading
network you train on your OWN footage (your camera, your license) or
any corpus you have the rights to. Nothing here downloads or depends on
restricted weights.

Pipeline:
- ``VSRNet``: 3D-conv spatiotemporal frontend → bidirectional GRU →
  per-frame character logits. Consumes the SAME (1,1,T,88,88) tensor the
  ONNX backend's ``preprocess_mouth_crops`` produces, and emits the SAME
  (T, vocab) logits the CTC decoders consume — so a model trained here
  exports straight into the existing inference path.
- ``train_vsr``: a real CTC-loss training loop (torch.nn.CTCLoss).
- ``export_onnx``: writes the ONNX the existing OnnxVSRBackend loads,
  with an honest self-owned provenance sidecar.

Verified by overfitting a synthetic labeled set to near-zero CTC loss
with exact greedy decode — proof the whole learnable pipeline closes,
using data generated locally with no license attached.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.perception.vsr_ctc import BLANK, Vocabulary, greedy_decode
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.VSRTrainer")

MOUTH_SIZE = 88
_TRAINER_ERRORS = (ImportError, RuntimeError, TypeError, ValueError)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class LabeledClip:
    """One training example: a mouth-crop clip and its transcript."""

    mouth_crops: np.ndarray  # (T, H, W) or (T, H, W, 1|3) uint8
    transcript: str


def _build_net(vocab_size: int):
    import torch
    from torch import nn

    class VSRNet(nn.Module):
        """3D-conv frontend + BiGRU + CTC head. Small by design: enough
        capacity to learn real lip dynamics, light enough to train on CPU
        for bootstrapping and on a self-recorded corpus."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            # GroupNorm, not BatchNorm: clips train one at a time (batch=1),
            # and GroupNorm's statistics are per-sample, so train and eval
            # behave identically — no running-stat drift that would make a
            # net that fits in training mispredict in eval.
            self.frontend = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
                nn.GroupNorm(4, 16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
            )
            self.rnn = nn.GRU(
                input_size=32, hidden_size=64, num_layers=2,
                bidirectional=True, batch_first=True,
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, video: torch.Tensor) -> torch.Tensor:
            # video: (B, 1, T, H, W) → conv (B, 32, T, H', W')
            features = self.frontend(video)
            # Global spatial average (over H', W'), keeping the dynamic
            # time axis — an explicit mean exports to ONNX where adaptive
            # pooling with a dynamic output size does not.
            features = features.mean(dim=(3, 4)).transpose(1, 2)  # (B, T, 32)
            sequence, _ = self.rnn(features)
            logits = self.head(sequence)  # (B, T, num_classes)
            return logits.log_softmax(dim=-1)

    return VSRNet(vocab_size)


def _clip_to_tensor(clip_frames: np.ndarray):
    """(T,H,W[,C]) uint8 → (1, T, 88, 88) float32 standardized — matches
    the ONNX front end so a trained net and the runtime agree on inputs."""
    from core.perception.vsr_onnx_backend import preprocess_mouth_crops

    # preprocess returns (1, 1, T, 88, 88); drop the outer batch axis.
    return preprocess_mouth_crops(clip_frames)[0]


@dataclass
class TrainResult:
    final_loss: float
    initial_loss: float
    epochs: int
    train_accuracy: float
    model: object
    vocab: Vocabulary


def train_vsr(
    clips: Sequence[LabeledClip],
    vocab: Vocabulary,
    *,
    epochs: int = 200,
    learning_rate: float = 3e-3,
    seed: int = 0,
) -> TrainResult:
    """Train a VSRNet with CTC loss on labeled clips. Pure-local; the
    only data is what the caller supplies."""
    if not torch_available():
        raise RuntimeError("torch is required to train the VSR model")
    if not clips:
        raise ValueError("need at least one labeled clip")
    import torch

    torch.manual_seed(seed)
    net = _build_net(vocab.size)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)

    tensors = [torch.from_numpy(_clip_to_tensor(clip.mouth_crops)) for clip in clips]
    targets = [
        torch.tensor(
            [vocab.alphabet.index(ch) + 1 for ch in clip.transcript],
            dtype=torch.long,
        )
        for clip in clips
    ]

    initial_loss = float("nan")
    final_loss = float("nan")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for tensor, target in zip(tensors, targets):
            optimizer.zero_grad()
            logits = net(tensor.unsqueeze(0))  # (1, T, V)
            log_probs = logits.transpose(0, 1)  # (T, 1, V) for CTCLoss
            input_len = torch.tensor([log_probs.shape[0]], dtype=torch.long)
            target_len = torch.tensor([target.numel()], dtype=torch.long)
            loss = ctc(log_probs, target.unsqueeze(0), input_len, target_len)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        epoch_loss /= len(clips)
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    accuracy = _train_accuracy(net, tensors, clips, vocab)
    return TrainResult(
        final_loss=final_loss,
        initial_loss=initial_loss,
        epochs=epochs,
        train_accuracy=accuracy,
        model=net,
        vocab=vocab,
    )


def _train_accuracy(net, tensors, clips, vocab: Vocabulary) -> float:
    import torch

    net.eval()
    correct = 0
    with torch.no_grad():
        for tensor, clip in zip(tensors, clips):
            logits = net(tensor.unsqueeze(0))[0].cpu().numpy()
            if greedy_decode(logits, vocab) == clip.transcript:
                correct += 1
    net.train()
    return correct / len(clips)


def predict_transcript(model, mouth_crops: np.ndarray, vocab: Vocabulary) -> str:
    import torch

    model.eval()
    tensor = torch.from_numpy(_clip_to_tensor(mouth_crops)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0].cpu().numpy()
    return greedy_decode(logits, vocab)


def export_onnx(model, path: str | Path, *, num_frames: int = 32) -> dict:
    """Export a trained VSRNet to the ONNX contract the runtime backend
    loads, with a self-owned provenance sidecar (no external license)."""
    if not torch_available():
        raise RuntimeError("torch is required to export ONNX")
    import torch

    from core.perception.vsr_onnx_backend import ModelProvenance, _record_provenance

    path = Path(path)
    model.eval()
    sample_input = torch.zeros(
        (1, 1, num_frames, MOUTH_SIZE, MOUTH_SIZE),
        dtype=torch.float32,
    )
    try:
        torch.onnx.export(
            model, sample_input, str(path),
            input_names=["mouth_crops"], output_names=["logits"],
            dynamic_axes={"mouth_crops": {2: "frames"}, "logits": {1: "frames"}},
            opset_version=17,
            dynamo=False,  # legacy tracer handles GRU + dynamic axes reliably
        )
    except _TRAINER_ERRORS as exc:
        record_degradation("vsr_trainer.export", exc)
        raise
    provenance = ModelProvenance(
        model_id=f"aura-self-trained-vsr-{path.stem}",
        license="owner-trained (no external data license)",
        training_data="operator-supplied (self-recorded / owned corpus)",
        acknowledged=True,  # self-owned data needs no external acknowledgement
    )
    _record_provenance(path, provenance)
    return provenance.to_dict()


# ── persistent labeled corpus (your data, governed writes) ───────

class VSRCorpus:
    """A persisted set of labeled clips under a directory: one .npz per
    clip plus a JSONL manifest. Fed by any source you have rights to —
    your webcam, screen captures, or the synthetic generator — and
    consumed directly by ``train_vsr``. All writes go through the file
    write gateway under a governed scope."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def add(self, clip: LabeledClip, *, clip_id: str) -> None:
        import io

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        cleaned = "".join(ch for ch in clip_id if ch.isalnum() or ch in "_-")
        if not cleaned:
            raise ValueError("clip_id must be alphanumeric")
        gateway = get_file_write_gateway()
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, mouth_crops=clip.mouth_crops,
            transcript=np.array(clip.transcript))
        manifest_line = _json_line(
            {"clip_id": cleaned, "transcript": clip.transcript,
             "frames": int(clip.mouth_crops.shape[0])})
        with local_internal_governed_scope(
            "perception.vsr_corpus", domain="file_write",
            receipt_prefix="vsr-corpus"):
            gateway.ensure_directory(self.root, source="perception.vsr_corpus")
            gateway.write_bytes(
                self.root / f"{cleaned}.npz", buffer.getvalue(),
                source="perception.vsr_corpus")
            gateway.append_text(
                self.root / "manifest.jsonl", manifest_line,
                source="perception.vsr_corpus")

    def load(self) -> list[LabeledClip]:
        clips: list[LabeledClip] = []
        if not self.root.exists():
            return clips
        for path in sorted(self.root.glob("*.npz")):
            try:
                data = np.load(path, allow_pickle=False)
                clips.append(LabeledClip(
                    mouth_crops=data["mouth_crops"],
                    transcript=str(data["transcript"])))
            except (OSError, ValueError, KeyError) as exc:
                record_degradation("perception.vsr_corpus.load", exc)
        return clips


def _json_line(payload: dict) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


# ── synthetic corpus for the closed-loop training proof ──────────

def synthesize_clip(transcript: str, vocab: Vocabulary, *, frames_per_char: int = 4,
                    seed: int = 0) -> LabeledClip:
    """Render a transcript into a deterministic mouth-crop clip via a
    per-character visual code — a legible signal a conv net can learn,
    standing in for real lip shapes so the training loop is provable
    without any licensed corpus."""
    rng = np.random.default_rng(seed)
    size = MOUTH_SIZE
    frames = []
    for char in transcript:
        index = vocab.alphabet.index(char) + 1
        base = np.zeros((size, size), dtype=np.float64)
        # Distinct spatial pattern per character index: a bright block
        # whose position and size are a function of the index.
        row = (index * 7) % (size - 20)
        col = (index * 11) % (size - 20)
        base[row:row + 18, col:col + 18] = 220.0
        base[size // 2 - index % 8: size // 2 + index % 8, :] += 60.0
        for _ in range(frames_per_char):
            noisy = np.clip(base + rng.normal(0, 4, base.shape), 0, 255)
            frames.append(noisy.astype(np.uint8))
    return LabeledClip(mouth_crops=np.stack(frames), transcript=transcript)

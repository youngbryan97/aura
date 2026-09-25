"""Source-fold direct-bank fitting keeps generated rivals label-blind."""

from types import SimpleNamespace

import pytest
import torch

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_decoder import ProgramDecoderConfig, SemanticProgramDecoder
from core.learning.semantic_program_ir import TokenSpan
from tools.refit_semantic_direct_bank import _digest, _generated_contrast, _sha, load_refit


def test_generated_contrast_labels_only_after_target_blind_proposal():
    target = Program(2, (Instruction("sub", (0, 1)),))
    wrong = Program(2, (Instruction("mul", (0, 1)),))
    equivalent = Program(2, (Instruction("sub", (0, 1)),))
    features = torch.ones(3, 4)
    spans = (TokenSpan(0, 1), TokenSpan(1, 2))
    kinds = ("integer", "integer")
    item = SimpleNamespace(public_inputs=(5, 2), ir=SimpleNamespace(
        source_text_sha256="a" * 64))

    class Proposer:
        def propose_beam(self, received_features, received_spans, received_kinds,
                         *, beam_width, max_programs):
            assert received_features is features
            assert received_spans is spans
            assert received_kinds is kinds
            assert beam_width == max_programs == 3
            return ((wrong, {}), (equivalent, {}))

    assert _generated_contrast(Proposer(), item, features, spans, kinds, target,
                               width=3) == (target, wrong)


def test_refit_load_refuses_holdout_contamination(tmp_path):
    config = ProgramDecoderConfig(4, width=16, max_steps=2)
    model = SemanticProgramDecoder(config)
    folds = {"receipt_sha256": "fold", "assignments": {"train": 1, "held": 0}}
    plan = {"schema": "aura.semantic_direct_bank_refit.v1", "fold": 0,
            "fold_receipt_sha256": "fold", "parent_direct_checkpoint_sha256": "parent",
            "source_bank_receipt_sha256": "bank", "training_ids": ["train"],
            "heldout_ids": ["held"], "optimizer_split": "train",
            "heldout_controls_training": False}
    checkpoint_path = tmp_path / "fit.pt"
    torch.save({"plan": plan, "model": model.state_dict(), "history": []}, checkpoint_path)
    report = {"plan": plan, "history": [], "checkpoint_sha256": _sha(checkpoint_path),
              "promotion_authorized": False, "validation_used": False, "test_used": False}
    report["receipt_sha256"] = _digest(report)
    restored = load_refit(report, checkpoint_path,
                          direct_checkpoint_sha256="parent", folds=folds,
                          source_bank_receipt_sha256="bank", config=config)
    assert isinstance(restored, SemanticProgramDecoder)
    contaminated = {**report, "plan": {**plan, "training_ids": ["held"]}}
    contaminated["receipt_sha256"] = _digest({
        key: value for key, value in contaminated.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="source-only fold"):
        load_refit(contaminated, checkpoint_path, direct_checkpoint_sha256="parent",
                   folds=folds, source_bank_receipt_sha256="bank", config=config)

"""A head is sized by the logits it is added to, and a head of another width says so when it is skipped."""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "train_endogenous_readout.py"
_spec = importlib.util.spec_from_file_location("train_endogenous_readout", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_the_width_comes_from_the_models_config_even_inside_text_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 248320}}))
    assert tool.logits_width(str(tmp_path)) == 248320
    assert tool.logits_width(str(tmp_path / "absent")) == 0


def test_a_head_of_another_width_is_skipped_and_said_once(caplog):
    from core.brain.llm.endogenous_decode import EndogenousLogitBiasProcessor

    processor = EndogenousLogitBiasProcessor(np.zeros(4, dtype=np.float32))
    try:
        import mlx.core as mx
    except ImportError:
        return
    logits = mx.zeros((1, 6))
    with caplog.at_level(logging.WARNING):
        assert processor(None, logits) is logits
        processor(None, logits)
    said = [r for r in caplog.records if "endogenous head not applied" in r.getMessage()]
    assert len(said) == 1 and "4 wide" in said[0].getMessage() and "are 6" in said[0].getMessage()

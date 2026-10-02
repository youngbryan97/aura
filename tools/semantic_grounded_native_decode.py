"""Attach a verified joint artifact to public, target-blind chart decoding.

The caller owns the exact model and exclusive lane. This loader does not load
backbone weights, acquire a lease or grant serving authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


def selected_chart_receipt(resolutions, outcome):
    if outcome.ir is None:
        return None
    anchors = (*outcome.ir.input_spans, *(instruction.operation_span for instruction in outcome.ir.instructions))
    signature = tuple(sorted((instruction.op, instruction.operation_span.start, instruction.operation_span.end,
        tuple((anchors[register].start, anchors[register].end) for register in instruction.args))
        for instruction in outcome.ir.instructions))
    matches = [row for row in resolutions if row.get("graph_signature") == signature
        and row.get("argument_graph_score") == outcome.pointer_scores["argument_graph_total"]
        and row["status"] == "bound"]
    if not matches:
        raise ValueError("joint decode output has no corresponding measured graph selection")
    return matches[0]


class GroundedNativeChartDecoder:
    def __init__(self, parent, engine, prefix, suffix, plan, verification):
        self.parent, self.engine = parent, engine
        self.prefix, self.suffix = prefix, suffix
        self.plan, self.verification = plan, verification
        self.last_receipt = None

    @classmethod
    def from_fit(cls, model, *, directory, parent_bytes, spec, allow_initial_checkpoint=False):
        from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
        from core.learning.semantic_grounded_binding_engine import GroundedBindingEngine
        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.runtime.file_read_gateway import read_stable_bytes
        from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis
        from tools.semantic_native_adapters import install_native_adapters
        from tools.verify_semantic_grounded_fit import verify

        directory = Path(directory)
        verification = verify(directory)
        report = json.loads(read_stable_bytes(directory / "report.json", max_bytes=64 * 1024 ** 2))
        plan = report.get("native_contract")
        if (type(allow_initial_checkpoint) is not bool
                or not verification["learned_checkpoint_selected"] and not allow_initial_checkpoint
                or not isinstance(plan, dict)
                or plan.get("source_basis", {}).get("parent_sha256") != hashlib.sha256(parent_bytes).hexdigest()
                or plan.get("model_descriptor_sha256") != spec.descriptor_sha256
                or plan.get("pointer_sha256") != spec.pointer_sha256
                or plan.get("model_path") != str(spec.model_path)
                or plan.get("installed_arithmetic") != installed_arithmetic_basis()
                or plan.get("precision") != "native" or plan.get("prefix_strategy") != "full"
                or plan.get("suffix_layer_execution") != "differentiable_native_ops_v1"):
            raise ValueError("joint chart decode changed its source parent, model or arithmetic custody")
        parent = compositional_semantic_program_transducer_from_dict(json.loads(parent_bytes))
        parent = parent.with_global_constraint_arguments().with_joint_operation_argument_scores()
        model.freeze()
        model.eval()
        from core.brain.llm.decoder_topology import decoder_backbone
        split = len(decoder_backbone(model).layers) - plan["suffix_layers"]
        prefix = FrozenDecoderPrefix(model, split_at=split)
        suffix = NativeDecoderSuffix(model, split_at=split)
        install_native_adapters(model, plan)
        for layer in suffix.layers:
            layer.train()
        engine = GroundedBindingEngine.load(directory, native_suffix=suffix, native_contract=plan)
        if engine.operation_field is not None:
            parent = parent.with_conditional_argument_choices()._with_coefficients(operation_length_penalty=0.)
        return cls(parent, engine, prefix, suffix, plan, verification)

    def decode(self, *, source_token_ids, hidden_states, public_inputs,
               source_text_sha256, model_basis_sha256, search_time_limit_s=10.):
        """Only public tokens, frozen parent features and public values enter.

        Native states come from the selected suffix, not teacher operations,
        arguments, construction identities or expected execution answers.
        """
        if self.engine.operation_field is not None:
            from tools.semantic_grounded_batched_chart import BatchedNativeChartDecoder
            decoder = BatchedNativeChartDecoder(self, score_policy="conditional_likelihood")
            outcome = decoder.decode(source_token_ids=source_token_ids, hidden_states=hidden_states,
                public_inputs=public_inputs, source_text_sha256=source_text_sha256,
                model_basis_sha256=model_basis_sha256, search_time_limit_s=search_time_limit_s)
            self.last_receipt = decoder.last_receipt
            return outcome
        import mlx.core as mx

        from core.learning.semantic_grounded_chart_bridge import GroundedBindingChartSolver
        from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis

        self.last_receipt = None
        tokens = tuple(source_token_ids)
        if (not tokens or len(tokens) > self.plan["max_tokens"]
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(search_time_limit_s) not in (int, float)
                or not math.isfinite(search_time_limit_s) or search_time_limit_s <= 0
                or model_basis_sha256 != self.parent.model_basis_sha256
                or self.plan["installed_arithmetic"] != installed_arithmetic_basis()):
            raise ValueError("joint chart decode needs unchanged arithmetic and bounded public source inputs")
        started = time.monotonic()
        hidden = self.prefix.capture(mx.array([tokens], dtype=mx.int32))
        states = mx.stack(self.suffix.layer_states(hidden, tuple(range(self.plan["suffix_layers"]))), axis=2)[0]
        mx.eval(states)
        remaining = search_time_limit_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("joint chart source acquisition exhausted its declared allowance")
        bridge = GroundedBindingChartSolver(self.engine, source_text_sha256, states, max_seconds=remaining)
        proposer = (self.engine.operation_field.proposal(source_text_sha256, states)
                    if self.engine.operation_field is not None else None)
        outcome = self.parent.decode(source_token_ids=tokens, hidden_states=hidden_states,
            public_inputs=public_inputs, source_text_sha256=source_text_sha256,
            model_basis_sha256=model_basis_sha256, search_time_limit_s=remaining,
            binding_chart_solver=bridge, operation_chart_proposer=proposer)
        selected = selected_chart_receipt(bridge.resolutions, outcome)
        self.last_receipt = {"schema": "aura.grounded_native_chart_decode.v1",
            "fit_receipt_sha256": self.verification["fit_receipt_sha256"],
            "weights_sha256": self.verification["weights_sha256"], "source_id": source_text_sha256,
            "source_tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "native_state_shape": list(states.shape), "source_only_native_capture": True,
            "parent_receipt_sha256": self.parent.receipt_sha256, "elapsed_seconds": time.monotonic() - started,
            "selected_chart": selected, "examined_charts": len(bridge.resolutions), "refusal": outcome.refusal,
            "selected_step": self.verification["selected_step"],
            "learned_checkpoint_selected": self.verification["learned_checkpoint_selected"],
            "operation_proposal": proposer.last_receipt if proposer is not None else None,
            "target_available_to_decoder": False, "serving_authority": False}
        return outcome

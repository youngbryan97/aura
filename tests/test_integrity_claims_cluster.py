"""CP126 integrity-claims cluster: proof, not assertion.

Six findings that share one bug class — a claim of integrity that nothing
verifies:

* ``6e1ef7be`` — params_unchanged / fast_weights_erased were independent
  mutable booleans with no relation to any measurement.
* ``16757b09`` — the first-logits digest was described as a universal causal
  audit it cannot be, and carried no algorithm/input binding.
* ``e93ffe9f`` — the service gate accepted those booleans as proof that
  resident weights survived and ephemeral weights were erased.
* ``866530f6`` — worker identity could not distinguish different adapters,
  tokenizers, quantization layouts or dtypes.
* ``79271a67`` — malformed provenance turned identity DEGRADATION reporting
  into an exception.
* ``869a0ce4`` / ``8923b135`` — a "vanilla" control was three fields, and
  32B class rested on one integer.
"""
from __future__ import annotations

import inspect

from core.brain.llm.latent_cortex.types import EpisodeReceipt, WeightIntegrityProof


class TestIntegrityIsProvenNotAsserted:
    def test_a_bare_assertion_is_unproven(self):
        receipt = EpisodeReceipt()
        receipt.params_unchanged = True
        receipt.fast_weights_erased = True
        verdicts = receipt.integrity_verdicts()
        assert verdicts["params_unchanged"]["verdict"] == "unproven"
        assert verdicts["fast_weights_erased"]["verdict"] == "unproven"

    def test_an_unproven_receipt_does_not_satisfy_the_strict_predicate(self):
        receipt = EpisodeReceipt()
        receipt.params_unchanged = True
        receipt.fast_weights_erased = True
        assert receipt.integrity_is_proven() is False

    def test_agreeing_digests_prove_it(self):
        receipt = EpisodeReceipt()
        receipt.weight_integrity = WeightIntegrityProof(
            params_before="a", params_after="a",
            canary_before="c", canary_after="c",
        )
        assert receipt.integrity_is_proven() is True

    def test_disagreeing_digests_refute_it(self):
        receipt = EpisodeReceipt()
        receipt.weight_integrity = WeightIntegrityProof(
            params_before="a", params_after="b",
            canary_before="c", canary_after="c",
        )
        verdicts = receipt.integrity_verdicts()
        assert verdicts["params_unchanged"]["verdict"] == "refuted"
        assert receipt.integrity_is_proven() is False

    def test_a_claim_its_evidence_refutes_is_named(self):
        receipt = EpisodeReceipt()
        receipt.params_unchanged = True
        receipt.weight_integrity = WeightIntegrityProof(
            params_before="a", params_after="b",
        )
        assert "params_unchanged_asserted_but_refuted" in (
            receipt.integrity_verdicts()["contradictions"]
        )

    def test_canary_evidence_is_what_proves_erasure(self):
        # Parameter digests alone are too coarse to prove an erase.
        proof = WeightIntegrityProof(params_before="a", params_after="a")
        assert proof.params_unchanged_proven is True
        assert proof.fast_weights_erased_proven is None

    def test_a_malformed_proof_is_no_proof(self):
        assert WeightIntegrityProof.from_dict("nope").unavailable_reason == (
            "proof_not_a_mapping"
        )
        assert WeightIntegrityProof.from_dict(None).params_unchanged_proven is None

    def test_the_proof_round_trips(self):
        proof = WeightIntegrityProof(
            params_before="a", params_after="a",
            canary_before="c", canary_after="c",
            erased_layer_ids=["layers.0", "layers.1"],
        )
        restored = WeightIntegrityProof.from_dict(proof.to_dict())
        assert restored.erased_layer_ids == ["layers.0", "layers.1"]
        assert restored.params_unchanged_proven is True

    def test_the_receipt_serializes_its_evidence(self):
        payload = EpisodeReceipt().to_dict()
        assert "weight_integrity" in payload
        assert "integrity_verdicts" in payload


class TestFirstLogitsDigestClaimIsNarrowed:
    def test_the_universal_claim_is_gone(self):
        from core.brain.llm.latent_cortex import types

        source = inspect.getsource(types)
        assert "any\nchange to the latent computation shows up here" not in source
        assert "any change to the latent computation shows up here" not in source

    def test_the_digest_carries_a_spec_binding(self):
        receipt = EpisodeReceipt()
        assert hasattr(receipt, "first_logits_digest_spec")
        assert "first_logits_digest_spec" in receipt.to_dict()

    def test_the_honest_direction_is_documented(self):
        from core.brain.llm.latent_cortex import types

        source = inspect.getsource(types)
        assert "NOT proof the latent path matched" in source


class TestServiceGateRequiresEvidence:
    def _verdict(self, receipt, claim="params_unchanged"):
        from core.brain.latent_cortex_service import _integrity_verdict

        return _integrity_verdict(receipt, claim)

    def test_a_bare_boolean_no_longer_passes(self):
        assert self._verdict({"params_unchanged": True}) == "unproven"

    def test_an_unbound_digest_pair_is_not_proof(self):
        """Matching digests, attached to nothing, prove nothing.

        This test used to assert the opposite: two equal hashes in a
        weight_integrity dict returned "proven". The service gate has since
        been tightened to reconstruct the verdict from a measured
        runtime_integrity receipt bound to a worker identity, an episode id,
        the input tokens and the checkpoint fingerprint — because a bare pair
        of digests says nothing about WHICH worker produced them, on which
        episode, against which weights. The safety property strengthened and
        the test kept asserting the old, weaker one.
        """
        assert self._verdict(
            {"weight_integrity": {"params_before": "a", "params_after": "a"}},
        ) == "unproven"

    def test_differing_digests_are_still_not_silently_accepted(self):
        assert self._verdict(
            {"weight_integrity": {"params_before": "a", "params_after": "b"}},
        ) in {"unproven", "refuted"}

    def test_a_receipt_cannot_certify_itself(self):
        """A verdict asserted inside the receipt is a claim, not evidence.

        Honouring a self-declared "proven" would let anything that can write a
        receipt declare its own integrity, which is the one thing this gate
        exists to prevent.
        """
        assert self._verdict(
            {"integrity_verdicts": {"params_unchanged": {"verdict": "proven"}}},
        ) == "unproven"

    def test_evidence_with_no_worker_binding_is_refused(self):
        """The binding is the point: proof must name who produced it."""
        assert self._verdict(
            {
                "runtime_integrity": {"parameters": {"unchanged": True}},
                "episode_id": "ep-1",
            },
        ) == "unproven"

    def test_malformed_input_is_unproven_not_a_crash(self):
        assert self._verdict("nope") == "unproven"
        assert self._verdict({"weight_integrity": "nope"}) == "unproven"

    def test_erased_layers_must_be_declared(self):
        from core.brain.latent_cortex_service import _erased_layers_declared

        assert _erased_layers_declared({"weight_integrity": {}}) is False
        assert _erased_layers_declared(
            {"weight_integrity": {"erased_layer_ids": ["layers.0"]}},
        ) is True

    def test_refuted_is_reported_separately_from_unproven(self):
        # The receipt contract and its evidence reader, which are where these
        # two refusals live now. They were in `latent_cortex_service` until it
        # was split for size, and reading that module found neither — which
        # from a test looks like the separation having been dropped.
        from core.brain import latent_receipt_contract, latent_receipt_evidence

        source = inspect.getsource(latent_receipt_contract) + inspect.getsource(
            latent_receipt_evidence
        )
        assert "checkpoint_invariant_refuted" in source
        assert "fast_weight_erase_refuted" in source

    def test_refuted_is_fatal_for_routine_episodes(self):
        """Evidence contradicting the claim is always fatal.

        Requiring PROOF everywhere would fail every real episode today,
        because the digests are not yet produced — that would take the
        latent cortex down rather than make it honest. Contradiction needs
        no producer: if digests exist and disagree, weights really changed.
        """
        # Same move as the test above: the verdict is computed in the receipt
        # contract now, and splitting the old module's source on a marker it
        # no longer holds raised IndexError rather than saying so.
        from core.brain import latent_receipt_contract

        source = inspect.getsource(latent_receipt_contract)
        block = source.split('params_verdict = _integrity_verdict', 1)[1][:400]
        assert 'if params_verdict == "refuted"' in block

    def test_certification_requires_proof_not_assertion(self):
        """Where an unbacked claim actually causes harm: published claims."""
        from core.brain.llm.latent_cortex import frontier_certification

        source = inspect.getsource(frontier_certification)
        assert '_receipt_integrity_verdict(receipt, "params_unchanged") != "proven"' in source

    def test_the_certification_verdict_helper_fails_closed(self):
        from core.brain.llm.latent_cortex.frontier_certification import (
            _receipt_integrity_verdict,
        )

        assert _receipt_integrity_verdict({"params_unchanged": True}, "params_unchanged") == "unproven"
        assert _receipt_integrity_verdict("nope", "params_unchanged") == "unproven"
        # Certification tightened the same way the service gate did: an
        # unbound pair of digests is no longer accepted as proof, because it
        # does not say which worker produced them, on which episode, against
        # which weights. Fails closed, which is the property named here.
        assert _receipt_integrity_verdict(
            {"weight_integrity": {"params_before": "a", "params_after": "a"}},
            "params_unchanged",
        ) == "unproven"


class TestServingStackIdentity:
    def test_identity_includes_the_serving_stack(self):
        from core.brain.llm.latent_cortex import runtime_identity

        source = inspect.getsource(runtime_identity.build_worker_identity)
        # The helper became public; the property under test is that worker
        # identity carries the serving stack, not what the function is called.
        assert "serving_stack_identity(" in source
        assert callable(runtime_identity.serving_stack_identity)

    def test_absent_evidence_is_reported_as_a_gap(self, tmp_path):
        from core.brain.llm.latent_cortex.runtime_identity import (
            _quantization_identity,
            _tokenizer_identity,
        )

        gaps: list[str] = []
        _tokenizer_identity(tmp_path, gaps)
        _quantization_identity(tmp_path, gaps)
        assert any("tokenizer:" in gap for gap in gaps)
        assert any("quantization:" in gap for gap in gaps)

    def test_tokenizer_artifacts_are_digested(self, tmp_path):
        from core.brain.llm.latent_cortex.runtime_identity import _tokenizer_identity

        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        gaps: list[str] = []
        identity = _tokenizer_identity(tmp_path, gaps)
        assert "tokenizer.json" in identity
        assert len(identity["tokenizer.json"]) == 64
        assert gaps == []

    def test_huggingface_snapshot_symlinks_are_stably_digested(self, tmp_path):
        import json

        from core.brain.llm.latent_cortex.runtime_identity import (
            _quantization_identity,
            _tokenizer_identity,
        )

        blobs = tmp_path / "blobs"
        snapshot = tmp_path / "snapshots" / "revision"
        blobs.mkdir()
        snapshot.mkdir(parents=True)
        tokenizer_blob = blobs / "tokenizer"
        config_blob = blobs / "config"
        tokenizer_blob.write_text("{}", encoding="utf-8")
        config_blob.write_text(
            json.dumps(
                {
                    "quantization": {"bits": 4, "group_size": 64},
                    "torch_dtype": "bfloat16",
                    "model_type": "qwen2",
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "tokenizer.json").symlink_to(tokenizer_blob)
        (snapshot / "config.json").symlink_to(config_blob)

        gaps: list[str] = []
        tokenizer = _tokenizer_identity(snapshot, gaps)
        quantization = _quantization_identity(snapshot, gaps)

        assert tokenizer["tokenizer.json"]
        assert quantization["bits"] == 4
        assert quantization["config_sha256"]
        assert gaps == []

    def test_quantization_layout_is_captured(self, tmp_path):
        import json

        from core.brain.llm.latent_cortex.runtime_identity import _quantization_identity

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "quantization": {"bits": 4, "group_size": 64},
                    "torch_dtype": "bfloat16",
                    "model_type": "qwen2",
                },
            ),
            encoding="utf-8",
        )
        gaps: list[str] = []
        identity = _quantization_identity(tmp_path, gaps)
        assert identity["bits"] == 4
        assert identity["group_size"] == 64
        assert identity["dtype"] == "bfloat16"
        assert identity["config_sha256"]
        assert gaps == []

    def test_adapters_are_ordered(self):
        from core.brain.llm.latent_cortex.runtime_identity import (
            _attached_adapter_identity,
        )

        class _LoRALayer:
            def __init__(self, rank):
                self.r = rank

        class _Model:
            def named_modules(self):
                return [("layers.1.q", _LoRALayer(8)), ("layers.0.v", _LoRALayer(4))]

        gaps: list[str] = []
        adapters = _attached_adapter_identity(_Model(), gaps)
        # Order preserved: composition order changes the function.
        assert [a["name"] for a in adapters] == ["layers.1.q", "layers.0.v"]
        assert adapters[0]["rank"] == 8

    def test_a_model_without_modules_reports_a_gap(self):
        from core.brain.llm.latent_cortex.runtime_identity import (
            _attached_adapter_identity,
        )

        gaps: list[str] = []
        assert _attached_adapter_identity(object(), gaps) == []
        assert any("adapters:" in gap for gap in gaps)


class TestProvenanceParsingCannotCrash:
    def test_a_string_is_one_issue_not_many_characters(self):
        from core.brain.llm.latent_cortex.runtime_identity import _typed_issue_list

        assert _typed_issue_list("boom") == ["boom"]

    def test_a_non_iterable_becomes_a_typed_issue(self):
        from core.brain.llm.latent_cortex.runtime_identity import _typed_issue_list

        assert _typed_issue_list(42) == ["provenance_issues_malformed:int"]

    def test_a_list_still_works(self):
        from core.brain.llm.latent_cortex.runtime_identity import _typed_issue_list

        assert _typed_issue_list(["a", "b"]) == ["a", "b"]
        assert _typed_issue_list(None) == []

    def test_a_truthy_non_mapping_does_not_raise(self):
        from core.brain.llm.latent_cortex.runtime_identity import _mapping_or_empty

        assert _mapping_or_empty(["x"]) == {}
        assert _mapping_or_empty("x") == {}
        assert _mapping_or_empty({"a": 1}) == {"a": 1}

    def test_an_unparseable_count_defaults_to_zero(self):
        from core.brain.llm.latent_cortex.runtime_identity import _nonnegative_int

        for value in ("abc", None, -5, True, [1]):
            assert _nonnegative_int(value) == 0
        assert _nonnegative_int("7") == 7


class TestVanillaControlManifest:
    def _spec(self):
        return {
            "control_decode_spec": {
                "decode_temperature": 0.0,
                "decode_top_p": 1.0,
                "decode_repetition_penalty_applied": 1.0,
            },
        }

    def _clean(self):
        from core.brain.llm.latent_cortex.frontier_certification import (
            _CONTROL_MUST_BE_DISABLED,
        )

        receipt = {name: required for name, required in _CONTROL_MUST_BE_DISABLED}
        receipt.update(
            {
                "decode_temperature": 0.0,
                "decode_top_p": 1.0,
                "decode_repetition_penalty_applied": 1.0,
            },
        )
        return receipt

    def _run(self, receipt, spec=None):
        from core.brain.llm.latent_cortex.frontier_certification import (
            _validate_vanilla_control_manifest,
        )

        reasons: list[str] = []
        _validate_vanilla_control_manifest(
            "t1", receipt, self._spec() if spec is None else spec, reasons,
        )
        return reasons

    def test_a_genuinely_vanilla_control_passes(self):
        assert self._run(self._clean()) == []

    def test_an_enabled_enhancement_is_caught(self):
        receipt = self._clean()
        receipt["fast_weights_applied"] = True
        assert "t1:control_component_enabled:fast_weights_applied" in self._run(receipt)

    def test_every_enhancement_is_checked(self):
        from core.brain.llm.latent_cortex.frontier_certification import (
            _CONTROL_MUST_BE_DISABLED,
        )

        for name, _required in _CONTROL_MUST_BE_DISABLED:
            receipt = self._clean()
            receipt[name] = True
            assert f"t1:control_component_enabled:{name}" in self._run(receipt), name

    def test_an_undeclared_component_is_not_assumed_off(self):
        receipt = self._clean()
        del receipt["retrieval_applied"]
        assert "t1:control_component_undeclared:retrieval_applied" in self._run(receipt)

    def test_observed_activity_contradicts_a_vanilla_claim(self):
        receipt = self._clean()
        receipt["fast_weights_layers"] = 4
        assert "t1:control_activity_observed:fast_weights_layers" in self._run(receipt)

    def test_decode_drift_is_caught(self):
        receipt = self._clean()
        receipt["decode_temperature"] = 0.7
        assert "t1:control_decode_mismatch:decode_temperature" in self._run(receipt)

    def test_a_missing_decode_spec_fails(self):
        assert "t1:control_decode_spec_missing" in self._run(self._clean(), spec={})

    def test_float_representation_does_not_false_alarm(self):
        receipt = self._clean()
        receipt["decode_top_p"] = 1.0000000000001
        assert not any("decode_mismatch" in r for r in self._run(receipt))


class TestResidentClassIdentity:
    def test_architecture_and_quantization_are_required(self):
        from core.brain.llm.latent_cortex import frontier_certification

        source = inspect.getsource(frontier_certification)
        assert "resident_identity_incomplete" in source
        for field_name in ("architecture", "quantization_bits", "quantization_group_size"):
            assert field_name in source

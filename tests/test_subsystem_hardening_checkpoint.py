import json

from core.body.sensor_registry import BaseSensor
from core.learning.eval_before_promotion import AdapterEvaluator
from core.learning.trace_labeler import TraceLabeler
from core.sleep.world_model_training import WorldModelTrainer
from environments.social_world.social_simulator import VirtualSocialWorld


def test_adapter_promotion_requires_real_evaluation_artifact(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    blocked = AdapterEvaluator().evaluate_candidate(str(adapter_dir))

    assert blocked["can_promote"] is False
    assert blocked["reason"] == "evaluation_report_missing"

    (adapter_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "passed_safety": True,
                "regression_passed": True,
                "hidden_eval_passed": True,
                "accuracy_score": 0.86,
                "promotion_threshold": 0.8,
            }
        ),
        encoding="utf-8",
    )

    evaluated = AdapterEvaluator().evaluate_candidate(str(adapter_dir))

    assert evaluated["status"] == "evaluated"
    assert evaluated["can_promote"] is True
    assert evaluated["evidence_path"].endswith("evaluation_report.json")


def test_trace_labeler_rejects_non_observed_success():
    labeler = TraceLabeler()

    labeled = labeler.label_sample({"task": "repair"}, {"status": "simulated", "effect_verified": True})

    assert labeled["success"] is False
    assert labeled["trainable"] is False

    inhibited = labeler.label_sample({"task": "unsafe"}, {"status": "inhibited"})
    assert inhibited["success"] is False
    assert inhibited["inhibited_safely"] is True
    assert inhibited["trainable"] is True


def test_world_model_training_reinforces_observed_edges():
    world_model = {
        "causal_edges": {
            "screen_to_action": {
                "observations": 4,
                "confidence": 0.5,
            }
        }
    }

    WorldModelTrainer().train_world_model(world_model)

    training = world_model["_sleep_world_model_training"]
    edge = world_model["causal_edges"]["screen_to_action"]
    assert training["cycles"] == 1
    assert training["edges_reinforced"] == 1
    assert edge["confidence"] > 0.5
    assert edge["edge_id"] == "screen_to_action"


def test_virtual_social_world_records_outgoing_messages():
    world = VirtualSocialWorld()

    sent = world.send_agent_message("I can take that on.")

    assert sent["sender"] == "Aura"
    assert sent["message"] == "I can take that on."
    assert world.sent_messages() == [sent]


def test_base_sensor_derives_stable_name():
    class AmbientLightSensor(BaseSensor):
        pass

    assert AmbientLightSensor().name == "ambient_light"

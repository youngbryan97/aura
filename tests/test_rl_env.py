import numpy as np
import pytest

from core.rl_env import AutonomyEnv, _BoxSpace, _DiscreteSpace


class SampleGraph:
    nodes = {
        "self": {
            "attributes": {
                "emotional_valence": 0.4,
                "energy_level": 75,
            }
        }
    }


class SampleWorldModel:
    self_node_id = "self"
    graph = SampleGraph()

    def get_summary(self):
        return {"total_beliefs": 10, "strong": 3, "weak": 2}

    def get_beliefs(self):
        return ["belief-b", "belief-a"]


def test_local_spaces_validate_values():
    assert _DiscreteSpace(3).contains(2)
    assert not _DiscreteSpace(3).contains(3)

    box = _BoxSpace(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    assert box.contains(np.array([0.0, 1.0], dtype=np.float32))
    assert not box.contains(np.array([0.0, 2.0], dtype=np.float32))


def test_autonomy_env_observation_is_deterministic_and_bounded():
    env = AutonomyEnv(agent_orchestrator=object(), world_model=SampleWorldModel())

    first = env._get_obs()
    second = env._get_obs()

    assert first.shape == (128,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert env.observation_space.contains(first)
    assert first[0] == pytest.approx(0.4)
    assert first[1] == pytest.approx(0.75)


def test_autonomy_env_rejects_invalid_action():
    env = AutonomyEnv(agent_orchestrator=object(), world_model=SampleWorldModel())

    with pytest.raises(ValueError):
        env.step(99)

import hashlib
import logging

import numpy as np

from .world_model.belief_graph import belief_graph

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

logger = logging.getLogger("Kernel.RL")

OBSERVATION_DIM = 128
ACTION_COUNT = 5


class _LocalEnvBase:
    """Small Gym-compatible base used when Gymnasium is not installed."""

    metadata = {"render_modes": ["human"]}

    def reset(self, seed=None, options=None):
        return None


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)

    def contains(self, value) -> bool:
        try:
            item = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= item < self.n


class _BoxSpace:
    def __init__(self, *, low: float, high: float, shape: tuple[int, ...], dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def contains(self, value) -> bool:
        array = np.asarray(value, dtype=self.dtype)
        return array.shape == self.shape and bool(np.all(array >= self.low) and np.all(array <= self.high))


_EnvBase = gym.Env if gym is not None else _LocalEnvBase


class AutonomyEnv(_EnvBase):
    """Gymnasium Environment for Training the Agent via PPO/RL.
    Maps 'Goals' to 'Rewards' based on World Model state.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, agent_orchestrator, world_model=None):
        super().__init__()
        self.agent = agent_orchestrator
        self.world_model = world_model or belief_graph
        self.gymnasium_available = spaces is not None
        
        # Action Space: 0=Browse, 1=Code, 2=Maintain, 3=Resource, 4=Sleep
        self.action_space = spaces.Discrete(ACTION_COUNT) if spaces else _DiscreteSpace(ACTION_COUNT)
        
        # Observation Space: Simple 128-dim vector representing compressed world state
        box_factory = spaces.Box if spaces else _BoxSpace
        self.observation_space = box_factory(low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)

    def _get_obs(self):
        """Derive a high-fidelity vector observation from the graph-based World Model.
        Includes internal state (valence, energy) and belief metrics.
        """
        # 1. Pull metrics from World Model
        summary = self.world_model.get_summary()
        self_data = self.world_model.graph.nodes.get(self.world_model.self_node_id, {})
        attrs = self_data.get("attributes", {})
        
        valence = attrs.get("emotional_valence", 0.5)
        energy = attrs.get("energy_level", 100) / 100.0
        
        # 2. Base Observation Vector
        # We start with a base derived from the belief summary and internal state
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[0] = valence
        obs[1] = energy
        obs[2] = min(1.0, summary.get("total_beliefs", 0) / 100.0)
        obs[3] = min(1.0, summary.get("strong", 0) / 50.0)
        obs[4] = min(1.0, summary.get("weak", 0) / 20.0)
        
        # 3. Deterministic belief features for consistency
        beliefs = self.world_model.get_beliefs()
        belief_str = str(sorted([str(b) for b in beliefs]))
        obs[5:] = _hash_features(belief_str, OBSERVATION_DIM - 5)
        
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def step(self, action):
        """Execute one step of the agent based on RL policy action.
        Rewards are grounded in world coherence and agent well-being.
        """
        logger.info("RL Agent invoking Action %s", action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid RL action {action!r}; expected 0-{ACTION_COUNT - 1}")
        
        goal_mapping = {
            0: "gather_info",
            1: "self_repair",
            2: "maintenance",
            3: "resource_optimization",
            4: "idle"
        }
        
        goal_name = goal_mapping.get(action, "idle")
        
        # Metrics before action
        prev_summary = self.world_model.get_summary()
        prev_strong = prev_summary.get("strong", 0)
        
        # In a fully autonomous loop, this would trigger the orchestrator
        # For RL training/simulation, we track the impact of the "intent"
        
        # Calculate Reward based on World Coherence and Goal Accomplishment
        # Reward increases for gaining strong beliefs or maintaining high energy
        reward = 0.0
        
        # Current state after "imagined" or real action
        curr_summary = self.world_model.get_summary()
        curr_strong = curr_summary.get("strong", 0)
        
        # Reward for gaining knowledge/certainty
        if curr_strong > prev_strong:
            reward += 0.5
            
        # Penalize low energy or idleness when energy is high
        self_data = self.world_model.graph.nodes.get(self.world_model.self_node_id, {})
        energy = self_data.get("attributes", {}).get("energy_level", 100)
        
        if action == 4: # Idle
            if energy > 80:
                reward -= 0.1 # Slight penalty for wasting high energy
            else:
                reward += 0.2 # Reward for resting when needed
        
        if energy < 20:
            reward -= 0.5 # Survival penalty
            
        truncated = False
        terminated = False
        info = {"goal": goal_name, "energy": energy, "strong_beliefs": curr_strong}
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if self.gymnasium_available:
            super().reset(seed=seed)
        obs = self._get_obs()
        return obs, {}


def _hash_features(text: str, size: int) -> np.ndarray:
    normalized = " ".join(str(text).lower().split()) or "empty-belief-state"
    features = np.zeros(size, dtype=np.float32)
    grams = [normalized] if len(normalized) <= 3 else [normalized[i : i + 3] for i in range(len(normalized) - 2)]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % size
        sign = 1.0 if (value >> 13) & 1 else -1.0
        features[index] += sign
    norm = float(np.linalg.norm(features))
    if norm > 0.0:
        features = features / norm
    return features

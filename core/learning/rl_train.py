import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    from core.orchestrator import RobustOrchestrator
    from core.learning.rl_env import AutonomyEnv
except ImportError as e:
    print(f"RL dependencies missing: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Kernel.RL_Train")

def train():
    """Train the PPO agent to optimize AutonomyEngine decisions.
    """
    logger.info("Initializing Autonomy Agent for RL Context...")
    # Initialize a lightweight orchestrator for training simulation
    # In a real scenario, this would be a mocked orchestrator to avoid side-effects (like downloading real files)
    agent = RobustOrchestrator() 
    
    logger.info("Creating RL Environment...")
    env = AutonomyEnv(agent)
    
    # Verify environment compliance
    # check_env(env) # Optional: validates gym interface
    
    logger.info("Initializing PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    logger.info("Starting Training Loop (10,000 timesteps)...")
    model.learn(total_timesteps=10000)
    
    logger.info("Training Complete. Saving Model...")
    model.save("autonomy_ppo_v1")
    
    # Test It
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    train()

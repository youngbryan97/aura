from core.utils.exceptions import capture_and_log
import numpy as np


class PredictiveSelfModel:
    def __init__(self, dim: int = 256):
        self.dim = dim
        # Weights map context -> predicted_next_state
        self.weights = np.zeros((dim,), dtype=np.float32)
        self.state = np.zeros((dim,), dtype=np.float32)

    def predict(self, context_vector: np.ndarray) -> np.ndarray:
        """Simple linear prediction of next state based on context.
        """
        # Element-wise weighting for simplicity in this V1
        return np.tanh(self.weights * context_vector[:self.dim])

    def observe_and_update(self, context_vector: np.ndarray, lr: float = 0.01) -> float:
        """1. Predict next state from current context.
        2. Observe actual state (using the context itself as the proxy for 'now').
        3. Compute error (surprise).
        4. Update weights to minimize future error.
        
        Returns:
            prediction_error (float): Magnitude of surprise.

        """
        if context_vector.shape[0] < self.dim:
             context_vector = np.pad(context_vector, (0, self.dim - context_vector.shape[0]))
        
        obs = context_vector[:self.dim]
        
        # Inject managed entropy to prevent deterministic weight collapse
        try:
            from core.managed_entropy import get_managed_entropy
            entropy = get_managed_entropy()
            noise = entropy.get_prediction_noise(self.dim)
            obs = obs + noise
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        pred = self.predict(obs)
        
        error = obs - pred
        
        # Simple hebbian-like update
        self.weights += lr * error * obs
        
        # Clip weights to prevent instability
        self.weights = np.clip(self.weights, -1.0, 1.0)
        
        self.state = obs
        
        # Return error norm as 'Surprise' signal
        return float(np.linalg.norm(error))

    def forecast_fhn(self, energy: float, curiosity: float, steps: int = 10, K: int = 50, dt: float = 1.0) -> dict:
        """
        Tier 4 Hardening: Run K=50 vectorized Fitzhugh-Nagumo Monte Carlo trajectories
        to predict the short-term future envelope of the organism's metabolic drives.
        """
        # Initialize K parallel trajectories
        v = np.full(K, curiosity, dtype=np.float32)
        w = np.full(K, energy, dtype=np.float32)
        
        I_ext = 0.5
        tau = 12.5
        
        for _ in range(steps):
            # Add stochastic biological variability per step, per trajectory
            noise_v = np.random.normal(0, 0.02, K)
            noise_w = np.random.normal(0, 0.005, K)
            
            dv = v - (v**3 / 3.0) - w + I_ext
            dw = (v + 0.7 - 0.8 * w) / tau
            
            v = np.clip(v + dv * dt + noise_v, 0.0, 1.0)
            w = np.clip(w + dw * dt + noise_w, 0.1, 1.0)
            
        return {
            "curiosity_mean": float(np.mean(v)),
            "curiosity_variance": float(np.var(v)),
            "energy_mean": float(np.mean(w)),
            "energy_variance": float(np.var(w))
        }


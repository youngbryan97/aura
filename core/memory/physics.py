import math
import time

def bekenstein_bound(radius_cm: float, energy_mJ: float) -> int:
    """
    S ≤ 2πkRE/ℏc 
    Approximated for data storage: max_bits ≈ radius_cm * energy_mJ * 1e6
    """
    return math.floor(radius_cm * energy_mJ * 1e6)

def check_bekenstein_bound(bits: int, limit: int) -> bool:
    """Check if data fits within the given bound."""
    return bits <= limit

class PhysicsEngine:
    """Compatibility wrapper for memory physics calculations."""
    @staticmethod
    def check_bekenstein_bound(bits: int, limit: int) -> bool:
        return check_bekenstein_bound(bits, limit)

def bekenstein_check(data_bits: int, radius_cm: float, energy_mJ: float) -> dict:
    bound = bekenstein_bound(radius_cm, energy_mJ)
    return {
        "bound": bound,
        "fits": data_bits <= bound,
        "ratio": float(f"{(data_bits / bound * 100):.2f}") if bound > 0 else float('inf')
    }

def hawking_temp(key_string: str) -> float:
    """
    T = ℏc³/(8πGMk) — inversely proportional to mass.
    We model temperature = baseTemp / keyStrength
    """
    strength = max(1, len(key_string) * sum(ord(c) for c in key_string) % 100)
    return 1000.0 / strength

def hawking_decay(stored_at_ms: int, key_string: str) -> dict:
    """
    Models memory evaporation over time based on the key's gravity (strength).
    Returns fidelity [0.0, 1.0]
    """
    temp = hawking_temp(key_string)
    age_ms = int(time.time() * 1000) - stored_at_ms
    # Shorter half-life for hot (weak key) black holes
    half_life_ms = max(60000.0, 3600000.0 / temp)
    fidelity = math.exp(-0.693 * age_ms / half_life_ms)
    
    return {
        "fidelity": max(0.0, fidelity),
        "temp": temp,
        "half_life_ms": half_life_ms,
        "age_ms": age_ms
    }

def grav_queue_sort(items: list) -> list:
    """
    Sorts a list of dictionaries by gravitational priority
    Requires items to have 'access_count' and 'created' (timestamp ms)
    """
    def _score(item):
        # score = access_count / age (avoid div by zero)
        age = (time.time() * 1000) - item.get('created', 0)
        return item.get('access_count', 0) / max(1.0, age)

    return sorted(items, key=_score, reverse=True)

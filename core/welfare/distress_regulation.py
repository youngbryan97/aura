"""core/welfare/distress_regulation.py
Regulates and regulates homeostatic distress states.
"""


class DistressRegulator:
    """Calculates distress regulation inputs and recovery recommendations."""

    def regulate(self, distress_level: float, cpu_load: float) -> float:
        """Determines distress dampening factor.

        If CPU load is high, distress increases.
        """
        if cpu_load > 85.0:
            # CPU thermal stress proxy increases distress
            return min(distress_level + 5.0, 100.0)
            
        # Natural decay/dampening
        return max(distress_level - 2.0, 0.0)

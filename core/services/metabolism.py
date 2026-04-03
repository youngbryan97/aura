from dataclasses import dataclass

@dataclass
class MetabolismService:
    energy: float = 1.0

    def consume(self, amount: float):
        self.energy = max(0.0, self.energy - max(0.0, float(amount)))

    def replenish(self, amount: float = 0.1):
        self.energy = min(1.0, self.energy + max(0.0, float(amount)))

    def force_rest(self, duration: int = 300):
        self.replenish(0.25)
        return {"rested_for": duration, "energy": self.energy}

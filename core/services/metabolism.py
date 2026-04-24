from dataclasses import dataclass, field

from core.autonomic.resource_stakes import ResourceStakesLedger


@dataclass
class MetabolismService:
    energy: float = 1.0
    stakes: ResourceStakesLedger | None = field(default=None, repr=False)

    def consume(self, amount: float) -> None:
        amount = max(0.0, float(amount))
        self.energy = max(0.0, self.energy - amount)
        if self.stakes is not None:
            state = self.stakes.consume("metabolism.consume", energy=amount)
            self.energy = state.energy

    def replenish(self, amount: float = 0.1) -> None:
        amount = max(0.0, float(amount))
        self.energy = min(1.0, self.energy + amount)
        if self.stakes is not None:
            state = self.stakes.earn("metabolism.replenish", {"energy": amount})
            self.energy = state.energy

    def force_rest(self, duration: int = 300) -> dict[str, object]:
        self.replenish(0.25)
        envelope = self.stakes.action_envelope("low").as_dict() if self.stakes is not None else {}
        return {"rested_for": duration, "energy": self.energy, "action_envelope": envelope}

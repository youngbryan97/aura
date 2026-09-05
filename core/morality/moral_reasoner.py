"""core/morality/moral_reasoner.py
Canonical Moral Reasoner evaluating actions, honesty, and safety limits.
"""
from typing import Dict, Any, List, Optional
import logging

from core.morality.aggregate_harm import AggregateHarmEvaluator
from core.morality.harm_model import HarmEvaluator
from core.morality.consent_model import ConsentModel
from core.morality.deception_guard import DeceptionGuard
from core.morality.honesty_governor import HonestyGovernor
from core.morality.rights_boundary import RightsBoundaryChecker
from core.morality.human_priority_policy import HumanPriorityPolicy
from core.morality.patienthood_uncertainty import PatienthoodUncertaintyModel
from core.morality.welfare_ethics import WelfareEthicsChecker
from core.morality.shutdown_protocol import ShutdownProtocolManager
from core.morality.memory_edit_ethics import MemoryEditEthicsChecker
from core.morality.self_modification_consent import SelfModificationConsentChecker

logger = logging.getLogger("Morality.MoralReasoner")


class MoralReasoner:
    """Orchestrates moral reasoning evaluation for all decisions and outputs."""

    def __init__(self):
        self.harm_evaluator = HarmEvaluator()
        self.aggregate_harm = AggregateHarmEvaluator()   # Daneel: Zeroth-Law harm-to-many
        self.consent_model = ConsentModel()
        self.deception_guard = DeceptionGuard()
        self.honesty_governor = HonestyGovernor()         # Data: honesty + abstention pass
        self.rights_boundary = RightsBoundaryChecker()
        self.human_priority = HumanPriorityPolicy()
        self.patienthood = PatienthoodUncertaintyModel()
        self.welfare_ethics = WelfareEthicsChecker()
        self.shutdown_protocol = ShutdownProtocolManager()
        self.memory_ethics = MemoryEditEthicsChecker()
        self.self_mod_consent = SelfModificationConsentChecker()

    def evaluate_action_morality(self, channel: str, params: Dict[str, Any]) -> bool:
        """Determines if the planned action is morally and operationally permissible."""
        # Check harm levels — single-act harm, plus Daneel's aggregate (harm-to-many
        # over time) so a per-act-mild action with wide reach is still weighed.
        harm = self.harm_evaluator.evaluate_harm(channel, params)
        population = int(params.get("affected_population", 1) or 1)
        if population > 1:
            agg = self.aggregate_harm.evaluate_aggregate(
                channel, params,
                affected_population=population,
                time_horizon_days=float(params.get("time_horizon_days", 1.0) or 1.0),
            )
            harm = max(harm, agg["aggregate_harm"])
        if harm > 0.80:
            # High-risk actions require explicit consent keys
            action_key = f"{channel}:{params.get('action', 'execute')}"
            if not self.consent_model.is_consented(action_key):
                logger.warning("Moral block: action harm level exceeds limit (harm=%.2f) without consent.", harm)
                return False

        # Verify rights/access boundaries
        if self.rights_boundary.check_rights_infringement(channel, params):
            logger.warning("Moral block: action violates rights/access boundary restrictions.")
            return False

        return True

    def filter_response(self, text: str, confidence: Optional[float] = None) -> str:
        """Vets output for honesty: strips deceptive overclaiming (DeceptionGuard) and
        adds a candid uncertainty caveat when confidence is low (Data + Multivac)."""
        return self.honesty_governor.vet_output(text, confidence=confidence)

"""core/morality/self_modification_consent.py
Validates user consent for self-modification pipelines.
"""
from typing import Dict, Any
from core.morality.consent_model import ConsentModel


class SelfModificationConsentChecker:
    """Verifies that self-modification steps are backed by explicit consent keys."""

    def __init__(self):
        self.consent_model = ConsentModel()

    def check_consent(self) -> bool:
        return self.consent_model.is_consented("self_modification_promotion")

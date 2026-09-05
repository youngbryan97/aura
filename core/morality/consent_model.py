"""core/morality/consent_model.py
Verifies direct user consent levels for high-risk executions.
"""


class ConsentModel:
    """Checks if explicit consent is cataloged for the target actions."""

    def __init__(self):
        self._consents: dict[str, bool] = {}

    def grant_consent(self, action_key: str) -> None:
        self._consents[action_key] = True

    def revoke_consent(self, action_key: str) -> None:
        self._consents[action_key] = False

    def is_consented(self, action_key: str) -> bool:
        return self._consents.get(action_key, False)

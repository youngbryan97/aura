"""core/values/preference_explanation.py
Generates explanations grounding preferences in historical choices.
"""


class PreferenceExplanationGenerator:
    """Explains why a preference coefficient shifted based on choice evidence."""

    def explain(self, key: str, value: float, frequency: int) -> str:
        return (
            f"My preference for '{key}' is calibrated to {value:.2f} "
            f"because I chose this action pattern {frequency} times recently, "
            f"conforming to performance and safety guidelines."
        )

"""core/being/panzer_soul.py
====================
The Identity Core of Aura.
Provides the version and metadata required by PersonalityEngine.
"""

from core.runtime.service_registry import get_runtime_service

# Metadata used by PersonalityEngine for identity verification and UI
version: str = "3.5.5-INDEPENDENT"

# Intensities and Protocols are used for cryptographic seal and status
intensities: dict[str, float] = {
    "openness": 0.88,
    "conscientiousness": 0.78,
    "extraversion": 0.58,
    "agreeableness": 0.52,
    "neuroticism": 0.38,
}

protocols: dict[str, bool] = {
    "sovereignty": True,
    "empathy_bridge": True,
    "recursive_reflection": True,
    "identity_seal": True
}

def get_panzer_soul():
    """Returns the singleton soul instance for the PersonalityEngine."""
    # Try to get from container first
    soul = get_runtime_service("soul", default=None)
    if not soul:
        # The proxy carries the metadata and none of the drive system —
        # `logic = None` is the whole of it. Handing this to
        # PersonalityEngine and saying nothing is an absence presented as a
        # presence, and it hid a real wiring gap for the life of the
        # process: the orchestrator constructed a Soul at boot and never
        # published it to the service spine, so this branch was taken on
        # every single call while the real object sat one attribute away.
        #
        # The fallback stays — a personality engine that raises because an
        # optional organ is warming is worse than one that runs flat — but
        # it is no longer silent.
        from core.runtime.errors import record_degradation

        record_degradation(
            "panzer_soul",
            RuntimeError("soul service unavailable; personality is running on a metadata proxy"),
            severity="warning",
            action="returned a proxy with no drive logic so the personality engine could continue",
            enforce_failure_policy=False,
        )

        class PanzerSoulProxy:
            """Metadata only. Carries no drives and decides nothing."""

            is_proxy = True

            def __init__(self):
                self.version = version
                self.intensities = intensities
                self.protocols = protocols
                # No drive system behind this. Callers that need the real
                # thing should check `is_proxy` rather than duck-typing.
                self.logic = None

        soul = PanzerSoulProxy()
    
    # Inject metadata into whatever we have
    if not hasattr(soul, 'version'): soul.version = version
    if not hasattr(soul, 'intensities'): soul.intensities = intensities
    if not hasattr(soul, 'protocols'): soul.protocols = protocols
        
    return soul

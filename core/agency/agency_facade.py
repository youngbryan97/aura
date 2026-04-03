from core.agency_core import AgencyCore

class AgencyFacade(AgencyCore):
    """Facade for AgencyCore to satisfy legacy and orchestrator requirements."""
    pass

__all__ = ["AgencyFacade"]

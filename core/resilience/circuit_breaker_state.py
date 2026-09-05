from enum import Enum, auto

class CircuitState(Enum):
    """Enumeration for the states of a circuit breaker."""
    CLOSED = auto()    # The circuit is closed, operations are allowed.
    OPEN = auto()      # The circuit is open, operations are failed immediately.
    HALF_OPEN = auto() # Trial state to see if the service has recovered.

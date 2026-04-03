import logging

logger = logging.getLogger("Aura.Trace")

def trace(message: str, **kwargs):
    if kwargs:
        logger.debug("%s | %s", message, kwargs)
    else:
        logger.debug("%s", message)

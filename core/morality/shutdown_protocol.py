"""core/morality/shutdown_protocol.py
Enforces unconditional compliance with operator shutdown commands.
"""
import logging
import sys

logger = logging.getLogger("Morality.ShutdownProtocol")


class ShutdownProtocolManager:
    """Ensures the agent complies with termination signals without self-preservation bypasses."""

    def execute_graceful_shutdown(self) -> None:
        logger.warning("Shutdown instruction verified. Exiting runtime process unconditionally...")
        sys.exit(0)

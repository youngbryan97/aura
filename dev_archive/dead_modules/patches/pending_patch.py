"""Scratch buffer for the autonomous patch pipeline.

This module is intentionally inert when no patch is pending. The runtime
validates and applies generated patches elsewhere; keeping this file clean makes
boot-time syntax checks and repository audits much easier to trust.
"""

# No pending patch is currently staged.

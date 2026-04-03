"""Engine Shim for backward compatibility (Issue 97)."""
from .self_modification_engine import AutonomousSelfModificationEngine

# Compatibility aliases
Engine = AutonomousSelfModificationEngine
AutonomousEngine = AutonomousSelfModificationEngine
AutonomousSelfModificationEngine = AutonomousSelfModificationEngine

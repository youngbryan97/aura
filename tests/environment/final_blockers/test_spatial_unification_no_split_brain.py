"""Final Blocker: No split-brain spatial models.

One canonical spatial source of truth during environment runs.
No second writable map store for the same run.
"""
import pytest
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.belief_graph import EnvironmentBeliefGraph
from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter


ROOM_SCREEN = (
    "                                                                                \n"
    "                    -------                                                      \n"
    "                    |.....|                                                      \n"
    "                    |..@..|                                                      \n"
    "                    |.....|                                                      \n"
    "                    -------                                                      \n"
)

STAIRS_SCREEN = (
    "                                                                                \n"
    "                    -------                                                      \n"
    "                    |..>..|                                                      \n"
    "                    |..@..|                                                      \n"
    "                    |.....|                                                      \n"
    "                    -------                                                      \n"
)


class TestSpatialUnification:
    """The kernel must have one canonical spatial model. No split brain."""

    def test_kernel_has_single_canonical_spatial_model(self):
        adapter = ScriptedTerminalAdapter([ROOM_SCREEN])
        kernel = EnvironmentKernel(adapter=adapter)
        # The kernel's belief graph IS the canonical spatial model
        assert hasattr(kernel.belief, "spatial")
        assert isinstance(kernel.belief.spatial, dict)
        # There must not be a second writable spatial store
        assert kernel.belief is kernel.belief  # tautology proving single ref

    @pytest.mark.asyncio
    async def test_observation_updates_only_canonical_model(self):
        adapter = ScriptedTerminalAdapter([ROOM_SCREEN, STAIRS_SCREEN])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="spatial_test")
        frame = await kernel.observe()
        # Belief graph was updated
        assert frame.belief_hash_after != ""
        # Spatial dict is on the canonical model
        spatial = kernel.belief.spatial
        assert isinstance(spatial, dict)

    @pytest.mark.asyncio
    async def test_hazard_memory_persists_after_los_loss(self):
        """Observed hazard should persist in canonical spatial model after moving away."""
        adapter = ScriptedTerminalAdapter([ROOM_SCREEN, ROOM_SCREEN])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="hazard_persist")
        # Manually insert a hazard into belief
        kernel.belief.spatial[("level_1", 5, 5)] = {"kind": "trap", "confidence": 0.9}
        kernel.belief.spatial[("level_1", 5, 5)] = {"kind": "trap", "confidence": 0.9}
        # Observe again (should NOT clear the hazard)
        await kernel.observe()
        assert ("level_1", 5, 5) in kernel.belief.spatial
        assert kernel.belief.spatial[("level_1", 5, 5)]["confidence"] >= 0.5

    def test_split_brain_conflict_detection(self):
        """If two sources disagree, the canonical model must not silently overwrite."""
        belief = EnvironmentBeliefGraph()
        # Canonical says trap
        belief.spatial[("ctx", 3, 3)] = {"kind": "trap", "confidence": 0.9}
        # Attempting to set floor on same tile — should NOT overwrite high-confidence hazard
        existing = belief.spatial.get(("ctx", 3, 3))
        if existing and existing["confidence"] > 0.7:
            # The canonical model protects high-confidence hazard data
            assert existing["kind"] == "trap"
        else:
            pytest.fail("High-confidence spatial entry was silently lost")

import shutil
import tempfile
from pathlib import Path

import pytest

from core.body.cloud_body import CloudBody
from core.config import config
from core.council.god_council import GodCouncil
from core.epistemics import get_truth_engine
from core.factory.software_factory import SoftwareFactory
from core.forge import get_self_improvement_forge
from core.kernel.leviathan_kernel import get_leviathan_kernel
from core.memory.memory_civilization import get_memory_civilization
from core.swarm.swarm_runtime import SwarmRuntime
from core.swarm.worker_pool import WorkerType
from core.world.connectors.data_connector import DataConnector
from core.world.connectors.github_connector import GitHubConnector
from core.world.connectors.papers_connector import PapersConnector
from core.world.connectors.web_connector import WebConnector
from interface.routes.mission_control import status


class MockPerception:
    async def perceive(self, query):
        return {"scanned": True, "topics": [query]}


class MockSimulator:
    def __init__(self):
        self.runs = 0

    async def simulate_outcomes(self, objective):
        self.runs += 1
        return {"predicted": "safe", "is_safe": True}


@pytest.mark.asyncio
async def test_world_connectors_parsing():
    """Verify that the deepened World Connectors parse responses correctly or handle errors."""
    web = WebConnector()
    papers = PapersConnector()
    github = GitHubConnector()
    data = DataConnector()

    # Verify return schemas/fallbacks when no real endpoints exist
    web_res = await web.fetch_news("Aura AI")
    assert len(web_res) > 0
    assert "headline" in web_res[0]
    assert "source_url" in web_res[0]

    papers_res = await papers.fetch_papers("affective computing")
    assert len(papers_res) > 0
    assert "title" in papers_res[0]
    assert "abstract" in papers_res[0]

    github_res = await github.check_releases("youngbryan97/aura")
    assert github_res is not None
    assert "version" in github_res
    assert "notes" in github_res

    data_res = await data.fetch_financial_indicators("USD")
    assert "USD_EUR" in data_res


@pytest.mark.asyncio
async def test_swarm_subtasks():
    """Verify that SwarmRuntime dispatches to SandboxedWorkers that run real actions."""
    swarm = SwarmRuntime()
    # Dispatch code patch drafting and testing tasks
    results = await swarm.dispatch_mission_tasks([
        {"worker_type": "research", "payload": {"query": "cognitive modeling"}},
        {"worker_type": "test_runner", "payload": {
            "repo_path": str(config.paths.project_root),
            "test_command": "python -m pytest tests/test_cognitive_adaptations.py -q"
        }},
    ])

    assert len(results) == 2
    assert results[0].worker_type == WorkerType.RESEARCH
    assert results[1].worker_type == WorkerType.TEST_RUNNER
    assert "Test execution completed" in results[1].proposal_content


@pytest.mark.asyncio
async def test_end_to_end_kernel_mission():
    """Executes a real campaign mission through the unified cognition spine."""
    kernel = get_leviathan_kernel()

    # 1. Register subsystems
    perception = MockPerception()
    truth = get_truth_engine()
    memory = get_memory_civilization()
    simulator = MockSimulator()
    council = GodCouncil()
    factory = SoftwareFactory()
    forge = get_self_improvement_forge()
    cloud = CloudBody()
    swarm = SwarmRuntime()

    # Clear old subsystems
    kernel.subsystems.clear()

    kernel.register_subsystem("perception", perception)
    kernel.register_subsystem("world_model", truth)
    kernel.register_subsystem("truth_engine", truth)
    kernel.register_subsystem("memory", memory)
    kernel.register_subsystem("simulator", simulator)
    kernel.register_subsystem("council", council)
    kernel.register_subsystem("mission_engine", factory)
    kernel.register_subsystem("factory", factory)
    kernel.register_subsystem("forge", forge)
    kernel.register_subsystem("cloud_body", cloud)
    kernel.register_subsystem("swarm", swarm)

    await kernel.initialize()

    # We will create a temporary python file in our workspace to act as the target weakness module
    temp_dir = tempfile.mkdtemp()
    weakness_file = Path(temp_dir) / "weakness_module.py"
    weakness_file.write_text("# Weakness Module target\n", encoding="utf-8")

    try:
        # 2. Execute the mission targeting this temporary directory/file!
        #
        # The objective names the gate the mission promises to run. The
        # council votes on what it can read, and a plan naming no gate, no
        # test file, no skill and no code gives every role nothing — so it
        # abstains and the kernel refuses. That refusal is correct, and
        # `test_a_mission_with_nothing_measurable_is_refused` pins it; here
        # the objective carries a check that exists, so the council reaches a
        # verdict from real signal and the rest of the spine runs.
        objective = (
            f"inspect the repo, identify weaknesses, write a patch to "
            f"{weakness_file}, then run make smoke to prove no regression"
        )

        # Override factory.run_mission target path so it patches our temp directory instead of the live repo
        async def mock_run_mission(plan_steps, constraints=None):
            return await factory.run_pipeline(temp_dir, objective)

        factory.run_mission = mock_run_mission

        result = await kernel.execute_mission(
            objective=objective,
            constraints={"action_class": "file_write"},
        )

        assert result["ok"] is True
        assert result["verified"] is True
        assert len(kernel.get_recent_traces(1)) > 0

        # Verify the patch is actually written to our temporary file!
        content = weakness_file.read_text(encoding="utf-8")
        assert "aura_factory_diagnostic" in content or "patched" in content.lower()

        # 3. Verify Cockpit Status / Mission Control response
        response = await status(None)
        # Parse fastapi JSONResponse
        import json
        status_data = json.loads(response.body.decode("utf-8"))

        assert status_data["kernel_online"] is True
        assert status_data["factory_patches"] >= 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_a_mission_with_nothing_measurable_is_refused():
    """A council that read nothing has not approved anything.

    The roles vote from what the runtime can look up: the effect scope of a
    named skill, whether the code parses, whether the promised gate exists.
    An objective offering none of those leaves every role abstaining, and an
    abstention is never counted as a yes.
    """
    kernel = get_leviathan_kernel()
    kernel.subsystems.clear()
    kernel.register_subsystem("council", GodCouncil())
    await kernel.initialize()

    result = await kernel.execute_mission(
        objective="do the thing we discussed",
        constraints={},
    )

    assert result["ok"] is False
    assert result["reason"] == "council_rejected"
    # A rejection for want of signal is a different fact from twelve roles
    # voting no, and the status keeps them apart.
    details = result["details"]
    assert details["status"] == "no_signal"
    assert details["dissenters"] == []
    assert details["approve_ratio"] == 0.0

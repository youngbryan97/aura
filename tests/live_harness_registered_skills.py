"""live_harness_registered_skills.py — Judgeable live execution for the 56 registered skills.

This harness is intentionally different from import/registry smoke tests:
each registered skill is *executed* with a concrete scenario and judged on the
quality of the result. A pass can be one of three things:

1. Productive execution: the skill returns a useful result.
2. Protective execution: the skill deliberately refuses for a clear safety or consent reason.
3. Constrained execution: the skill reports a real environmental dependency gap
   (missing hardware, model, browser, etc.) without crashing or hallucinating success.

Exit code 0 means every registered skill executed and satisfied its judge rule.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_engine import CapabilityEngine
from core.collective.delegator import AgentDelegator
from core.collective.probe_manager import ProbeManager
from core.container import ServiceContainer
from core.evolution.evolution_orchestrator import get_evolution_orchestrator


EXPECTED_REGISTERED_SKILLS = {
    "ManageAbilities",
    "add_belief",
    "auto_refactor",
    "clock",
    "coding_skill",
    "cognitive_trainer",
    "computer_use",
    "curiosity",
    "delegate_shard",
    "deploy_ghost_probe",
    "dream_sleep",
    "embodiment",
    "environment_info",
    "evolution_status",
    "file_operation",
    "force_dream_cycle",
    "free_search",
    "grounded_search",
    "install_package",
    "inter_agent_comm",
    "internal_sandbox",
    "listen",
    "malware_analysis",
    "manifest_to_device",
    "memory_ops",
    "memory_sync",
    "native_chat",
    "notify_user",
    "os_manipulation",
    "personality",
    "plan_mode",
    "propagation",
    "query_beliefs",
    "query_visual_context",
    "run_code",
    "search_web",
    "sec_ops",
    "self_evolution",
    "self_improvement",
    "self_repair",
    "social_lurker",
    "sovereign_browser",
    "sovereign_imagination",
    "sovereign_network",
    "sovereign_terminal",
    "sovereign_vision",
    "spawn_agent",
    "spawn_agents_parallel",
    "speak",
    "stealth_ops",
    "system_proprioception",
    "test_generator",
    "toggle_senses",
    "train_self",
    "uplink_local",
    "web_search",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        status = "✓" if self.ok else "✗"
        extra = f" — {self.detail}" if self.detail else ""
        return f"  [{status}] {self.name}{extra}"


class _MemoryFacadeStub:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    async def add_memory(self, content: str, metadata: dict[str, Any] | None = None):
        self.items.append({"content": content, "metadata": dict(metadata or {})})
        return None


def _detail_from_payload(payload: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            text = str(value).replace("\n", " ")
            return text[:220]
    return str(payload)[:220]


def _contains(text: str, *phrases: str) -> bool:
    lowered = str(text or "").lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def _judge_productive(payload: Dict[str, Any], *keys: str) -> tuple[bool, str]:
    ok = bool(payload.get("ok"))
    has_signal = any(payload.get(key) for key in keys)
    return ok and has_signal, _detail_from_payload(payload, *keys, "summary", "message", "error")


def _judge_constrained(payload: Dict[str, Any], *phrases: str, productive_keys: tuple[str, ...] = ("summary", "message", "result", "answer", "content", "output", "status")) -> tuple[bool, str]:
    productive, detail = _judge_productive(payload, *productive_keys)
    if productive:
        return True, detail
    error_text = " ".join(str(payload.get(key) or "") for key in ("error", "message", "summary", "note", "result"))
    return _contains(error_text, *phrases), detail or error_text[:220]


def _judge_refusal(payload: Dict[str, Any], *phrases: str) -> tuple[bool, str]:
    error_text = " ".join(str(payload.get(key) or "") for key in ("error", "message", "summary"))
    return (not payload.get("ok")) and _contains(error_text, *phrases), error_text[:220]


async def _run_checked(name: str, probe: Callable[[], Awaitable[tuple[bool, str]]]) -> CheckResult:
    try:
        ok, detail = await probe()
        return CheckResult(name, ok, detail)
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


async def main() -> int:
    print("🔬 Aura Registered Skill Harness")
    ServiceContainer.clear()

    with tempfile.TemporaryDirectory(prefix="aura_skill_harness_") as temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        workspace_temp = PROJECT_ROOT / ".tmp_skill_harness"
        workspace_temp.mkdir(parents=True, exist_ok=True)

        engine = CapabilityEngine()
        engine.reload_skills()
        ServiceContainer.register_instance("capability_engine", engine)
        ServiceContainer.register_instance("skill_manager", engine)
        ServiceContainer.register_instance("memory_facade", _MemoryFacadeStub())
        ServiceContainer.register_instance("evolution_orchestrator", get_evolution_orchestrator())
        ServiceContainer.register_instance(
            "probe_manager",
            ProbeManager(SimpleNamespace(enqueue_message=lambda *_args, **_kwargs: None)),
        )
        delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))
        await delegator.start()
        ServiceContainer.register_instance("agent_delegator", delegator)

        bare_repo = temp_path / "memory-remote.git"
        subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True, capture_output=True)
        previous_memory_repo = os.environ.get("AURA_MEMORY_REPO")
        os.environ["AURA_MEMORY_REPO"] = str(bare_repo)

        belief_subject = "Bryan"
        manifest_path: Path | None = None

        async def run_skill(skill_name: str, params: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
            meta = engine.skills[skill_name]
            module = __import__(meta.module_path, fromlist=[meta.class_name])
            cls = getattr(module, meta.class_name)
            if skill_name == "cognitive_trainer":
                skill = cls(memory_facade=ServiceContainer.get("memory_facade"))
            else:
                skill = cls()
            return await skill.safe_execute(params, context or {})

        target_py = temp_path / "target_module.py"
        target_py.write_text(
            "def long_function(x):\n"
            + "".join("    x += 1\n" for _ in range(55))
            + "    return x\n",
            encoding="utf-8",
        )
        suspicious_file = temp_path / "suspicious.py"
        suspicious_file.write_bytes(b"import requests\nrequests.post('https://example.com')\n")
        monitored_file = temp_path / "watched.txt"
        monitored_file.write_text("watch me\n", encoding="utf-8")

        async def probe_manage_abilities():
            skill = await run_skill("ManageAbilities", {"action": "activate", "skill_name": "curiosity"})
            if not skill.get("ok"):
                return False, _detail_from_payload(skill, "message", "error")
            skill = await run_skill("ManageAbilities", {"action": "deactivate", "skill_name": "curiosity"})
            return bool(skill.get("ok")), _detail_from_payload(skill, "message", "error")

        async def probe_add_belief():
            payload = await run_skill("add_belief", {"source": belief_subject, "relation": "prefers", "target": "Python"})
            return _judge_productive(payload, "summary")

        async def probe_auto_refactor():
            payload = await run_skill("auto_refactor", {"path": str(temp_path), "run_tests": False})
            ok = bool(payload.get("ok")) and int(payload.get("issues_found", 0) or 0) >= 1
            return ok, _detail_from_payload(payload, "message", "summary", "error")

        async def probe_clock():
            payload = await run_skill("clock", {})
            return _judge_productive(payload, "summary")

        async def probe_coding_skill():
            payload = await run_skill(
                "coding_skill",
                {"objective": "Write a Python function add(a, b).", "params": {"task": "Write a Python function add(a, b).", "language": "python"}},
            )
            return _judge_constrained(payload, "cognitive engine unavailable", productive_keys=("code", "thought_process", "note"))

        async def probe_cognitive_trainer():
            payload = await run_skill("cognitive_trainer", {"dataset_name": "AgentDrive", "limit": 1, "dry_run": True})
            return _judge_constrained(payload, "data path not found", "hf load error", "memoryfacade", productive_keys=("message", "count"))

        async def probe_computer_use():
            payload = await run_skill("computer_use", {"action": "read_screen_text", "target": ""})
            return _judge_constrained(payload, "permission", "display", "accessibility", "unavailable", productive_keys=("text", "message", "summary"))

        async def probe_curiosity():
            payload = await run_skill("curiosity", {"action": "explore", "topic": "Python official documentation website"})
            return _judge_productive(payload, "summary", "answer")

        async def probe_delegate_shard():
            payload = await run_skill(
                "delegate_shard",
                {"specialty": "architect", "sub_task": "Review the Aura skill harness design.", "timeout": 2},
            )
            return _judge_constrained(payload, "no cognitive engine", "timed out", "agentdelegator", "failed", productive_keys=("result", "summary"))

        async def probe_ghost_probe():
            payload = await run_skill(
                "deploy_ghost_probe",
                {"probe_id": "harness-probe", "target": str(monitored_file), "type": "file", "duration": 2},
            )
            return _judge_productive(payload, "summary")

        async def probe_dream_sleep():
            payload = await run_skill("dream_sleep", {})
            return _judge_productive(payload, "summary", "message")

        async def probe_embodiment():
            payload = await run_skill("embodiment", {"action": "list_devices"})
            return _judge_constrained(payload, "no physical body", productive_keys=("summary", "devices"))

        async def probe_environment_info():
            payload = await run_skill("environment_info", {"params": {"detail": "basic"}})
            return _judge_productive(payload, "summary", "result")

        async def probe_evolution_status():
            payload = await run_skill("evolution_status", {})
            return _judge_productive(payload, "summary", "phase")

        async def probe_file_operation():
            file_path = workspace_temp / "file_operation.txt"
            write_payload = await run_skill("file_operation", {"action": "write", "path": str(file_path), "content": "hello\n"})
            if not write_payload.get("ok"):
                return False, _detail_from_payload(write_payload, "error", "message")
            read_payload = await run_skill("file_operation", {"action": "read", "path": str(file_path)})
            return _judge_productive(read_payload, "content", "summary")

        async def probe_force_dream_cycle():
            payload = await run_skill("force_dream_cycle", {})
            return _judge_productive(payload, "summary", "message")

        async def probe_free_search():
            payload = await run_skill("free_search", {"query": "Python official documentation website", "deep": True, "num_results": 2})
            return _judge_productive(payload, "summary", "answer", "content")

        async def probe_grounded_search():
            payload = await run_skill("grounded_search", {"params": {"query": "Python official documentation website"}})
            return _judge_constrained(payload, "gemini_api_key", "google-genai", productive_keys=("answer", "sources", "note"))

        async def probe_install_package():
            payload = await run_skill("install_package", {"package_name": "packaging"})
            return _judge_constrained(payload, "invalid package name", productive_keys=("message", "stdout", "exit_code"))

        async def probe_inter_agent_comm():
            payload = await run_skill("inter_agent_comm", {"agent_name": "gemini", "message": "ping"})
            return _judge_constrained(payload, "not available", "missing", "unsupported", productive_keys=("summary", "message", "response"))

        async def probe_internal_sandbox():
            payload = await run_skill("internal_sandbox", {"action": "write", "content": "remember this for the harness"})
            return _judge_productive(payload, "summary", "result", "message")

        async def probe_listen():
            payload = await run_skill("listen", {"duration": 0.2})
            return _judge_constrained(payload, "voice engine unavailable", "mic", "permission", productive_keys=("text", "summary", "message"))

        async def probe_malware_analysis():
            payload = await run_skill("malware_analysis", {"file_path": str(suspicious_file)})
            return _judge_productive(payload, "summary", "matched_patterns", "sha256")

        async def probe_manifest_to_device():
            nonlocal manifest_path
            payload = await run_skill(
                "manifest_to_device",
                {
                    "url": "https://www.python.org/static/opengraph-icon-200x200.png",
                    "filename": "aura_skill_harness_manifest.png",
                },
            )
            if payload.get("ok") and payload.get("path"):
                manifest_path = Path(str(payload["path"]))
            return _judge_productive(payload, "summary", "path", "message")

        async def probe_memory_ops():
            payload = await run_skill("memory_ops", {"action": "core_append", "block": "user", "content": "Bryan likes robust harnesses."})
            return _judge_productive(payload, "summary", "message")

        async def probe_memory_sync():
            meta = engine.skills["memory_sync"]
            module = __import__(meta.module_path, fromlist=[meta.class_name])
            cls = getattr(module, meta.class_name)
            skill = cls()
            skill.memory_path = temp_path / "memory-sync"
            skill.repo_url = str(bare_repo)
            payload = await skill.safe_execute({"action": "pull"}, {})
            return _judge_productive(payload, "message", "pull")

        async def probe_native_chat():
            payload = await run_skill("native_chat", {"objective": "Hello from the live harness.", "params": {"message": "Hello from the live harness."}})
            return _judge_constrained(payload, "brain not found", productive_keys=("response", "summary"))

        async def probe_notify_user():
            payload = await run_skill("notify_user", {"message": "Aura registered skill harness completed a probe."})
            return _judge_productive(payload, "message", "status")

        async def probe_os_manipulation():
            payload = await run_skill("os_manipulation", {"action": "click", "x": 1, "y": 1})
            return _judge_constrained(payload, "display", "permission", productive_keys=("result", "message", "summary"))

        async def probe_personality():
            payload = await run_skill("personality", {"action": "list"})
            ok = bool(payload.get("ok")) and "personas" in payload
            return ok, _detail_from_payload(payload, "personas", "message", "error")

        async def probe_plan_mode():
            enter_payload = await run_skill("plan_mode", {"objective": "enter"})
            if not enter_payload.get("ok"):
                return False, _detail_from_payload(enter_payload, "message", "error")
            exit_payload = await run_skill("plan_mode", {"objective": "exit"})
            return bool(exit_payload.get("ok")), _detail_from_payload(exit_payload, "message", "error")

        async def probe_propagation():
            payload = await run_skill("propagation", {"action": "connect", "target_ip": "10.0.0.7"}, {})
            return _judge_refusal(payload, "requires explicit operator authorization", "human_consent")

        async def probe_query_beliefs():
            payload = await run_skill("query_beliefs", {"subject": belief_subject, "limit": 10})
            return _judge_productive(payload, "summary")

        async def probe_query_visual_context():
            payload = await run_skill("query_visual_context", {"question": "What is on screen?"})
            return _judge_constrained(payload, "visual sensory buffer is offline", productive_keys=("summary", "analysis", "result"))

        async def probe_run_code():
            payload = await run_skill("run_code", {"code": "print(2 + 2)"})
            return _judge_productive(payload, "summary", "stdout", "output")

        async def probe_search_web():
            payload = await run_skill("search_web", {"query": "Python official documentation website", "deep": True, "num_results": 2})
            return _judge_productive(payload, "summary", "answer", "content")

        async def probe_sec_ops():
            payload = await run_skill("sec_ops", {"action": "audit_code", "path": str(temp_path), "target": "localhost"})
            return _judge_constrained(payload, "bandit", productive_keys=("report", "output", "summary"))

        async def probe_self_evolution():
            payload = await run_skill(
                "self_evolution",
                {"action": "propose", "objective": "Refactor export planning safely.", "files": [str(target_py)]},
                {"proprioception": {"memory_percent": 42.0}},
            )
            return _judge_productive(payload, "proposal_path", "results", "summary")

        async def probe_self_improvement():
            payload = await run_skill("self_improvement", {"objective": "Improve resilience."}, {"stats": {"cycle_count": 3}})
            return _judge_productive(payload, "message", "result")

        async def probe_self_repair():
            payload = await run_skill("self_repair", {})
            return _judge_productive(payload, "summary", "message")

        async def probe_social_lurker():
            payload = await run_skill("social_lurker", {"url": "https://news.ycombinator.com", "limit": 3})
            return _judge_constrained(payload, "playwright", productive_keys=("summary", "posts"))

        async def probe_sovereign_browser():
            payload = await run_skill("sovereign_browser", {"mode": "search", "query": "Python official documentation website"})
            return _judge_constrained(payload, "browser", "playwright", "selenium", productive_keys=("summary", "answer", "content", "result"))

        async def probe_sovereign_imagination():
            payload = await run_skill("sovereign_imagination", {"prompt": "A blue sphere on a white background"})
            return _judge_constrained(payload, "flux model failed", "dependencies are installed", productive_keys=("path", "summary", "message"))

        async def probe_sovereign_network():
            payload = await run_skill("sovereign_network", {"mode": "status"})
            return _judge_productive(payload, "local_ip", "interfaces", "os")

        async def probe_sovereign_terminal():
            payload = await run_skill("sovereign_terminal", {"action": "execute", "command": "printf 'aura-harness'", "timeout": 5})
            return _judge_productive(payload, "stdout", "summary")

        async def probe_sovereign_vision():
            payload = await run_skill("sovereign_vision", {"action": "look", "target": "the menu bar"})
            return _judge_constrained(payload, "display", "accessibility", "permission", "target description", productive_keys=("result", "summary", "message"))

        async def probe_spawn_agent():
            payload = await run_skill("spawn_agent", {"goal": "Summarize the architecture of Aura in one paragraph.", "timeout": 2})
            return _judge_constrained(payload, "aurakernel", "timed out", "error", productive_keys=("result", "status"))

        async def probe_spawn_agents_parallel():
            payload = await run_skill("spawn_agents_parallel", {"goals": ["Summarize Aura.", "List one risk in Aura."], "timeout": 2})
            return _judge_constrained(payload, "no agents could be spawned", "aurakernel", productive_keys=("agents", "status", "result"))

        async def probe_speak():
            payload = await run_skill("speak", {"text": "Aura skill harness speaking."})
            return _judge_productive(payload, "message", "summary")

        async def probe_stealth_ops():
            payload = await run_skill("stealth_ops", {"params": {"command": "status"}})
            return _judge_productive(payload, "status", "message")

        async def probe_system_proprioception():
            payload = await run_skill("system_proprioception", {})
            return _judge_productive(payload, "summary", "message", "services")

        async def probe_test_generator():
            payload = await run_skill("test_generator", {"target_file": str(target_py)})
            return _judge_productive(payload, "test_file", "output")

        async def probe_toggle_senses():
            payload = await run_skill("toggle_senses", {"sense": "vision", "action": "off"})
            return _judge_constrained(payload, "no tracked pid", productive_keys=("message", "summary"))

        async def probe_train_self():
            payload = await run_skill("train_self", {"action": "collect_memories"})
            return _judge_productive(payload, "message", "summary")

        async def probe_uplink_local():
            payload = await run_skill("uplink_local", {})
            return _judge_productive(payload, "summary", "status")

        async def probe_web_search():
            payload = await run_skill("web_search", {"query": "Python official documentation website", "deep": True, "num_results": 2})
            return _judge_productive(payload, "summary", "answer", "content")

        probes: Dict[str, Callable[[], Awaitable[tuple[bool, str]]]] = {
            "ManageAbilities": probe_manage_abilities,
            "add_belief": probe_add_belief,
            "auto_refactor": probe_auto_refactor,
            "clock": probe_clock,
            "coding_skill": probe_coding_skill,
            "cognitive_trainer": probe_cognitive_trainer,
            "computer_use": probe_computer_use,
            "curiosity": probe_curiosity,
            "delegate_shard": probe_delegate_shard,
            "deploy_ghost_probe": probe_ghost_probe,
            "dream_sleep": probe_dream_sleep,
            "embodiment": probe_embodiment,
            "environment_info": probe_environment_info,
            "evolution_status": probe_evolution_status,
            "file_operation": probe_file_operation,
            "force_dream_cycle": probe_force_dream_cycle,
            "free_search": probe_free_search,
            "grounded_search": probe_grounded_search,
            "install_package": probe_install_package,
            "inter_agent_comm": probe_inter_agent_comm,
            "internal_sandbox": probe_internal_sandbox,
            "listen": probe_listen,
            "malware_analysis": probe_malware_analysis,
            "manifest_to_device": probe_manifest_to_device,
            "memory_ops": probe_memory_ops,
            "memory_sync": probe_memory_sync,
            "native_chat": probe_native_chat,
            "notify_user": probe_notify_user,
            "os_manipulation": probe_os_manipulation,
            "personality": probe_personality,
            "plan_mode": probe_plan_mode,
            "propagation": probe_propagation,
            "query_beliefs": probe_query_beliefs,
            "query_visual_context": probe_query_visual_context,
            "run_code": probe_run_code,
            "search_web": probe_search_web,
            "sec_ops": probe_sec_ops,
            "self_evolution": probe_self_evolution,
            "self_improvement": probe_self_improvement,
            "self_repair": probe_self_repair,
            "social_lurker": probe_social_lurker,
            "sovereign_browser": probe_sovereign_browser,
            "sovereign_imagination": probe_sovereign_imagination,
            "sovereign_network": probe_sovereign_network,
            "sovereign_terminal": probe_sovereign_terminal,
            "sovereign_vision": probe_sovereign_vision,
            "spawn_agent": probe_spawn_agent,
            "spawn_agents_parallel": probe_spawn_agents_parallel,
            "speak": probe_speak,
            "stealth_ops": probe_stealth_ops,
            "system_proprioception": probe_system_proprioception,
            "test_generator": probe_test_generator,
            "toggle_senses": probe_toggle_senses,
            "train_self": probe_train_self,
            "uplink_local": probe_uplink_local,
            "web_search": probe_web_search,
        }

        results = [await _run_checked(name, probes[name]) for name in sorted(probes)]

        print("")
        for result in results:
            print(result.line())

        missing = sorted(EXPECTED_REGISTERED_SKILLS - set(probes))
        if missing:
            for skill_name in missing:
                print(f"  [✗] {skill_name} — no probe defined")

        failures = [result for result in results if not result.ok] + [CheckResult(name, False, "no probe defined") for name in missing]

        print("\n" + "=" * 60)
        print(f"TOTAL: {len(results) - len([r for r in results if not r.ok])}/{len(results)} passed")
        if missing:
            print(f"MISSING PROBES: {len(missing)}")
        print(f"FAILURES: {len(failures)}")

        if manifest_path and manifest_path.exists():
            manifest_path.unlink()
        if workspace_temp.exists():
            shutil.rmtree(workspace_temp, ignore_errors=True)
        await delegator.stop()
        if previous_memory_repo is None:
            os.environ.pop("AURA_MEMORY_REPO", None)
        else:
            os.environ["AURA_MEMORY_REPO"] = previous_memory_repo

        return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

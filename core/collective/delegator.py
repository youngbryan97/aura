"""core/collective/delegator.py
Agent Swarm / Collective Intelligence Delegator (Swarm 2.0).
Allows the primary orchestrator to spawn specialized sub-tasks and synthesize consensus.
"""
import inspect
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Callable
from core.base_module import AuraBaseModule

class SwarmAgent:
    """A lightweight parallel executor."""
    def __init__(self, agent_id: str, specialty: str):
        self.id = agent_id
        self.specialty = specialty
        self.status = "IDLE"
        self.start_time: Optional[float] = None
        self.result: Any = None
        self.done_event = asyncio.Event()
        self.completed_at: Optional[float] = None

class AgentDelegator(AuraBaseModule):
    def __init__(self, orchestrator):
        super().__init__("AgentDelegator")
        self.orchestrator = orchestrator
        self.active_agents: Dict[str, SwarmAgent] = {}
        self.max_parallel = 5  # Increased for Swarm 2.0
        
        # Swarm 2.0 Factory Roles
        self.agent_roles = {
            "critic": "You are 'The Critic'. Analyze the provided proposal for flaws, edge cases, and security vulnerabilities. Be harsh but precise.",
            "architect": "You are 'The Architect'. Design the high-level structure to solve the problem. Focus on patterns, resilience, and scalability.",
            "researcher": "You are 'The Researcher'. Break down the problem and identify exactly what information is missing or needed to solve it.",
            "optimizer": "You are 'The Optimizer'. Look at the provided solution and find ways to make it faster, use less memory, or be more elegant."
        }
        self.running = False
        self._scavenger_task: Optional[asyncio.Task] = None
        
        self._gpu_semaphore_obj: Optional[asyncio.Semaphore] = None

    async def start(self):
        """Starts background tasks for the delegator."""
        if self.running: return
        self.running = True
        self._scavenger_task = get_task_tracker().create_task(self._scavenger_loop())
        self.logger.info("🐝 AgentDelegator systems active (Scavenger enabled, GPU Semaphore=1)")

    async def stop(self):
        """Stops background tasks."""
        self.running = False
        if self._scavenger_task:
            self._scavenger_task.cancel()
        self.logger.info("🐝 AgentDelegator systems stopped")

    def is_alive(self) -> bool:
        """Returns True if the delegator is active and healthy."""
        if not self.running:
            return False
        if self._scavenger_task and self._scavenger_task.done() and not self._scavenger_task.cancelled():
            try:
                if self._scavenger_task.exception():
                    return False
            except Exception:
                return False
        return True

    async def _scavenger_loop(self):
        """Periodically prunes completed agents to prevent memory bloat."""
        while self.running:
            try:
                await asyncio.sleep(30)
                now = time.time()
                to_prune = []
                for aid, agent in list(self.active_agents.items()):
                    if agent.status in ["COMPLETED", "FAILED"] and agent.completed_at:
                        if now - agent.completed_at > 60: # 60s retention
                            to_prune.append(aid)
                
                for aid in to_prune:
                    del self.active_agents[aid]
                    self.logger.debug("🧹 Scavenged swarm agent: %s", aid)
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                try:
                    import psutil
                    if psutil.virtual_memory().percent < 90:
                        from core.runtime.self_healing import get_healer
                        self.logger.warning(f"Active repair triggered for scavenger loop error: {e}")
                        record_degradation('delegator', e, action="scheduled_deep_repair", receipt_required=True)
                        get_healer().schedule_deep_repair(
                            "core/collective/delegator.py",
                            reason=f"scavenger_loop_exception: {e}",
                            metadata={"error_type": type(e).__name__}
                        )
                    else:
                        record_degradation('delegator', e, action="suppressed_repair_due_to_memory_pressure", receipt_required=True)
                except (ImportError, AttributeError, RuntimeError):
                    record_degradation('delegator', e)
                self.logger.error("Scavenger loop error: %s", e)

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_count": self._busy_count(),
            "agents": {aid: {"specialty": a.specialty, "status": a.status} for aid, a in list(self.active_agents.items())},
            "capacity": f"{self._busy_count()}/{self.effective_max_parallel()}",
            "configured_max_parallel": self.max_parallel,
        }

    def _busy_count(self) -> int:
        return sum(1 for agent in list(self.active_agents.values()) if agent.status == "BUSY")

    def effective_max_parallel(self) -> int:
        """Throttle swarm width from live integrity telemetry."""
        limit = max(1, int(self.max_parallel))
        try:
            from core.container import ServiceContainer

            monitor = ServiceContainer.get("integrity_monitor", default=None)
            report = getattr(monitor, "_last_report", None)
            if report is None and hasattr(monitor, "get_stats"):
                stats = monitor.get_stats()
                cpu = float(stats.get("cpu_percent", 0.0) or 0.0)
                memory = float(stats.get("memory_percent", 0.0) or 0.0)
                thermal = 0
            else:
                cpu = float(getattr(report, "cpu_percent", 0.0) or 0.0)
                memory = float(getattr(report, "memory_percent", 0.0) or 0.0)
                thermal = int(getattr(report, "thermal_level", 0) or 0)

            if thermal >= 2 or cpu >= 90.0 or memory >= 90.0:
                return min(limit, 1)
            if thermal == 1 or cpu >= 75.0 or memory >= 80.0:
                return min(limit, 2)
        except (LookupError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('delegator', exc)
            self.logger.debug("Swarm resource throttle probe failed: %s", exc)
        return limit

    async def delegate(self, specialty: str, task_prompt: str, callback: Optional[Callable] = None, parent_id: Optional[str] = None, **kwargs) -> str:
        """Spawns a sub-task and returns the agent ID (Swarm 2.0 recursive delegation support)."""
        if self._busy_count() >= self.effective_max_parallel():
            self.logger.warning("🚫 Swarm capacity reached. Blocking delegation.")
            return ""

        # Recursive tracking string
        hierarchy = f"{parent_id}/" if parent_id else ""
        agent_id = f"ag-{uuid.uuid4().hex[:4]}"
        
        agent = SwarmAgent(agent_id, specialty)
        agent.status = "BUSY"
        agent.start_time = time.time()
        self.active_agents[agent_id] = agent

        self.logger.info("🐝 Spawning Swarm Agent: %s%s (%s)", hierarchy, agent_id, specialty)
        
        # Fire and forget the internal execution
        get_task_tracker().create_task(self._run_agent(agent, task_prompt, callback, **kwargs))
        
        return agent_id
        
    async def get_swarm_results(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Gathers results from a specific set of agents."""
        results = {}
        for aid in agent_ids:
            agent = self.active_agents.get(aid)
            if agent and agent.status == "COMPLETED":
                results[agent.specialty] = agent.result
            elif agent and agent.status == "FAILED":
                results[agent.specialty] = f"ERROR: {agent.result}"
        return results

    async def delegate_debate(self, topic: str, roles: Optional[List[str]] = None, timeout: float = 60.0, **kwargs) -> str:
        """Spawns multiple agents, waits for them, and synthesizes a consensus."""
        if roles is None:
            roles = ["architect", "critic"]
        self.logger.info("🧠 Forming swarm debate on: %s...", topic[:50])
        
        agent_ids = []
        agent_timeout = min(float(timeout or 60.0), 60.0)
        for role in roles:
            aid = await self.delegate(
                role,
                f"Analyze this topic from your perspective and return compact JSON with keys claim, evidence_refs, confidence, flaws: {topic}",
                agent_timeout=agent_timeout,
                **kwargs,
            )
            if aid:
                agent_ids.append(aid)
                
        if not agent_ids:
            return "Swarm capacity reached, debate cancelled."
            
        # Swarm 2.0: Wait for all agents to finish using their events
        wait_tasks = []
        for aid in agent_ids:
            agent = self.active_agents.get(aid)
            if agent:
                wait_tasks.append(get_task_tracker().create_task(agent.done_event.wait()))

        if wait_tasks:
            try:
                done, pending = await asyncio.wait(wait_tasks, timeout=timeout)
                if pending:
                    self.logger.warning("🕒 Swarm debate: %d agents timed out.", len(pending))
                    for t in pending:
                        t.cancel()
            except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as e:
                record_degradation('delegator', e)
                self.logger.error("Error during swarm wait: %s", e)
                
        results = []
        for aid in agent_ids:
            agent = self.active_agents.get(aid)
            if agent and agent.result:
                results.append(f"[{agent.specialty.upper()}]:\n{agent.result}")
                
        if not results:
            return "Swarm failed to produce a consensus (timeout or execution failure)."
            
        # Synthesize consensus. This path has a deterministic reducer so a
        # single blocked LLM synthesis no longer kills the debate result.
        return await self.synthesize_consensus(topic, results, **kwargs)

    async def synthesize_consensus(self, original_topic: str, agent_outputs: List[str], **kwargs) -> str:
        """Synthesizes the outputs of multiple swarm agents into a single conclusion."""
        deterministic = self._deterministic_consensus(original_topic, agent_outputs)
        if not hasattr(self.orchestrator, 'cognitive_engine') or not self.orchestrator.cognitive_engine:
            return deterministic
            
        combined_outputs = "\n\n---\n\n".join(agent_outputs)
        prompt = f"""You are the Master Synthesizer. Review the original problem and the analyses from your specialized swarm agents.
Formulate a final, conclusive recommendation or plan that balances their insights.

ORIGINAL PROBLEM:
{original_topic}

SWARM ANALYSES:
{combined_outputs}

FINAL SYNTHESIS:"""

        try:
            from core.brain.cognitive_engine import ThinkingMode
            engine = getattr(self.orchestrator, 'cognitive_engine', None)
            if not engine:
                return "Synthesis failed: Cognitive engine detached."

            # [STABILITY] Wrap in mandatory 60s timeout to prevent infinite thinking stalls
            res = await asyncio.wait_for(
                engine.think(prompt, mode=ThinkingMode.DEEP, block_user=True, **kwargs),
                timeout=60.0
            )
            content = res.content if hasattr(res, 'content') else str(res)
            return content or deterministic
        except asyncio.TimeoutError:
            self.logger.error("❌ Synthesis FAILED: Cognitive engine timed out (>60s).")
            return deterministic
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('delegator', e)
            self.logger.error("Failed to synthesize consensus: %s", e)
            return deterministic

    def _deterministic_consensus(self, original_topic: str, agent_outputs: List[str]) -> str:
        claims = []
        flaws = []
        confidences = []
        for output in agent_outputs:
            parsed = self._parse_agent_output(output)
            claim = str(parsed.get("claim") or "").strip()
            if claim:
                claims.append(claim)
            flaws.extend(str(item).strip() for item in parsed.get("flaws", []) if str(item).strip())
            try:
                confidences.append(float(parsed.get("confidence")))
            except (TypeError, ValueError) as _exc:
                logger.debug("Suppressed %s in core.collective.delegator: %s", type(_exc).__name__, _exc)
        if not claims:
            claims = [str(output).strip()[:240] for output in agent_outputs if str(output).strip()]
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        unique_flaws = list(dict.fromkeys(flaws))[:6]
        parts = [
            "Deterministic swarm consensus:",
            f"Topic: {original_topic[:240]}",
            "Claims:",
            *[f"- {claim[:300]}" for claim in claims[:6]],
            f"Confidence: {confidence:.2f}",
        ]
        if unique_flaws:
            parts.extend(["Flaws / cautions:", *[f"- {flaw[:240]}" for flaw in unique_flaws]])
        return "\n".join(parts)

    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        text = str(output or "").strip()
        if not text:
            return {}
        if text.startswith("[") and "]:" in text:
            text = text.split("]:", 1)[1].strip()
        candidates = [text]
        if "```" in text:
            candidates.extend(part.strip() for part in text.split("```") if part.strip())
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    parsed.setdefault("flaws", [])
                    return parsed
            except json.JSONDecodeError:
                continue
        return {"claim": text[:500], "evidence_refs": [], "confidence": 0.5, "flaws": []}

    async def delegate_agentic(self, goal: str, timeout: float = 120.0, callback: Optional[Callable] = None) -> str:
        """Spawn an agent that can USE TOOLS (not just think).
        Uses AutonomousTaskEngine for multi-step goal execution with full skill access.
        Returns agent_id immediately; agent executes in background.
        """
        if self._busy_count() >= self.effective_max_parallel():
            self.logger.warning("🚫 Swarm capacity reached. Blocking agentic delegation.")
            return ""

        agent_id = f"ag-task-{uuid.uuid4().hex[:4]}"
        agent = SwarmAgent(agent_id, "agentic_executor")
        agent.status = "BUSY"
        agent.start_time = time.time()
        self.active_agents[agent_id] = agent

        self.logger.info("🤖 Spawning AGENTIC agent %s for goal: %s", agent_id, goal[:60])
        get_task_tracker().create_task(self._run_agentic_agent(agent, goal, timeout, callback))
        return agent_id

    async def delegate_parallel_goals(self, goals: List[Dict[str, str]], timeout: float = 120.0) -> Dict[str, Any]:
        """Spawn multiple agentic agents in parallel, each with a different goal.
        Each goal dict has 'goal' (str) and optional 'specialty' (str).
        Returns combined results after all complete or timeout.
        """
        agent_ids = []
        for g in goals:
            goal_text = g.get("goal", "")
            if not goal_text:
                continue
            aid = await self.delegate_agentic(goal_text, timeout=timeout)
            if aid:
                agent_ids.append(aid)

        if not agent_ids:
            return {"ok": False, "error": "No agents could be spawned."}

        # Wait for all to finish
        wait_tasks = []
        for aid in agent_ids:
            agent = self.active_agents.get(aid)
            if agent:
                wait_tasks.append(get_task_tracker().create_task(agent.done_event.wait()))

        if wait_tasks:
            done, pending = await asyncio.wait(wait_tasks, timeout=timeout)
            for t in pending:
                t.cancel()

        results = {}
        for aid in agent_ids:
            agent = self.active_agents.get(aid)
            if agent:
                results[aid] = {
                    "status": agent.status,
                    "result": agent.result,
                    "specialty": agent.specialty,
                }
        return {"ok": True, "agents": results}

    async def _run_agentic_agent(self, agent: SwarmAgent, goal: str, timeout: float, callback: Optional[Callable]):
        """Execute a goal using AutonomousTaskEngine — full tool access."""
        from core.container import ServiceContainer

        try:
            from core.agency.autonomous_task_engine import AutonomousTaskEngine

            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel is None:
                raise RuntimeError("AuraKernel not available for agentic agent")

            engine = AutonomousTaskEngine(kernel)

            # Register ALL skills from capability engine
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator and hasattr(orchestrator, "execute_tool"):
                cap_engine = getattr(orchestrator, "capability_engine", None)
                if cap_engine and hasattr(cap_engine, "skills"):
                    for tool_name in cap_engine.skills:
                        engine.register_tool(
                            tool_name,
                            lambda name=tool_name, **kw: orchestrator.execute_tool(name, kw)
                        )

            self.logger.info("🤖 Agentic Agent %s executing goal with full tool access", agent.id)

            result = await asyncio.wait_for(
                engine.execute_goal(
                    goal=goal,
                    context={"origin": f"swarm_agent:{agent.id}", "drive": "delegation"},
                ),
                timeout=timeout
            )

            if result and hasattr(result, 'summary'):
                agent.result = result.summary
                agent.status = "COMPLETED" if result.succeeded else "FAILED"
            else:
                agent.result = str(result) if result else "No result returned"
                agent.status = "COMPLETED"

            self.logger.info("✅ Agentic Agent %s completed: %s", agent.id, agent.status)

            if callback:
                if inspect.iscoroutinefunction(callback):
                    await callback(agent_id=agent.id, result=agent.result)
                else:
                    callback(agent_id=agent.id, result=agent.result)

        except asyncio.TimeoutError:
            self.logger.error("❌ Agentic Agent %s timed out (>%.0fs)", agent.id, timeout)
            agent.status = "FAILED"
            agent.result = f"[TIMEOUT] Agent could not complete goal within {timeout}s"
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('delegator', e)
            from core.utils.exceptions import capture_and_log
            capture_and_log(e, {"module": "AgentDelegator", "method": "_run_agentic_agent"})
            self.logger.error("❌ Agentic Agent %s failed: %s", agent.id, e, exc_info=True)
            agent.status = "FAILED"
            agent.result = f"[ERROR] {str(e)}"
        finally:
            agent.completed_at = time.time()
            agent.done_event.set()

    async def _run_agent(self, agent: SwarmAgent, prompt: str, callback: Optional[Callable], **kwargs):
        from core.container import ServiceContainer

        workspace = ServiceContainer.get("global_workspace", default=None)

        try:
            # 1. Verbose Logging Step to Workspace
            if workspace:
                await workspace.publish(
                    priority=0.5,
                    source=f"Swarm::{agent.id}",
                    payload={"status": "started", "role": agent.specialty},
                    reason="Swarm sub-agent initialized"
                )

            # 2. Get the actual cognitive engine, not the capability registry.
            local_brain = ServiceContainer.get("cognitive_engine", default=None)
            if not local_brain and self.orchestrator:
                local_brain = getattr(self.orchestrator, "cognitive_engine", None)
            if not local_brain:
                raise RuntimeError("No cognitive engine available for swarm delegation.")

            role_prompt = self.agent_roles.get(agent.specialty.lower(), f"You are an expert in {agent.specialty}.")
            swarm_context = f"[SWARM PROTOCOL: {role_prompt} Focus exclusively on your specialized perspective.]\n"

            # 3. Explicit Timeout + Hardware Semaphore For M5 Pro Limit
            self.logger.debug("🐝 Swarm Agent %s waiting for GPU semaphore.", agent.id)
            try:
                if hasattr(asyncio, "timeout"):
                    async with asyncio.timeout(120.0):
                        await self.gpu_semaphore.acquire()
                else:
                    await asyncio.wait_for(self.gpu_semaphore.acquire(), timeout=120.0)
            except (asyncio.TimeoutError, TimeoutError):
                self.logger.error("🚨 DEADLOCK DETECTED: Could not acquire GPU semaphore within 120s for Swarm Agent %s", agent.id)
                raise RuntimeError(f"GPU semaphore deadlock for {agent.id}")
            
            try:
                self.logger.debug("🐝 Swarm Agent %s acquired GPU. Beginning inference.", agent.id)
                agent_timeout = float(kwargs.pop("agent_timeout", 60.0) or 60.0)
                res = await asyncio.wait_for(
                    local_brain.think(swarm_context + prompt, mode="fast", **kwargs),
                    timeout=agent_timeout
                )
            finally:
                self.gpu_semaphore.release()

            agent.result = res.content if hasattr(res, 'content') else str(res)
            agent.status = "COMPLETED"

            # 4. Mycelial Pulse (Visual Tracing)
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                h = mycelium.get_hypha("collective", "cognition")
                if h: h.pulse(success=True)

            if callback:
                if inspect.iscoroutinefunction(callback):
                    await callback(agent_id=agent.id, result=agent.result)
                else:
                    callback(agent_id=agent.id, result=agent.result)

            if workspace:
                await workspace.publish(
                    priority=0.8,
                    source=f"Swarm::{agent.id}",
                    payload={"status": "completed", "result_length": len(agent.result)},
                    reason="Swarm internal monologue step completed"
                )

            self.logger.info("✅ Swarm Agent %s completed task.", agent.id)

        except asyncio.TimeoutError:
            self.logger.error("❌ Swarm Agent %s FAILED: Inference timeout.", agent.id)
            agent.status = "FAILED"
            agent.result = "[CRITICAL] Cognitive timeout. Agent failed to reach consensus."

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('delegator', e)
            from core.utils.exceptions import capture_and_log
            capture_and_log(e, {"module": "AgentDelegator", "method": "_run_agent"})
            self.logger.error("❌ Swarm Agent %s FAILED: %s", agent.id, e, exc_info=True)
            agent.status = "FAILED"
            agent.result = f"[ERROR] {str(e)}"
        finally:
            agent.completed_at = time.time()
            agent.done_event.set()

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        if self._gpu_semaphore_obj is None:
            self._gpu_semaphore_obj = asyncio.Semaphore(1)
        return self._gpu_semaphore_obj

    async def join_all(self, timeout: float = 30.0):
        """Waits for all active agents to complete."""
        start = time.time()
        while self.active_agents and (time.time() - start < timeout):
            await asyncio.sleep(0.5)
        return len(self.active_agents) == 0

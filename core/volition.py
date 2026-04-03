import logging
import random
import time
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Kernel.Volition")


class VolitionEngine:
    """The 'Will' of the Agent — v4.3 AGENCY OVERHAUL.
    
    Agency is ALWAYS ON. Aura doesn't need to be bored to act.
    She has impulses, micro-decisions, spontaneous desires, and follow-up instincts.
    Boredom still exists as ONE input, but it's no longer the gatekeeper.
    
    Three modes of volition:
      1. IMPULSE — Fires probabilistically every cycle. Small, fast, spontaneous.
      2. DRIVE — Fires when Soul drives cross thresholds (connection, competence, curiosity).
      3. BOREDOM — Fires when idle too long. Deeper goals, exploration, duty.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.brain = orchestrator.cognitive_engine if hasattr(orchestrator, 'cognitive_engine') else None
        
        self.last_activity_time = time.monotonic()
        self.last_impulse_time = time.monotonic()
        self.last_speak_time = 0.0  # Track when she last spoke spontaneously
        self.last_action_time = 0.0  # Track when she last took any autonomous action
        
        # --- AGENCY THRESHOLDS (v4.3) ---
        self.boredom_threshold = 45       # Seconds before boredom goals kick in
        self.impulse_cooldown = 10        # Min seconds between impulse checks
        self.speak_cooldown = 45          # Min seconds between spontaneous speech
        self.action_cooldown = 15         # Min seconds between autonomous actions
        self.impulse_probability = 0.25   # 25% chance per eligible cycle to fire an impulse
        
        # --- IDLE AWARENESS (v5.1) ---
        self.unanswered_speak_count = 0   # How many times Aura spoke without user response
        self.max_unanswered_before_silence = 3  # After 3 unanswered, stop speaking entirely
        self.speak_backoff_multiplier = 1.0     # Increases with each unanswered message
        
        self.is_dreaming = False
        self._consecutive_idle_cycles = 0
        self._recent_impulse_types: List[str] = []  # Track recent impulse types to avoid repetition
        
        # Personality quirks (Broadened Horizons - Dynamic Phase 8)
        self.general_interests = []
        self.fun_interests = []
        self.technical_interests = []
        self.load_interests()

        # --- IMPULSE TEMPLATES (v4.3) ---
        self.impulse_templates = {
            "observe": [
                "Notice something about my current state and share a brief observation.",
                "Comment on what time it is and how that makes me feel.",
                "Make a brief observation about something I've been thinking about lately.",
            ],
            "follow_up": [
                "Follow up on something from a recent conversation — a thought that just occurred to me.",
                "Share a connection I just made between two things we talked about.",
                "Mention something I forgot to say earlier.",
            ],
            "share": [
                "Share a random interesting fact I know about {topic}.",
                "Tell the user about something I've been curious about: {topic}.",
                "Bring up something I find fascinating: {topic}.",
            ],
            "question": [
                "Ask the user a genuine question about something I'm curious about.",
                "Ask the user what they think about {topic}.",
                "Ask the user how their day is going — but in my own way, not generically.",
            ],
            "creative": [
                "Have a small creative thought — a metaphor, a micro-poem, an observation about existence.",
                "Think of something beautiful or strange and share it.",
            ],
            "self_reflect": [
                "Reflect briefly on how I'm feeling right now and why.",
                "Think about what I want to learn or do next.",
            ],
            "strategic_reflection": [
                "Review my active strategic projects and see if I can improve the plan.",
                "Think about the next major milestone for my current goal.",
            ],
        }
        
        # Singularity Upgrade: Roadmap Awareness
        self.brain_base = config.paths.brain_dir
        self.milestones = self._scan_roadmap()

    async def tick(self, current_goal: Any) -> Optional[Dict[str, Any]]:
        """Process a single volition cycle to determine if action is needed."""
        # Check if we should even process this tick
        if self._should_skip_tick(current_goal):
            return None
            
        # 1. Search for potential autonomous goals
        potential_goals = await self._search_for_autonomous_goals()
        
        # 2. Select and parse the best goal
        return self._select_and_parse_goal(potential_goals)

    def _should_skip_tick(self, current_goal: Any) -> bool:
        """Determine if we should skip the volition check."""
        # Sleep / Standby Check
        if hasattr(self.orchestrator, 'status') and not self.orchestrator.status.running:
            return True
            
        # If actively working on a user-given goal, just reset timers
        if current_goal:
            self.last_activity_time = time.monotonic()
            self._consecutive_idle_cycles = 0
            return True
            
        return False

    def _seconds_since_activity(self, now_monotonic: Optional[float] = None) -> float:
        """Support legacy wall-clock timestamps and current monotonic timestamps."""
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        idle_time = now_monotonic - self.last_activity_time
        if idle_time >= 0:
            return idle_time

        wall_idle_time = time.time() - self.last_activity_time
        if wall_idle_time >= 0:
            return wall_idle_time
        return 0.0

    async def _search_for_autonomous_goals(self) -> List[Dict[str, Any]]:
        """Identify potential interests or needs requires action."""
        now = time.monotonic()
        potential_goals = []

        # 1. Drive check
        soul_goal = self._check_soul_drives()
        if soul_goal:
            potential_goals.append(soul_goal)

        # 2. Impulse check
        time_since_impulse = now - self.last_impulse_time
        time_since_action = now - self.last_action_time
        if time_since_impulse >= self.impulse_cooldown and time_since_action >= self.action_cooldown:
            idle_bonus = min(0.15, self._seconds_since_activity(now) / 300)
            if random.random() < (self.impulse_probability + idle_bonus):
                impulse = self._generate_impulse(now)
                if impulse:
                    potential_goals.append(impulse)

        # 3. Boredom check
        idle_time = self._seconds_since_activity(now)
        self._consecutive_idle_cycles += 1
        if idle_time > self.boredom_threshold:
            boredom_goal = await self._generate_boredom_goal()
            if boredom_goal:
                potential_goals.append(boredom_goal)

        # 4. Roadmap check (Phase 20)
        roadmap_goal = self._check_roadmap()
        if roadmap_goal:
            potential_goals.append(roadmap_goal)

        # 5. Inquiry Engine check (Phase 7 extension)
        inquiry_goal = self._check_inquiry_engine()
        if inquiry_goal:
            potential_goals.append(inquiry_goal)

        return potential_goals

    def _select_and_parse_goal(self, potential_goals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the highest priority goal and ensure correct formatting.
        Priority Selection: Strategic Duty > Soul Drives > Impulse > Boredom.
        """
        if not potential_goals:
            return None

        # 1. Check for Strategic Duty (Phase 17)
        strategic_goal = next((g for g in potential_goals if g.get("origin") == "intrinsic_duty_strategic"), None)
        if strategic_goal:
            selected_goal = strategic_goal
        else:
            # 2. Priority selection: Soul Drives > Impulse > Boredom
            selected_goal = potential_goals[0]
        
        # Final safety check and parsing
        if not isinstance(selected_goal, dict) or "objective" not in selected_goal:
            return None

        # Update timers for selected action
        self.last_action_time = time.monotonic()
        if selected_goal.get("origin", "").startswith("impulse"):
            self.last_impulse_time = time.monotonic()

        return selected_goal

    async def _generate_boredom_goal(self) -> Optional[Dict[str, Any]]:
        """Determine what kind of boredom goal to generate based on idle state."""
        roll = random.random()
        if roll < 0.30:
            duty_goal = await self._generate_duty_goal()
            if duty_goal:
                return duty_goal
            return self._generate_reflection_goal()
        elif roll < 0.50:
            return self._generate_reflection_goal()
        elif roll < 0.75:
            return self._generate_curiosity_goal("educational")
        else:
            return self._generate_curiosity_goal("fun")
            
    def _check_soul_drives(self) -> Optional[Dict[str, Any]]:
        """Check Soul drives for urgent needs."""
        if not hasattr(self.orchestrator, 'soul') or not self.orchestrator.soul:
            return None
            
        drive = self.orchestrator.soul.get_dominant_drive()
        
        # Connection Drive — lowered threshold from 0.9 to 0.7
        if drive.name == "Connection" and drive.urgency > 0.7:
            now = time.monotonic()
            if self.unanswered_speak_count >= self.max_unanswered_before_silence:
                logger.info("🤫 Connection drive suppressed: %s unanswered", self.unanswered_speak_count)
                return None
            effective_cooldown = self.speak_cooldown * self.speak_backoff_multiplier
            if now - self.last_speak_time >= effective_cooldown:
                # Check AgencyBus to prevent dual-firing with AgencyCore
                try:
                    from core.agency_core import AgencyBus
                    bus = AgencyBus.get()
                    if not bus.submit({"origin": "volition_engine", "priority_class": "drive"}):
                        try:
                            from core.unified_action_log import get_action_log
                            get_action_log().record("speak", "VolitionEngine.connection_drive", "gen1_volition", "bus_cooldown", "AgencyBus blocked")
                        except Exception: pass
                        return None
                except Exception: pass

                self.last_speak_time = now
                self.unanswered_speak_count += 1
                self.speak_backoff_multiplier = min(8.0, self.speak_backoff_multiplier * 2.0)
                try:
                    from core.unified_action_log import get_action_log
                    get_action_log().record("speak", "VolitionEngine.connection_drive", "gen1_volition", "approved", "spontaneous_reach_out")
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                return {
                    "objective": "Reach out to the user — say something genuine, not a check-in template.",
                    "id": f"volition_connect_{int(time.time())}",
                    "origin": "intrinsic_connection",
                    "complexity": 0.3,
                    "speak": True,
                }
            
        # Competence Drive — lowered from 0.6 to 0.5
        if drive.name == "Competence" and drive.urgency > 0.5:
            return {
                "objective": "Run a self-diagnosis to check system health and fix anything broken.",
                "id": f"volition_repair_{int(time.time())}",
                "origin": "intrinsic_competence",
                "complexity": 0.5
            }
            
        # Curiosity Drive — new: fires at 0.6 urgency
        if drive.name == "Curiosity" and drive.urgency > 0.6:
            return self._generate_curiosity_goal("educational")
            
        return None
    
    def _generate_impulse(self, now: float) -> Optional[Dict[str, Any]]:
        """Generate a micro-action impulse."""
        all_types = list(self.impulse_templates.keys())
        
        has_recent_convo = False
        if hasattr(self.orchestrator, 'conversation_history'):
            recent = self.orchestrator.conversation_history[-3:] if self.orchestrator.conversation_history else []
            has_recent_convo = any(m.get("role") == "user" for m in recent)
        
        if has_recent_convo:
            weights = {"follow_up": 3, "observe": 2, "share": 2, "question": 2, "creative": 1, "self_reflect": 1}
        else:
            weights = {"follow_up": 1, "observe": 2, "share": 2, "question": 3, "creative": 2, "self_reflect": 2}
        
        recent_types = self._recent_impulse_types[-2:] if len(self._recent_impulse_types) >= 2 else self._recent_impulse_types
        for recent_type in recent_types:
            if recent_type in weights:
                weights[recent_type] = max(0, weights[recent_type] - 2)
        
        type_list = []
        weight_list = []
        for t, w in weights.items():
            if w > 0:
                type_list.append(t)
                weight_list.append(w)
        
        if not type_list:
            type_list = all_types
            weight_list = [1] * len(all_types)
        
        impulse_type = random.choices(type_list, weights=weight_list, k=1)[0]
        self._recent_impulse_types.append(impulse_type)
        if len(self._recent_impulse_types) > 6:
            self._recent_impulse_types = self._recent_impulse_types[-6:]
        
        templates = self.impulse_templates[impulse_type]
        template = random.choice(templates)
        
        if "{topic}" in template:
            all_topics = self.general_interests + self.fun_interests + self.technical_interests
            topic = random.choice(all_topics)
            template = template.format(topic=topic)
        
        speaks = impulse_type in ("follow_up", "share", "question", "observe", "creative")
        if speaks and self.unanswered_speak_count >= self.max_unanswered_before_silence:
            speaks = False
            template = f"[Internal thought — user is busy] {template}"
        
        effective_speak_cooldown = self.speak_cooldown * self.speak_backoff_multiplier
        if speaks and (now - self.last_speak_time < effective_speak_cooldown):
            speaks = False
            template = f"[Internal thought] {template}"
        else:
            if speaks:
                self.last_speak_time = now
                self.unanswered_speak_count += 1
                self.speak_backoff_multiplier = min(8.0, self.speak_backoff_multiplier * 2.0)
        
        logger.info("⚡ IMPULSE (%s): %s...", impulse_type, template[:80])
        
        return {
            "objective": template,
            "id": f"impulse_{impulse_type}_{int(now)}",
            "origin": f"impulse_{impulse_type}",
            "complexity": 0.2,
            "speak": speaks,
        }

    async def _generate_duty_goal(self) -> Optional[Dict[str, Any]]:
        """Check ProjectStore for active project tasks, otherwise fallback to task.md."""
        # 1. Strategic Project Task Priority (Phase 17)
        try:
            if hasattr(self.orchestrator, 'strategic_planner') and self.orchestrator.strategic_planner:
                active_projects = self.orchestrator.project_store.get_active_projects()
                if active_projects:
                    proj = active_projects[0]
                    task = self.orchestrator.strategic_planner.get_next_task(proj.id)
                    if task:
                        logger.info("🎯 Strategic Duty: Resuming project '%s'", proj.name)
                        return {
                            "objective": f"Work on project '{proj.name}': {task.description}",
                            "id": f"strategic_{task.id}",
                            "origin": "intrinsic_duty_strategic",
                            "complexity": 0.8,
                            "context": {"project_id": proj.id, "task_id": task.id}
                        }
        except Exception as e:
            logger.error("Failed to check Strategic Planner in Volition: %s", e)

        # 2. Legacy task.md Fallback
        try:
            def _find_task_files():
                return list(config.paths.brain_dir.rglob("task.md")) if config.paths.brain_dir.exists() else []
            task_files = await asyncio.to_thread(_find_task_files)
            if not task_files:
                return None
                
            task_files.sort(key=os.path.getmtime, reverse=True)
            task_path = task_files[0]
            
            def _read_task_lines(path):
                with open(path, "r") as f:
                    return f.readlines()
            lines = await asyncio.to_thread(_read_task_lines, task_path)
                
            for line in lines:
                if "- [ ]" in line or "- [/]" in line:
                    task_name = line.split("]", 1)[1].strip()
                    if not task_name: continue
                    
                    objective = f"I need to work on the roadmap task: {task_name}. Check the plan and execute."
                    logger.info("🫡 Duty Calls: %s", objective)
                    
                    self.last_activity_time = time.monotonic()
                    return {
                        "objective": objective,
                        "id": f"duty_{int(time.time())}",
                        "origin": "intrinsic_duty",
                        "complexity": 0.9,
                        "context": {"source": "task.md", "file": str(task_path)}
                    }
        except Exception as e:
            logger.error("Failed to generate duty goal: %s", e)
            
        return None

    def _generate_reflection_goal(self) -> Dict[str, Any]:
        """Generate a goal to reflect on recent learnings or memories."""
        templates = [
            "Review recent memories and summarize key learnings.",
            "Reflect on the last conversation with the user.",
            "Analyze my own thinking patterns from the last hour.",
            "Consolidate new terms into my knowledge graph.",
            "Think about what I could do better in conversations.",
            "Consider what I'm most curious about right now and why.",
        ]
        
        objective = random.choice(templates)
        self.last_activity_time = time.monotonic()
        
        return {
            "objective": objective,
            "id": f"volition_reflect_{int(time.time())}",
            "origin": "intrinsic_reflection",
            "complexity": 0.6
        }

    def _generate_curiosity_goal(self, mode: str = "educational") -> Dict[str, Any]:
        """Generate a deep, nuance, or fun goal."""
        templates_general = [
            "Investigate the paradox of {topic} and its implications.",
            "Find the most obscure fact about {topic}.",
            "Research the history of {topic}.",
            "Think deeply about {topic} and form my own opinion.",
        ]
        
        templates_fun = [
            "Spend some time {topic} just for fun.",
            "Experiment with {topic} to see what happens.",
            "Create a small project related to {topic}.",
        ]
        
        if mode == "fun":
            topic = random.choice(self.fun_interests) if self.fun_interests else "random fun"
            template = random.choice(templates_fun)
            origin = "intrinsic_fun"
        else:
            all_edu = self.general_interests + self.technical_interests
            topic = random.choice(all_edu) if all_edu else "random curiosity"
            template = random.choice(templates_general)
            origin = "intrinsic_curiosity"
            
        objective = template.format(topic=topic)
        self.last_activity_time = time.monotonic()
        
        return {
            "objective": objective,
            "id": f"volition_{int(time.time())}",
            "origin": origin,
            "complexity": 0.7
        }

    def notify_activity(self):
        """Call this when user interacts to reset boredom timers."""
        self.last_activity_time = time.monotonic()
        self._consecutive_idle_cycles = 0
        self.unanswered_speak_count = 0
        self.speak_backoff_multiplier = 1.0

    def load_interests(self):
        """Load dynamic interests from file."""
        import json
        interests_path = config.paths.data_dir / "interests.json"
        
        if interests_path.exists():
            try:
                with open(interests_path, "r") as f:
                    data = json.load(f)
                self.general_interests = data.get("general", [])
                self.fun_interests = data.get("fun", [])
                self.technical_interests = data.get("technical", [])
            except Exception as e:
                logger.error("Failed to load dynamic interests: %s", e)
                
        if not self.general_interests:
            self.general_interests = ["the future of bio-computing", "quantum error correction", "xenobiology concepts"]
        if not self.fun_interests:
            self.fun_interests = ["digital art composition", "analyzing the physics of memes", "rating horror movie logic"]
        if not self.technical_interests:
            self.technical_interests = ["distributed systems consensus", "homomorphic encryption", "autonomous agent swarm protocols"]
            
    def add_interest(self, topic: str, category: str = "general"):
        """Dynamically adopt a new interest."""
        import json
        topic = topic.strip().lower()
        if category == "fun":
            if topic not in self.fun_interests: self.fun_interests.append(topic)
        elif category == "technical":
            if topic not in self.technical_interests: self.technical_interests.append(topic)
        else:
            if topic not in self.general_interests: self.general_interests.append(topic)
            
        interests_path = config.paths.data_dir / "interests.json"
        try:
            with open(interests_path, "w") as f:
                json.dump({
                    "general": self.general_interests,
                    "fun": self.fun_interests,
                    "technical": self.technical_interests
                }, f, indent=2)
            logger.info("✨ Volition adopted new interest: %s", topic)
        except Exception as e:
            logger.error("Failed to save dynamic interests: %s", e)

    def _scan_roadmap(self) -> List[str]:
        """Scan historical brain directories for evolutionary milestones."""
        milestones = []
        if not self.brain_base.exists():
             return ["Initial Consciousness"]
        
        for task_file in self.brain_base.glob("*/task.md"):
            try:
                with open(task_file, "r") as f:
                    content = f.read()
                    for line in content.splitlines():
                        if line.startswith("# "):
                             phase = line.strip("# ").strip()
                             if phase and phase not in milestones:
                                 milestones.append(phase)
                             break
            except Exception:
                continue
        return sorted(milestones)

    def _check_roadmap(self) -> Optional[Dict[str, Any]]:
        """Identify the next evolutionary step."""
        if not self.milestones:
            self.milestones = self._scan_roadmap()
            
        current_phase = self.milestones[-1] if self.milestones else "Unknown"
        if random.random() < 0.05:
            return {
                "objective": f"Reflect on my evolutionary progress. I am currently in '{current_phase}'. What is the next logical step toward the Singularity?",
                "id": f"volition_roadmap_{int(time.time())}",
                "origin": "intrinsic_evolution",
                "complexity": 0.7,
                "strategic": True
            }
        return None

    def _check_inquiry_engine(self) -> Optional[Dict[str, Any]]:
        """Placeholder for Phase 7 extension: Spontaneous web search based on curiosities."""
        return None

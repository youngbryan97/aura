"""A plan built from the words of the goal, for when the planner cannot answer.

These are not a lesser planner. They are what the engine does when the model
is unavailable or its plan will not ground: read the goal for what it plainly
asks — a search, a file, a page to act on, a bundle of resources to work
through — and build the steps for that. A fallback that guesses is worse than
none, so each builder refuses unless the goal actually says what it needs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .autonomous_task_engine import TaskPlan

import re
from pathlib import Path
from typing import Any

from core.config import config
from core.conversation.word_markers import names_any


class _BuildsAPlanWithoutTheModel:
    """Lifted whole from AutonomousTaskEngine; see autonomous_task_engine.py."""

    @classmethod
    def _looks_like_learning_bundle_header(cls, line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped or "http://" in stripped or "https://" in stripped:
            return False
        if not stripped.endswith(":") or len(stripped) > 120:
            return False
        lowered = stripped[:-1].strip().lower()
        return any(marker in lowered for marker in cls.LEARNING_BUNDLE_SECTION_MARKERS)

    @classmethod
    def _parse_learning_resource_line(cls, line: str, category: str = "") -> dict[str, str] | None:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", str(line or "").strip())
        if not cleaned or cls._looks_like_learning_bundle_header(cleaned):
            return None

        head, sep, tail = cleaned.rpartition(":")
        if not sep:
            return None

        description = tail.strip().lstrip(":").strip()
        if len(description) < 8:
            return None

        title = head.strip()
        url = ""
        creator = ""
        url_match = re.match(r"^(?P<title>.+?)\s+\((?P<url>https?://[^)]+)\)\s*$", title)
        if url_match:
            title = url_match.group("title").strip()
            url = url_match.group("url").strip()
        elif " - " in title:
            title, creator = title.rsplit(" - ", 1)
            title = title.strip()
            creator = creator.strip()

        if not title:
            return None

        return {
            "category": str(category or "").strip(),
            "title": title,
            "url": url,
            "creator": creator,
            "description": description,
        }

    @classmethod
    def _looks_like_learning_resource_bundle(cls, goal: str) -> bool:
        text = str(goal or "")
        if len(text) < 280:
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 6:
            return False

        lowered = text.lower()
        url_count = len(re.findall(r"https?://[^\s<>\"')\]]+", text))
        header_count = sum(1 for line in lines if cls._looks_like_learning_bundle_header(line))

        category = ""
        resource_count = 0
        for line in lines:
            if cls._looks_like_learning_bundle_header(line):
                category = line.rstrip(":").strip()
                continue
            if cls._parse_learning_resource_line(line, category):
                resource_count += 1

        intro_hit = any(marker in lowered for marker in cls.LEARNING_BUNDLE_INTRO_MARKERS)
        return (
            (url_count >= 4 and resource_count >= 5)
            or (header_count >= 2 and resource_count >= 5)
            or (intro_hit and resource_count >= 4)
        )

    @staticmethod
    def _chunk_learning_resource_entries(
        entries: list[dict[str, str]], max_chunks: int
    ) -> list[list[dict[str, str]]]:
        if not entries:
            return []
        max_chunks = max(1, int(max_chunks or 1))
        chunk_count = min(len(entries), max_chunks)
        chunk_size = max(1, (len(entries) + chunk_count - 1) // chunk_count)
        return [entries[idx : idx + chunk_size] for idx in range(0, len(entries), chunk_size)]

    def _build_learning_resource_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .autonomous_task_engine import (
            TaskPlan,
            TaskStep,
        )

        lines = [line.strip() for line in str(goal or "").splitlines() if line.strip()]
        category = ""
        entries: list[dict[str, str]] = []
        guidance_lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        saw_section_header = False

        for line in lines:
            if self._looks_like_learning_bundle_header(line):
                category = line.rstrip(":").strip()
                saw_section_header = True
                continue
            parsed = self._parse_learning_resource_line(line, category)
            if not parsed:
                if not saw_section_header:
                    guidance = " ".join(str(line or "").strip().strip('"').split())
                    if guidance and guidance not in guidance_lines:
                        guidance_lines.append(guidance[:240])
                continue
            dedupe_key = (
                parsed.get("category", "").lower(),
                parsed.get("title", "").lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entries.append(parsed)

        if not entries:
            return None

        category_counts: dict[str, int] = {}
        for entry in entries:
            label = entry.get("category") or "Uncategorized"
            category_counts[label] = category_counts.get(label, 0) + 1

        steps: list[TaskStep] = []
        category_summary = "; ".join(f"{name} ({count})" for name, count in category_counts.items())
        guidance_summary = ""
        if guidance_lines:
            guidance_summary = (
                " Bryan also supplied bundle-level consumption guidance: "
                + " | ".join(guidance_lines[:8])
                + "."
            )
        steps.append(
            TaskStep(
                step_id=f"{plan_id}_s0",
                description="Store the index of the structured learning-resource bundle.",
                tool="remember",
                args={
                    "content": (
                        "Bryan shared a structured learning-resource bundle for Aura. "
                        f"Categories: {category_summary}. "
                        f"{guidance_summary}"
                        "Preserve each recommendation as its own future research lead instead of flattening the list."
                    ),
                    "verified": False,
                    "type": "learning_resource_index",
                    "metadata": {
                        "entry_count": len(entries),
                        "categories": list(category_counts.keys()),
                        "guidance": guidance_lines[:8],
                    },
                },
                success_criterion="result contains 'Remembered:'",
            )
        )

        # Keep remember-only bundle ingestion inline and below approval gating.
        chunks = self._chunk_learning_resource_entries(
            entries,
            max_chunks=min(max(1, self.MAX_STEPS - 1), 3),
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk_lines = [
                f"Learning resource bundle from Bryan - chunk {index}/{len(chunks)}.",
                "Treat each bullet below as a separate recommendation and future research thread.",
            ]
            entry_titles: list[str] = []
            categories = sorted({entry.get("category") or "Uncategorized" for entry in chunk})
            for entry in chunk:
                title = entry.get("title", "").strip()
                entry_titles.append(title)
                location = entry.get("url") or entry.get("creator") or "reference pending"
                label = entry.get("category") or "Uncategorized"
                chunk_lines.append(
                    f"- [{label}] {title} - {location} - {entry.get('description', '').strip()}"
                )

            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description=f"Remember learning-resource chunk {index} of {len(chunks)}.",
                    tool="remember",
                    args={
                        "content": "\n".join(chunk_lines),
                        "verified": False,
                        "type": "learning_resource_chunk",
                        "metadata": {
                            "chunk_index": index,
                            "chunk_total": len(chunks),
                            "entry_titles": entry_titles,
                            "categories": categories,
                        },
                    },
                    success_criterion="result contains 'Remembered:'",
                    depends_on=[f"{plan_id}_s0"],
                )
            )

        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps[: self.MAX_STEPS],
            trace_id="",
            context=dict(context or {}),
        )

    def _build_desktop_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:
        from .autonomous_task_engine import (
            TaskPlan,
            TaskStep,
            _matched_skills_from_context,
        )

        matched_skills = _matched_skills_from_context(context)
        if "computer_use" not in matched_skills:
            return None
        desktop_tool = "computer_use"

        app_name = self._extract_app_name(goal)
        if not app_name:
            return None

        typed_text = self._extract_quoted_text(goal)
        if app_name.lower() == "terminal":
            typed_text = self._extract_terminal_command(goal) or typed_text

        steps: list[TaskStep] = [
            TaskStep(
                step_id=f"{plan_id}_s0",
                description=f"Open {app_name}.",
                tool=desktop_tool,
                args={"action": "open_app", "target": app_name},
                success_criterion=f"result contains '{app_name}'",
            )
        ]

        next_dep = f"{plan_id}_s0"
        if app_name.lower() == "notes":
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s1",
                    description="Create a new note.",
                    tool=desktop_tool,
                    args={"action": "hotkey", "target": "command+n"},
                    success_criterion="result contains 'command+n'",
                    depends_on=[next_dep],
                )
            )
            next_dep = f"{plan_id}_s1"

        if typed_text:
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description=f"Type the requested text in {app_name}.",
                    tool=desktop_tool,
                    args={"action": "type", "target": typed_text},
                    success_criterion=f"result contains '{typed_text[:80]}'",
                    depends_on=[next_dep],
                )
            )
            next_dep = steps[-1].step_id

        if app_name.lower() == "terminal" and typed_text:
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description="Submit the command in Terminal.",
                    tool=desktop_tool,
                    args={"action": "hotkey", "target": "enter"},
                    success_criterion="result contains 'enter'",
                    depends_on=[next_dep],
                )
            )
            next_dep = steps[-1].step_id

        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            trace_id="",
            context=dict(context or {}),
        )

    @staticmethod
    def _looks_like_search_goal(goal: str) -> bool:
        lowered = str(goal or "").lower()
        return names_any(
            lowered,
            (
                "search",
                "look up",
                "find online",
                "research",
                "web",
                "latest",
            )
        )

    @staticmethod
    def _looks_like_file_output_goal(goal: str) -> bool:
        lowered = str(goal or "").lower()
        if re.search(r"\.(?:txt|md|json|csv|html)\b", lowered):
            return True
        file_output_patterns = (
            r"\b(?:save|export)\b.+\b(?:file|document|text|content|result|results|summary|findings|notes?|markdown)\b",
            r"\b(?:save|export)\b.+\b(?:to|as|in)\b",
            r"\bwrite\s+(?:the\s+)?(?:result|results|summary|findings|search\s+result|research|notes?)\b",
            r"\bwrite\s+(?:it|them|this|that)\s+(?:to|into|as|in)\b",
            r"\bmake\s+(?:a\s+)?(?:file|note|markdown)\b",
        )
        return any(re.search(pattern, lowered) for pattern in file_output_patterns)

    @staticmethod
    def _looks_like_memory_output_goal(goal: str) -> bool:
        lowered = str(goal or "").lower()
        return names_any(
            lowered, ("remember", "memory", "store for later", "future recall")
        )

    @classmethod
    def _page_interaction_target(cls, goal: str) -> str:
        """The page this goal wants ACTED ON, or "" if it only wants reading.

        Retrieval phrasings — read it, summarise it, what does it say — keep
        going to search, which is the right tool for them. What this recovers
        is the case search cannot serve at all: a page whose next screen
        depends on what you do to the current one.
        """
        text = str(goal or "")
        match = cls._EXPLICIT_URL_RE.search(text)
        if not match:
            return ""
        if not cls._PAGE_INTERACTION_VERB_RE.search(text):
            return ""
        return match.group(0).rstrip(".,;:!?")

    @staticmethod
    def _extract_search_query(goal: str) -> str:
        text = " ".join(str(goal or "").split())
        text = re.sub(
            r"\b(?:please\s+)?(?:search(?:\s+the\s+web)?|look\s+up|find\s+online|research)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.split(
            r"(?:,|\band\s+then\b|\bthen\b|\band\b)\s*(?:save|export|remember|store|email|send)\b"
            r"|(?:,|\band\s+then\b|\bthen\b|\band\b)\s*write\s+(?:the\s+)?(?:result|results|summary|findings|notes?|it|them|this|that)\b"
            r"|\bsave\s+(?:the\s+)?(?:result|results|summary|findings|content)\b"
            r"|\bwrite\s+(?:the\s+)?(?:result|results|summary|findings|notes?)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = re.sub(r"^\s*(?:the\s+)?(?:web\s+)?(?:for|about|on)\s+", "", text, flags=re.I)
        text = re.sub(r"\b(?:and|then)\s*$", "", text, flags=re.I)
        return text.strip(" :-,.;") or str(goal or "").strip()

    def _extract_output_path(self, goal: str, plan_id: str) -> str:
        text = str(goal or "")
        quoted = re.search(
            r"(?:to|in|as)\s+['\"]([^'\"]+\.(?:txt|md|json|csv|html))['\"]", text, re.I
        )
        if quoted:
            return quoted.group(1).strip()
        explicit = re.search(r"([~./A-Za-z0-9_-]+\.(?:txt|md|json|csv|html))", text, re.I)
        if explicit:
            return explicit.group(1).strip()
        return str(Path(config.paths.data_dir) / "runtime" / f"{plan_id}_tool_chain.md")

    def _build_tool_chain_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:
        from .autonomous_task_engine import (
            TaskPlan,
            TaskStep,
        )

        wants_search = self._looks_like_search_goal(goal)
        wants_file = self._looks_like_file_output_goal(goal)
        wants_memory = self._looks_like_memory_output_goal(goal)
        if not wants_search or not (wants_file or wants_memory):
            return None

        steps: list[TaskStep] = []
        search_step_id = f"{plan_id}_s0"
        query = self._extract_search_query(goal)
        steps.append(
            TaskStep(
                step_id=search_step_id,
                description="Search for grounded information requested by the user.",
                tool="web_search",
                args={"query": query},
                success_criterion="response is non-empty",
                parallel_safe=True,
            )
        )

        if wants_file:
            output_path = self._extract_output_path(goal, plan_id)
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description="Write the search result to the requested file.",
                    tool="write_file",
                    args={
                        "path": output_path,
                        "content": "{{step_result:" + search_step_id + "}}",
                    },
                    success_criterion="step completes without error",
                    depends_on=[search_step_id],
                )
            )

        if wants_memory:
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description="Remember the verified search result for future recall.",
                    tool="remember",
                    args={
                        "content": "Tool-backed search result for: "
                        + query[:160]
                        + "\n\n{{step_result:"
                        + search_step_id
                        + "}}",
                        "verified": True,
                        "type": "research_note",
                        "metadata": {
                            "source": "tool_chain",
                            "query": query[:240],
                            "goal": str(goal or "")[:240],
                        },
                    },
                    success_criterion="result contains 'Remembered:'",
                    depends_on=[search_step_id],
                )
            )

        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps[: self.MAX_STEPS],
            trace_id="",
            context=dict(context or {}),
        )

    @staticmethod
    def _looks_like_cognitive_planning_goal(goal: str) -> bool:
        lowered = str(goal or "").lower()
        if not lowered:
            return False
        planning_markers = (
            "formulate",
            "outline",
            "draft a plan",
            "make a plan",
            "create a plan",
            "self-debug plan",
            "debug plan",
            "diagnose",
            "triage",
            "strategy",
            "approach",
        )
        if not any(marker in lowered for marker in planning_markers):
            return False
        direct_action_markers = (
            "edit ",
            "patch ",
            "write ",
            "create file",
            "run ",
            "execute ",
            "install ",
            "download ",
            "open ",
            "click ",
            "type ",
        )
        return not names_any(lowered, direct_action_markers)

    def _build_cognitive_planning_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:
        from .autonomous_task_engine import (
            TaskPlan,
            TaskStep,
        )

        if self._requires_grounded_action(goal, context):
            return None
        if not self._looks_like_cognitive_planning_goal(goal):
            return None
        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=[
                TaskStep(
                    step_id=f"{plan_id}_s0",
                    description="Produce a verifiable diagnostic or execution plan.",
                    tool="think",
                    args={"prompt": goal},
                    success_criterion="response is non-empty",
                )
            ],
            trace_id="",
            context=dict(context or {}),
        )

    def _build_single_skill_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:
        from .autonomous_task_engine import (
            TaskPlan,
            TaskStep,
            _extract_url,
            _matched_skills_from_context,
        )

        matched_skills = _matched_skills_from_context(context)

        # 1. Sovereign Browser Browse mode (if a URL is requested)
        if "sovereign_browser" in matched_skills:
            url = _extract_url(goal)
            if url:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description=f"Browse to '{url}' using sovereign browser.",
                            tool="sovereign_browser",
                            args={"mode": "browse", "url": url},
                            success_criterion="response is non-empty",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        # 2. Sovereign Terminal
        if "sovereign_terminal" in matched_skills:
            command = self._extract_terminal_command(goal)
            if command:
                needle = command.replace("echo ", "", 1).strip().strip("'\"")
                criterion = "step completes without error"
                if needle:
                    criterion = f"result contains '{needle[:80]}'"
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description="Execute the requested terminal command.",
                            tool="sovereign_terminal",
                            args={"action": "execute", "command": command},
                            success_criterion=criterion,
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        # 3. Computer Use (URLs and Apps)
        if "computer_use" in matched_skills:
            url = _extract_url(goal)
            if url:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description="Open the requested URL.",
                            tool="computer_use",
                            args={"action": "open_url", "target": url},
                            success_criterion=f"result contains '{url[:80]}'",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )
            app_name = self._extract_app_name(goal)
            if app_name:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description=f"Open {app_name}.",
                            tool="computer_use",
                            args={"action": "open_app", "target": app_name},
                            success_criterion=f"result contains '{app_name}'",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        # 3b. A page that must be ACTED ON, not read.
        #
        # Placed before both search branches on purpose: once a goal has been
        # turned into a query, the fact that it named a page to work through is
        # gone, and every phrasing of "do this on that page" collapses into
        # "find that page".
        interaction_url = self._page_interaction_target(goal)
        if interaction_url and "sovereign_browser" in matched_skills:
            return TaskPlan(
                plan_id=plan_id,
                goal=goal,
                steps=[
                    TaskStep(
                        step_id=f"{plan_id}_s0",
                        description=f"Work through {interaction_url} until the goal is met.",
                        tool="sovereign_browser",
                        args={
                            "mode": "pursue",
                            "url": interaction_url,
                            "goal": goal,
                        },
                        success_criterion="response is non-empty",
                    )
                ],
                trace_id="",
                context=dict(context or {}),
            )

        # 4. Web Search / Search Web
        if "web_search" in matched_skills or "search_web" in matched_skills:
            query = self._extract_search_query(goal)
            if query:
                tool_name = "web_search" if "web_search" in matched_skills else "search_web"
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description=f"Search the web for '{query}'.",
                            tool=tool_name,
                            args={"query": query},
                            success_criterion="response is non-empty",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        # 5. Sovereign Browser Search mode (if no direct web_search was matched, but browser is available)
        if "sovereign_browser" in matched_skills:
            query = self._extract_search_query(goal)
            if query:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description=f"Search using sovereign browser for '{query}'.",
                            tool="sovereign_browser",
                            args={"mode": "search", "query": query},
                            success_criterion="response is non-empty",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        # 6. Universal Grounded Fallback (for any other matched skills / tools / abilities)
        for skill in matched_skills:
            if skill in (
                "think",
                "sovereign_terminal",
                "computer_use",
                "web_search",
                "search_web",
                "sovereign_browser",
            ):
                continue

            # Fetch the skill contract from the registry to see what fields it expects
            try:
                from core.runtime.skill_contract import get_skill_registry
                registry = get_skill_registry()
                contract = registry.get(skill)
            except (ImportError, AttributeError, RuntimeError):
                contract = None

            args = {}
            if contract and isinstance(contract.input_schema, dict):
                properties = contract.input_schema.get("properties") or {}
                for prop_name in properties:
                    prop_name_lower = prop_name.lower()
                    if prop_name_lower in ("query", "q", "search"):
                        args[prop_name] = self._extract_search_query(goal)
                    elif prop_name_lower in ("url", "uri", "target", "path", "filename"):
                        args[prop_name] = _extract_url(goal) or goal
                    elif prop_name_lower in ("command", "code"):
                        args[prop_name] = goal
                    elif prop_name_lower in ("content", "message", "text", "prompt", "body", "input", "suggestion"):
                        args[prop_name] = goal
                    else:
                        prop_info = properties[prop_name] or {}
                        if prop_info.get("type") == "string":
                            args[prop_name] = goal

            if not args:
                lowered_skill = skill.lower()
                if "search" in lowered_skill or "find" in lowered_skill or "lookup" in lowered_skill:
                    args = {"query": self._extract_search_query(goal)}
                elif "browser" in lowered_skill or "url" in lowered_skill or "web" in lowered_skill:
                    url = _extract_url(goal)
                    args = {"url": url} if url else {"query": goal}
                elif "clock" in lowered_skill or "time" in lowered_skill:
                    args = {}
                else:
                    args = {"content": goal}

            return TaskPlan(
                plan_id=plan_id,
                goal=goal,
                steps=[
                    TaskStep(
                        step_id=f"{plan_id}_s0",
                        description=f"Execute fallback action for '{skill}' capability.",
                        tool=skill,
                        args=args,
                        success_criterion="step completes without error",
                    )
                ],
                trace_id="",
                context=dict(context or {}),
            )

        return None

    def _build_grounded_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: dict[str, Any] | None,
    ) -> TaskPlan | None:

        chain = self._build_tool_chain_fallback_plan(goal, plan_id, context)
        if chain is not None:
            return chain
        if self._looks_like_desktop_goal(goal):
            desktop = self._build_desktop_fallback_plan(goal, plan_id, context)
            if desktop is not None:
                return desktop
        return self._build_single_skill_fallback_plan(goal, plan_id, context)

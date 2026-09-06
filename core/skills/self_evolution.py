import ast
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill


class EvolutionInput(BaseModel):
    action: str = Field("propose", description="Action: 'propose' | 'apply' | 'scramble'")
    objective: str = Field(..., description="The feature or improvement to implement.")
    files: list[str] | None = Field(None, description="Specific files to target.")
    read_only: bool = Field(
        False,
        description=(
            "Inspect and return a proposal without writing proposal or source artifacts."
        ),
    )

class SelfEvolutionSkill(BaseSkill):
    """Skill for autonomous self-improvement. Performs the Research -> Synthesize -> Propose loop for code updates."""
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    
    name = "self_evolution"
    description = "Research and propose code updates for the agent's own codebase."
    input_model = EvolutionInput
    output = "Proposed implementation plan or patch."
    MAX_PREVIEW = 1000  # Max characters to include in proposal preview

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"Skills.{self.name}")
        self.code_base = config.paths.project_root
        self.error_registry = self._load_error_registry()

    @staticmethod
    def _evolution_dir() -> Path:
        path = config.paths.data_dir / "evolution"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_error_registry(self) -> dict:
        """Load or initialize error registry for self-correction."""
        filepath = self._evolution_dir() / "error_registry.json"
        if filepath.exists():
            try:
                with open(filepath) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                self.logger.warning("Could not load error registry: %s", e)
        return {}

    def _save_error_registry(self):
        """Save updated error registry."""
        filepath = self._evolution_dir() / "error_registry.json"
        get_file_write_gateway().write_text(
            filepath,
            json.dumps(self.error_registry),
            source="skills.self_evolution.error_registry",
        )

    def _resolve_brain(self, context: dict[str, Any] | None) -> Any:
        ctx = context or {}
        if "brain" in ctx:
            return ctx.get("brain")

        try:
            from core.container import ServiceContainer

            # Self-evolution may use the running mind, but it does not own the
            # mind's service graph and therefore cannot initialize it ad hoc.
            return ServiceContainer.peek("cognitive_engine", default=None)
        except (ImportError, AttributeError, RuntimeError):
            return None

    def _resolve_target_paths(self, files: list[str] | None) -> list[Path]:
        resolved: list[Path] = []
        for raw_path in files or []:
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (self.code_base / candidate).resolve()
            if candidate.exists() and candidate not in resolved:
                resolved.append(candidate)
        return resolved

    @staticmethod
    def _effective_timeout(context: dict[str, Any] | None, default: float = 15.0) -> float:
        ctx = context or {}
        timeout_raw = (
            ctx.get("timeout_s")
            or (ctx.get("executive_constraints", {}) or {}).get("timeout_s")
            or default
        )
        try:
            return max(5.0, float(timeout_raw))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_read_only(context: dict[str, Any] | None) -> bool:
        ctx = context or {}
        if ctx.get("read_only") is not None:
            return bool(ctx.get("read_only"))
        return bool((ctx.get("executive_constraints", {}) or {}).get("read_only"))

    async def _think_with_timeout(
        self,
        brain: Any,
        prompt: str,
        context: dict[str, Any] | None,
        *,
        default_timeout: float = 12.0,
    ) -> Any:
        timeout_s = min(self._effective_timeout(context, default_timeout), max(default_timeout, 20.0))
        return await asyncio.wait_for(brain.think(prompt), timeout=timeout_s)

    @staticmethod
    def _scan_python_file(path: Path) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "file": str(path),
            "exists": path.exists(),
            "long_functions": [],
            "action_items": [],
            "parse_error": None,
        }
        if not path.exists():
            return summary

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('self_evolution', exc)
            summary["parse_error"] = str(exc)
            return summary

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                length = max(1, end_lineno - node.lineno + 1)
                if length > 45:
                    summary["long_functions"].append(
                        {"name": node.name, "line": node.lineno, "length": length}
                    )

        for lineno, line in enumerate(source.splitlines(), start=1):
            if "to-do" in line.lower():
                summary["action_items"].append({"line": lineno, "text": line.strip()})

        return summary

    def _build_fallback_proposal(
        self,
        objective: str,
        files: list[str] | None,
        proprioception: dict[str, Any],
        research_summary: str,
        *,
        reason: str | None = None,
    ) -> str:
        target_paths = self._resolve_target_paths(files)
        file_summaries = [self._scan_python_file(path) for path in target_paths[:3]]

        lines = [
            "# Self-Evolution Proposal",
            "",
            "## Objective",
            objective or "Stabilize and improve the identified target.",
            "",
            "## Operating Context",
            f"- Repository root: `{self.code_base}`",
            f"- Proprioception snapshot: `{research_summary}`",
        ]
        if reason:
            lines.append(f"- Planning mode: deterministic fallback (`{reason}`)")

        if file_summaries:
            lines.extend(["", "## File Analysis"])
            for summary in file_summaries:
                resolved_file = Path(summary["file"]).resolve()
                try:
                    display_path = resolved_file.relative_to(self.code_base.resolve()).as_posix()
                except ValueError:
                    display_path = str(resolved_file)
                lines.append(f"- `{display_path}`")
                if summary["parse_error"]:
                    lines.append(f"  - Parse friction: {summary['parse_error']}")
                    continue
                if summary["long_functions"]:
                    for item in summary["long_functions"][:4]:
                        lines.append(
                            f"  - Long function `{item['name']}` at line {item['line']} spans {item['length']} lines."
                        )
                else:
                    lines.append("  - No oversized functions detected by the structural scanner.")
                if summary["action_items"]:
                    for item in summary["action_items"][:3]:
                        lines.append(f"  - Action item at line {item['line']}: {item['text']}")

        lines.extend(
            [
                "",
                "## Implementation Plan",
                "1. Narrow the change to the named target file and preserve public behavior.",
                "2. Extract or split oversized functions into smaller helpers with explicit names.",
                "3. Keep all path filtering, priority ordering, and export limits behaviorally identical unless tests prove an intended improvement.",
                "4. Add or refresh regression tests around the touched behavior before applying a live patch.",
                "",
                "## Safety Rails",
                "- Do not broaden file inclusion or weaken exclusion rules.",
                "- Preserve deterministic ordering so exports remain reproducible.",
                "- Prefer internal helper extraction over semantic rewrites.",
                "",
                "## Validation",
                "- Run the targeted unit tests for the touched module.",
                "- Re-run the self-development/autonomy visibility checks if background messaging behavior changes.",
                "- Run the canonical audit suite after patching to confirm no wider regression.",
            ]
        )

        if proprioception:
            lines.extend(["", "## Live Constraints", f"- Current proprioception: `{json.dumps(proprioception, sort_keys=True)}`"])

        return "\n".join(lines)

    async def execute(self, params: EvolutionInput, context: dict[str, Any]) -> dict[str, Any]:
        """Execute self-evolution loop."""
        context = context or {}
        if isinstance(params, dict):
            try:
                params = EvolutionInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('self_evolution', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action
        objective = params.objective
        read_only = bool(params.read_only or self._is_read_only(context))

        if action == "scramble":
            return self._perform_scrambling()
            
        if not objective:
            return {"ok": False, "error": "No objective provided for self-evolution."}

        self.logger.info("Initiating Self-Evolution for: %s", objective)

        # 1. RESEARCH - Grounded in Proprioception
        proprioception = context.get("proprioception", {})
        research_summary = f"System Status: {json.dumps(proprioception)}"
        brain = self._resolve_brain(context)
        fallback_reason: str | None = None
        proposal: str | None = None
        
        # 2. SYNTHESIZE
        if read_only:
            fallback_reason = "read_only_deterministic_planning"
        elif brain:
            try:
                prompt = (
                    f"Objective: {objective}\n"
                    f"Body Proprioception: {research_summary}\n\n"
                    f"TASK: Perform a self-evolution cycle. Propose code changes to achieve the objective.\n"
                    f"Format: Provide a detailed Implementation Plan in Markdown.\n"
                    f"Security: Ensure all code matches Aura v14 'Sovereign' protocols."
                )
                thought = await self._think_with_timeout(brain, prompt, context, default_timeout=12.0)
                proposal = thought.content
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                fallback_reason = str(e)
                record_degradation(
                    "self_evolution",
                    e,
                    severity="warning",
                    action="used deterministic self-evolution plan after model proposal failed",
                    extra={"objective": objective[:240]},
                )
                self.logger.warning(
                    "Falling back to deterministic self-evolution planning for '%s': %s",
                    objective,
                    e,
                )
        else:
            fallback_reason = "cognitive_engine_unavailable"

        if not proposal:
            proposal = self._build_fallback_proposal(
                objective,
                params.files,
                proprioception,
                research_summary,
                reason=fallback_reason,
            )

        if action == "apply":
            if read_only:
                return {
                    "ok": False,
                    "error": "Read-only execution cannot apply live self-evolution changes.",
                    "read_only": True,
                }
            if not brain:
                return {
                    "ok": False,
                    "error": "Live apply requires a cognitive engine. A proposal was generated instead.",
                    "fallback": True,
                    "results": proposal[:self.MAX_PREVIEW],
                }
            return await self._apply_evolution(objective, params.files, brain, context)

        # 3. PROPOSE
        if read_only:
            return {
                "ok": True,
                "summary": "Proposal drafted in read-only mode.",
                "results": proposal[:self.MAX_PREVIEW],
                "fallback": bool(fallback_reason),
                "read_only": True,
            }

        timestamp = int(time.time())
        filename = f"evolution_proposal_{timestamp}.md"
        filepath = self._evolution_dir() / filename
        try:
            await get_file_write_gateway().write_text_async(
                filepath,
                f"# Self-Evolution Proposal\n\n**Objective**: {objective}\n\n{proposal}",
                source="skills.self_evolution.proposal",
            )
            self.logger.info("Proposal saved to %s", filepath)
            return {
                "ok": True,
                "summary": f"Proposal created: {filename}",
                "proposal_path": str(filepath),
                "results": proposal[:self.MAX_PREVIEW],
                "fallback": bool(fallback_reason),
            }
        except OSError as e:
            record_degradation('self_evolution', e)
            return {"ok": False, "error": f"Failed to save proposal: {e}"}

    async def _apply_evolution(self, objective: str, files: list[str] | None, brain: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Apply the evolution objective autonomously using sandboxed verification."""
        self.logger.info("Applying Self-Evolution for: %s", objective)
        
        # 1. Identify target file/line if not provided
        if not files:
             # Use LLM to identify the best file to target
             search_prompt = f"Objective: {objective}\nTask: Identify the SINGLE most relevant file in the codebase to modify."
             search_thought = await self._think_with_timeout(brain, search_prompt, context, default_timeout=10.0)
             # Basic extraction for now
             target_file = search_thought.content.strip().split('\n')[0].strip('` ')
        else:
             target_file = files[0]

        # 2. Use AutonomousCodeRepair to generate and test
        from core.self_modification.code_repair import AutonomousCodeRepair
        from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine
        
        repair_engine = AutonomousCodeRepair(brain, str(self.code_base))
        
        # Synthetic diagnosis
        diagnosis = {
            "hypotheses": [
                {
                    "root_cause": "Evolution Objective",
                    "explanation": objective,
                    "potential_fix": "Implement requested improvement",
                    "confidence": "high"
                }
            ]
        }
        
        # We need a line number. If it's a new feature, we might target line 1 or a specific function.
        # For simplicity in this upgrade, we target line 1.
        success, fix, test_results = await repair_engine.repair_bug(target_file, 1, diagnosis)
        
        if not success or not fix:
             error_msg = test_results.get("error") if isinstance(test_results, dict) else str(test_results)
             return {"ok": False, "error": f"Evolution failed in sandbox: {error_msg}"}
        
        # 3. Apply via SafeSelfModification (integrated in SME)
        sme = AutonomousSelfModificationEngine(brain, str(self.code_base))
        proposal = {
            "bug": {"pattern": {"events": [{"error_type": "evolution"}]}},
            "fix": fix,
            "test_results": test_results
        }
        
        final_success = await sme.apply_fix(proposal, force=False)
        
        if final_success:
             return {"ok": True, "summary": f"Evolution applied successfully to {target_file}", "applied": True}
        else:
             return {"ok": False, "error": "Failed to apply validated fix to live system."}

    def _perform_scrambling(self) -> dict[str, Any]:
        """Scramble is disabled. Returns honest status."""
        self.logger.info("Scramble requested but feature is disabled by safety protocols.")
        return {
            "ok": False,
            "status": "disabled",
            "message": "Source scrambling is disabled by Sovereign safety protocols to protect core integrity."
        }

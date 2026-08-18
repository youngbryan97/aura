"""Error Intelligence System - Autonomous Bug Detection & Analysis
Tracks execution, detects patterns, and generates diagnoses.
"""
import asyncio
import hashlib
import json
import logging
import os
import time
import traceback as tb
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.governance_context import governed_scope_sync
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("SelfModification.ErrorIntelligence")
_SOURCE_ROOT_REALPATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


#: Checkouts that live INSIDE the source root and are not her body.
#:
#: .claude/worktrees holds other agents' working copies. They pass the
#: containment check below because they are literally under the source root,
#: so a traceback through one was accepted as a location in her own code and
#: self-repair went after it: 178 repair attempts against
#: worktrees/codex-autonomy-deferral/core/collective/delegator.py alone, 54
#: more against a codex-wow-cp400 copy, and FileNotFoundError storms for
#: worktrees that had since been deleted.
#:
#: Editing them is wrong twice over. They are someone else's in-flight work,
#: and they are transient — a fix applied there is discarded with the branch,
#: so the same "bug" is rediscovered and re-repaired forever while her actual
#: source keeps the defect.
_NESTED_CHECKOUT_MARKERS = (
    f"{os.sep}.claude{os.sep}worktrees{os.sep}",
    f"{os.sep}dist{os.sep}",
    f"{os.sep}site-packages{os.sep}",
    f"{os.sep}.venv{os.sep}",
)


def _is_her_own_source(resolved_path: str) -> bool:
    """Inside this checkout AND not inside a checkout nested within it."""
    try:
        if (
            os.path.commonpath((_SOURCE_ROOT_REALPATH, resolved_path))
            != _SOURCE_ROOT_REALPATH
        ):
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    return not any(marker in resolved_path for marker in _NESTED_CHECKOUT_MARKERS)


def _deepest_aura_traceback_frame(error: BaseException) -> tuple[str | None, int | None]:
    for frame in reversed(tb.extract_tb(getattr(error, "__traceback__", None))):
        try:
            resolved_frame = os.path.realpath(frame.filename)
            if not _is_her_own_source(resolved_frame):
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        # traceback frames carry lineno as int | None; a frame with no line
        # number is still a useful location, so report 0 rather than crash
        # the error handler that is already handling an error.
        return resolved_frame, int(frame.lineno or 0)
    return None, None


@dataclass
class ErrorEvent:
    """Structured error representation"""

    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    context: dict[str, Any]
    skill_name: str | None = None
    goal: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def fingerprint(self) -> str:
        """Generate unique identifier for this error type"""
        if self.file_path or self.line_number:
            path = self.file_path or "unknown_file"
            line = str(self.line_number) if self.line_number else "0"
            key = f"{self.error_type}:location:{path}:{line}"
        else:
            # Synthetic health incidents have no traceback. Group them by
            # their structured subsystem/reason identity so unrelated runtime
            # degradations do not all collapse into RuntimeError:unknown:0.
            subsystem = str(self.context.get("subsystem") or self.skill_name or "unknown")
            reason = str(self.context.get("reason") or self.goal or "unknown")
            classification = str(self.context.get("classification") or "unknown")
            key = (
                f"{self.error_type}:structured:{subsystem}:"
                f"{reason}:{classification}"
            )
        return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class ErrorPattern:
    """Cluster of similar errors"""

    fingerprint: str
    occurrences: int
    first_seen: float
    last_seen: float
    events: list[ErrorEvent]
    severity: str  # 'critical', 'high', 'medium', 'low'
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "severity": self.severity,
            "sample_events": [e.to_dict() for e in self.events[:3]]  # First 3
        }


class StructuredErrorLogger:
    """Comprehensive error tracking system.
    Logs every error with full context for analysis.
    """
    
    def __init__(self, log_dir: str | None = None):
        if log_dir is None:
            from core.config import config
            self.log_dir = config.paths.data_dir / "error_logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_log_path = self.log_dir / "error_events.jsonl"
        self.execution_log_path = self.log_dir / "execution_log.jsonl"
        
        # In-memory cache for fast access
        self.recent_errors: list[ErrorEvent] = []
        from core.memory.retention_policy import working_history_retention_policy
        self.max_recent = working_history_retention_policy("AURA_ERROR_INTELLIGENCE_RECENT_MAX").max_items
        
        logger.info("StructuredErrorLogger initialized at %s", self.log_dir)
    
    async def log_error(
        self,
        error: Exception,
        context: dict[str, Any],
        skill_name: str | None = None,
        goal: str | None = None
    ) -> ErrorEvent:
        """Log an error with full context (Async)."""
        # Extract stack trace information
        stack_trace = "".join(
            tb.format_exception(type(error), error, getattr(error, "__traceback__", None))
        )
        # Tracebacks are ordered outermost to innermost. Resolve paths through
        # symlinks and choose the deepest frame inside this Aura checkout.
        # String matching against a few checkout names lost locations in
        # /private/tmp worktrees and any future installation prefix.
        file_path, line_number = _deepest_aura_traceback_frame(error)
        # Create error event
        event = ErrorEvent(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=stack_trace,
            context=context,
            skill_name=skill_name,
            goal=goal,
            file_path=file_path,
            line_number=line_number
        )
        
        # Store in memory
        self.recent_errors.append(event)
        if len(self.recent_errors) > self.max_recent:
            self.recent_errors = self.recent_errors[-self.max_recent:]
        
        # Persist to disk (Async)
        await self._append_to_log(self.error_log_path, event.to_dict())
        
        if str(skill_name or "").strip().lower().startswith("omni_log_"):
            logger.debug(
                "Error logged from logging telemetry path: %s in %s",
                event.error_type,
                skill_name or "unknown",
            )
        else:
            reason = str(context.get("reason") or goal or "unknown")[:120]
            classification = str(context.get("classification") or "unknown")[:80]
            detail = str(context.get("detail") or event.error_message or "")[:240]
            logger.warning(
                "Error logged: %s in %s reason=%s classification=%s detail=%s",
                event.error_type,
                skill_name or "unknown",
                reason,
                classification,
                detail,
            )
        
        return event
    
    def log_execution(
        self,
        skill_name: str,
        goal: dict[str, Any],
        result: dict[str, Any],
        duration: float,
    ) -> None:
        """Log successful execution for comparison with failures.
        
        Args:
            skill_name: Which skill executed
            goal: What was attempted
            result: Outcome
            duration: Execution time in seconds

        """
        execution_event = {
            "timestamp": time.time(),
            "skill_name": skill_name,
            "goal": str(goal),
            "success": result.get("ok", False),
            "duration": duration,
            "result": result
        }

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._append_to_log_sync(self.execution_log_path, execution_event)
        else:
            get_task_tracker().create_task(
                self._append_to_log(self.execution_log_path, execution_event),
                name="error_intelligence.log_execution",
            )

    def _append_to_log_sync(self, path: Path, data: dict[str, Any]) -> None:
        line = json.dumps(data) + '\n'
        decision = SimpleNamespace(
            receipt_id=f"self_mod_error_log:{time.time_ns()}",
            domain="self_modification",
            source="self_modification.error_intelligence",
            constraints={
                "operation": "append_error_intelligence_log",
                "path": str(path),
            },
        )
        with governed_scope_sync(decision):
            get_file_write_gateway().append_text(
                path,
                line,
                source="self_modification.error_intelligence.execution_log",
            )

    async def _append_to_log(self, path: Path, data: dict[str, Any]) -> None:
        """Append JSON line to log file without blocking the event loop."""
        try:
            await asyncio.to_thread(self._append_to_log_sync, path, data)
        except asyncio.CancelledError:
            logger.debug("Log append cancelled for %s", path)
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as e:
            record_degradation('error_intelligence', e)
            logger.error("Failed to append to log %s: %s", path, e)
    
    def get_recent_errors(self, limit: int = 50) -> list[ErrorEvent]:
        """Get most recent errors"""
        return self.recent_errors[-limit:]
    
    def load_all_errors(self) -> list[ErrorEvent]:
        """Load all errors from disk (expensive operation)"""
        errors = []
        if self.error_log_path.exists():
            with open(self.error_log_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        errors.append(ErrorEvent(**data))
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        record_degradation('error_intelligence', e)
                        logger.error("Failed to parse error event: %s", e)
        return errors


class ErrorPatternAnalyzer:
    """Detects patterns in errors to identify recurring bugs.
    Uses clustering to group similar failures.
    """
    
    def __init__(self, error_logger: StructuredErrorLogger):
        self.logger_system = error_logger
        
        # Pattern storage
        self.patterns: dict[str, ErrorPattern] = {}
        
        # Thresholds (v18 Detection Overdrive)
        self.pattern_threshold = 2  # 2 occurrences = pattern (was 3)
        self.critical_threshold = 3  # 3 occurrences = critical (was 1)
        self.high_threshold = 3      # 3 occurrences = high
        
        logger.info("ErrorPatternAnalyzer initialized")
    
    def analyze_recent(self, window: int = 100) -> list[ErrorPattern]:
        """Analyze recent errors for patterns.
        
        Args:
            window: How many recent errors to analyze
            
        Returns:
            List of detected patterns

        """
        errors = self.logger_system.get_recent_errors(limit=window)
        return self._cluster_errors(errors)
    
    def analyze_all(self) -> list[ErrorPattern]:
        """Analyze all historical errors (expensive).
        
        Returns:
            List of all detected patterns

        """
        errors = self.logger_system.load_all_errors()
        return self._cluster_errors(errors)
    
    def _cluster_errors(self, errors: list[ErrorEvent]) -> list[ErrorPattern]:
        """Group errors by similarity.
        
        Args:
            errors: List of error events
            
        Returns:
            List of error patterns

        """
        # Group by fingerprint
        clusters = defaultdict(list)
        for error in errors:
            fingerprint = error.fingerprint()
            clusters[fingerprint].append(error)
        
        # Create patterns
        patterns = []
        for fingerprint, events in clusters.items():
            if len(events) >= self.pattern_threshold:
                # Determine severity (v18 Detection Overdrive)
                occurrences = len(events)
                # v18 FIX: Check for critical types even if occurrence counts are low
                is_crash = any(e.error_type in ["AttributeError", "TypeError", "ImportError", "ServiceNotFoundError", "SyntaxError"] for e in events)
                
                if occurrences >= self.critical_threshold and is_crash:
                    severity = 'critical'
                elif occurrences >= 7:
                    severity = 'high'
                elif occurrences >= self.high_threshold:
                    severity = 'high'
                elif occurrences >= 5:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                pattern = ErrorPattern(
                    fingerprint=fingerprint,
                    occurrences=occurrences,
                    first_seen=min(e.timestamp for e in events),
                    last_seen=max(e.timestamp for e in events),
                    events=events,
                    severity=severity
                )
                patterns.append(pattern)
        
        # Update internal storage
        for pattern in patterns:
            self.patterns[pattern.fingerprint] = pattern
        
        # Sort by severity and recency
        patterns.sort(
            key=lambda p: (
                {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[p.severity],
                -p.last_seen
            )
        )
        
        logger.info("Detected %d error patterns", len(patterns))
        return patterns
    
    def get_pattern(self, fingerprint: str) -> ErrorPattern | None:
        """Get specific pattern by fingerprint"""
        return self.patterns.get(fingerprint)
    
    def get_critical_patterns(self) -> list[ErrorPattern]:
        """Get only critical patterns that need immediate attention"""
        return [p for p in self.patterns.values() if p.severity == 'critical']
    
    def should_trigger_fix(self, pattern: ErrorPattern) -> bool:
        """Determine if a pattern warrants autonomous fix attempt.
        
        Args:
            pattern: Error pattern to evaluate
            
        Returns:
            True if should attempt fix

        """
        # Criteria for autonomous fixing (v18 Overdrive):
        # 1. At least 1-2 occurrences depending on severity
        # 2. Recent (within last hour or critical severity)
        # 3. Same error location (file + line) or critical type
        # 4. Not a systemic issue (doesn't affect too many different skills)
        
        needed = self.pattern_threshold if pattern.severity != 'critical' else 1
        if pattern.occurrences < needed:
            return False
        
        # Check recency
        one_hour_ago = time.time() - 3600
        recent = pattern.last_seen > one_hour_ago
        
        if pattern.severity == 'critical' or recent:
            # Check if it's localized (fixable)
            located_events = [
                event
                for event in pattern.events
                if event.file_path and event.line_number
            ]
            if not located_events:
                return False

            unique_files = set(e.file_path for e in located_events if e.file_path)
            unique_skills = set(e.skill_name for e in pattern.events if e.skill_name)
            
            # If error is in 1-2 files and 1-3 skills, it's probably fixable
            is_localized = len(unique_files) <= 2 and len(unique_skills) <= 3
            
            return is_localized
        
        return False


class AutomatedDiagnosisEngine:
    """Uses LLM to diagnose error patterns and propose root causes.
    """
    
    def __init__(self, cognitive_engine: Any) -> None:
        self.brain = cognitive_engine
        logger.info("AutomatedDiagnosisEngine initialized")

    def _deterministic_diagnosis(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Produce a cheap diagnosis without waking a local model."""
        sample = pattern.events[0] if pattern.events else None
        if sample is None:
            return {"ok": False, "error": "empty_pattern", "hypotheses": []}

        message = str(sample.error_message or "")
        error_type = str(sample.error_type or "Error")
        location = (
            f"{sample.file_path}:{sample.line_number}"
            if sample.file_path and sample.line_number
            else sample.file_path or "unknown location"
        )

        if "is not defined" in message.lower() or error_type == "NameError":
            root = "A symbol is referenced before it is imported, declared, or made available in this scope."
            test = f"Run a focused import/compile check for {location} and the failing call path."
            fix = "Add the missing import/definition or guard the reference with the existing optional-service pattern."
            confidence = "high"
        elif "was never awaited" in message.lower() or "coroutine" in message.lower():
            root = "An async coroutine is being called like a synchronous function."
            test = f"Exercise the scheduler path that reaches {location} with runtime warnings enabled."
            fix = "Await the coroutine or schedule it through the task tracker with a stable task name."
            confidence = "high"
        elif "timeout" in message.lower():
            root = "A background operation exceeded its live-runtime budget or contended with the foreground lane."
            test = "Replay the operation under the background policy gates and confirm it defers under load."
            fix = "Move expensive work behind background policy, shorten the budget, or replace it with a deterministic fast path."
            confidence = "medium"
        else:
            root = f"{error_type} recurred at {location} and should be isolated before any autonomous patch."
            test = "Run the narrowest reproducer for the recorded traceback and inspect adjacent recent changes."
            fix = "Prefer a small targeted guard or local invariant restoration after the reproducer confirms the cause."
            confidence = "low"

        return {
            "ok": True,
            "hypotheses": [
                {
                    "root_cause": root,
                    "explanation": f"The same {error_type} occurred {pattern.occurrences} time(s), centered on {location}: {message[:240]}",
                    "diagnostic_test": test,
                    "potential_fix": fix,
                    "confidence": confidence,
                }
            ],
            "additional_context_needed": "",
            "diagnosis_source": "deterministic_static",
        }
    
    async def diagnose_pattern(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Generate diagnosis for an error pattern.
        
        Args:
            pattern: Error pattern to diagnose
            
        Returns:
            Diagnosis dictionary with hypotheses and suggested tests

        """
        logger.info("Diagnosing pattern %s (%d occurrences)", pattern.fingerprint, pattern.occurrences)

        use_llm = str(os.environ.get("AURA_SELFMOD_LLM_DIAGNOSIS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not use_llm:
            return self._deterministic_diagnosis(pattern)
        
        # Build diagnostic prompt
        prompt = self._build_diagnostic_prompt(pattern)
        
        # Get LLM analysis
        try:
            thought = await self.brain.think(
                prompt,
                priority=0.1,
                origin="self_modification_diagnosis",
                is_background=True,
            )
            # Guard: brain.think() returns None when all LLM endpoints are down
            raw_content = ""
            if thought is None:
                raw_content = ""
            elif hasattr(thought, 'content'):
                raw_content = str(thought.content or "")
            else:
                raw_content = str(thought or "")

            if not raw_content.strip():
                logger.debug("Diagnosis skipped: LLM returned no content (endpoints may be unavailable)")
                return {
                    "ok": False,
                    "error": "llm_returned_empty",
                    "hypotheses": []
                }

            diagnosis = self._parse_diagnosis(raw_content)
            
            logger.info("Generated %d hypotheses", len(diagnosis.get('hypotheses', [])))
            return diagnosis
            
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('error_intelligence', e)
            logger.error("Diagnosis failed: %s", e)
            return {
                "ok": False,
                "error": str(e),
                "hypotheses": []
            }
    
    def _build_diagnostic_prompt(self, pattern: ErrorPattern) -> str:
        """Build prompt for LLM diagnosis"""
        # Get sample events
        samples = pattern.events[:5]  # First 5 occurrences
        
        # Extract context
        error_type = samples[0].error_type
        error_messages = [e.error_message for e in samples]
        stack_traces = [e.stack_trace for e in samples]
        
        # Get common file/line
        file_path = samples[0].file_path
        line_number = samples[0].line_number
        
        prompt = f'''You are diagnosing a recurring bug in your own code.

ERROR PATTERN ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Severity: {pattern.severity.upper()}
Occurrences: {pattern.occurrences}
First seen: {time.ctime(pattern.first_seen)}
Last seen: {time.ctime(pattern.last_seen)}

Error Type: {error_type}
Location: {file_path}:{line_number}

Sample Error Messages:
{chr(10).join('- ' + msg for msg in error_messages[:3])}

Sample Stack Trace:
{stack_traces[0]}

TASK: Generate 2-3 hypotheses for the root cause of this error.

For each hypothesis, provide:
1. Root cause (what's actually broken)
2. Why this explains the error pattern
3. Diagnostic test (how to confirm this hypothesis)
4. Potential fix (if hypothesis is correct)

Return your analysis as JSON:
{{
  "hypotheses": [
    {{
      "root_cause": "Description of what's broken",
      "explanation": "Why this causes the observed error",
      "diagnostic_test": "How to verify this hypothesis",
      "potential_fix": "What code change would fix it",
      "confidence": "high/medium/low"
    }}
  ],
  "additional_context_needed": "What information would help narrow this down"
}}

Return ONLY the JSON, no other text.'''
        
        return prompt
    
    def _parse_diagnosis(self, response: str) -> dict[str, Any]:
        """Parse LLM diagnosis response"""
        # Try to extract JSON
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        
        try:
            diagnosis = json.loads(response)
            if not isinstance(diagnosis, dict):
                raise json.JSONDecodeError("diagnosis must be a JSON object", response, 0)
            diagnosis["ok"] = True
            return {str(key): value for key, value in diagnosis.items()}
        except json.JSONDecodeError as e:
            logger.error("Failed to parse diagnosis JSON: %s", e)
            logger.debug("Response was: %s", response[:500])
            return {
                "ok": False,
                "error": "json_parse_failed",
                "raw_response": response,
                "hypotheses": []
            }


# Integration helper
class ErrorIntelligenceSystem:
    """Complete error intelligence system combining logging, analysis, and diagnosis.
    """
    
    def __init__(self, cognitive_engine: Any, log_dir: str | None = None) -> None:
        self.logger_system = StructuredErrorLogger(log_dir)
        self.analyzer = ErrorPatternAnalyzer(self.logger_system)
        self.diagnostics = AutomatedDiagnosisEngine(cognitive_engine)
        
        logger.info("ErrorIntelligenceSystem fully initialized")
    
    async def on_error(
        self,
        error: Exception,
        context: dict[str, Any],
        skill_name: str | None = None,
        goal: str | None = None
    ) -> ErrorEvent:
        """Handle an error occurrence (Async)"""
        return await self.logger_system.log_error(error, context, skill_name, goal)
    
    def on_execution(
        self,
        skill_name: str,
        goal: dict[str, Any],
        result: dict[str, Any],
        duration: float,
    ) -> None:
        """Handle a successful execution"""
        self.logger_system.log_execution(skill_name, goal, result, duration)
    
    async def find_bugs_to_fix(self) -> list[dict[str, Any]]:
        """Find bugs that should be fixed autonomously.
        
        Returns:
            List of bugs with diagnoses, sorted by priority

        """
        # Analyze recent errors
        patterns = self.analyzer.analyze_recent(window=200)
        
        # Filter to fixable patterns
        fixable = [p for p in patterns if self.analyzer.should_trigger_fix(p)]
        
        # Generate diagnoses
        bugs_with_diagnosis: list[dict[str, Any]] = []
        for pattern in fixable:
            diagnosis = await self.diagnostics.diagnose_pattern(pattern)
            if diagnosis.get("ok") and diagnosis.get("hypotheses"):
                bugs_with_diagnosis.append({
                    "pattern": pattern,
                    "diagnosis": diagnosis,
                    "priority": self._calculate_priority(pattern)
                })
        
        # Sort by priority
        bugs_with_diagnosis.sort(key=lambda x: x["priority"], reverse=True)
        
        return bugs_with_diagnosis
    
    def _calculate_priority(self, pattern: ErrorPattern) -> float:
        """Calculate fix priority.
        
        Returns:
            Priority score (higher = more urgent)

        """
        severity_scores = {
            'critical': 100,
            'high': 50,
            'medium': 25,
            'low': 10
        }
        
        severity_score = severity_scores.get(pattern.severity, 0)
        
        # Recency bonus (errors in last hour get boost)
        one_hour_ago = time.time() - 3600
        recency_bonus = 50 if pattern.last_seen > one_hour_ago else 0
        
        # Frequency factor
        frequency_factor = min(pattern.occurrences / 10, 2.0)  # Cap at 2x
        
        priority = (severity_score + recency_bonus) * frequency_factor
        
        return priority
    
    def get_status(self) -> dict[str, Any]:
        """Get current error intelligence status"""
        recent_errors = self.logger_system.get_recent_errors(limit=50)
        patterns = self.analyzer.analyze_recent(window=200)
        critical = self.analyzer.get_critical_patterns()
        
        return {
            "recent_error_count": len(recent_errors),
            "total_patterns": len(patterns),
            "critical_patterns": len(critical),
            "patterns_by_severity": {
                "critical": len([p for p in patterns if p.severity == 'critical']),
                "high": len([p for p in patterns if p.severity == 'high']),
                "medium": len([p for p in patterns if p.severity == 'medium']),
                "low": len([p for p in patterns if p.severity == 'low'])
            }
        }

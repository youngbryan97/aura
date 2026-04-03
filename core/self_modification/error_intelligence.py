"""Error Intelligence System - Autonomous Bug Detection & Analysis
Tracks execution, detects patterns, and generates diagnoses.
"""
import asyncio
import hashlib
import json
import logging
import time
import traceback as tb
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SelfModification.ErrorIntelligence")


@dataclass
class ErrorEvent:
    """Structured error representation"""

    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    skill_name: Optional[str] = None
    goal: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)
    
    def fingerprint(self) -> str:
        """Generate unique identifier for this error type"""
        # Hash based on error type + location, not message (messages vary)
        # v18: Fallback for context-free errors to still allow grouping
        path = self.file_path or "unknown_file"
        line = str(self.line_number) if self.line_number else "0"
        key = f"{self.error_type}:{path}:{line}"
        return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class ErrorPattern:
    """Cluster of similar errors"""

    fingerprint: str
    occurrences: int
    first_seen: float
    last_seen: float
    events: List[ErrorEvent]
    severity: str  # 'critical', 'high', 'medium', 'low'
    
    def to_dict(self):
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
    
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            from core.config import config
            self.log_dir = config.paths.data_dir / "error_logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_log_path = self.log_dir / "error_events.jsonl"
        self.execution_log_path = self.log_dir / "execution_log.jsonl"
        
        # In-memory cache for fast access
        self.recent_errors: List[ErrorEvent] = []
        self.max_recent = 1000
        
        logger.info("StructuredErrorLogger initialized at %s", self.log_dir)
    
    async def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        skill_name: Optional[str] = None,
        goal: Optional[str] = None
    ) -> ErrorEvent:
        """Log an error with full context (Async)."""
        # Extract stack trace information
        stack_trace = tb.format_exc()
        trace_lines = stack_trace.split('\n')
        
        # Try to find the actual error location (not in framework code)
        file_path = None
        line_number = None
        for line in trace_lines:
            if 'File "' in line and '/autonomy_engine/' in line:
                # Parse: File "/path/to/file.py", line 123
                try:
                    file_part = line.split('File "')[1].split('"')[0]
                    line_part = line.split('line ')[1].split(',')[0]
                    file_path = file_part
                    line_number = int(line_part)
                    break
                except Exception:
                    logger.debug("Traceback line parse failed: %s", line)
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
        
        logger.warning("Error logged: %s in %s", event.error_type, skill_name or 'unknown')
        
        return event
    
    def log_execution(
        self,
        skill_name: str,
        goal: Dict[str, Any],
        result: Dict[str, Any],
        duration: float
    ):
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
        
        self._append_to_log(self.execution_log_path, execution_event)
    
    async def _append_to_log(self, path: Path, data: Dict[str, Any]):
        """Append JSON line to log file (Async)"""
        try:
            line = json.dumps(data) + '\n'
            def _write():
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(line)
            await asyncio.to_thread(_write)
        except Exception as e:
            logger.error("Failed to append to log %s: %s", path, e)
    
    def get_recent_errors(self, limit: int = 50) -> List[ErrorEvent]:
        """Get most recent errors"""
        return self.recent_errors[-limit:]
    
    def load_all_errors(self) -> List[ErrorEvent]:
        """Load all errors from disk (expensive operation)"""
        errors = []
        if self.error_log_path.exists():
            with open(self.error_log_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        errors.append(ErrorEvent(**data))
                    except Exception as e:
                        logger.error("Failed to parse error event: %s", e)
        return errors


class ErrorPatternAnalyzer:
    """Detects patterns in errors to identify recurring bugs.
    Uses clustering to group similar failures.
    """
    
    def __init__(self, error_logger: StructuredErrorLogger):
        self.logger_system = error_logger
        
        # Pattern storage
        self.patterns: Dict[str, ErrorPattern] = {}
        
        # Thresholds (v18 Detection Overdrive)
        self.pattern_threshold = 2  # 2 occurrences = pattern (was 3)
        self.critical_threshold = 3  # 3 occurrences = critical (was 1)
        self.high_threshold = 3      # 3 occurrences = high
        
        logger.info("ErrorPatternAnalyzer initialized")
    
    def analyze_recent(self, window: int = 100) -> List[ErrorPattern]:
        """Analyze recent errors for patterns.
        
        Args:
            window: How many recent errors to analyze
            
        Returns:
            List of detected patterns

        """
        errors = self.logger_system.get_recent_errors(limit=window)
        return self._cluster_errors(errors)
    
    def analyze_all(self) -> List[ErrorPattern]:
        """Analyze all historical errors (expensive).
        
        Returns:
            List of all detected patterns

        """
        errors = self.logger_system.load_all_errors()
        return self._cluster_errors(errors)
    
    def _cluster_errors(self, errors: List[ErrorEvent]) -> List[ErrorPattern]:
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
    
    def get_pattern(self, fingerprint: str) -> Optional[ErrorPattern]:
        """Get specific pattern by fingerprint"""
        return self.patterns.get(fingerprint)
    
    def get_critical_patterns(self) -> List[ErrorPattern]:
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
            unique_files = set(e.file_path for e in pattern.events if e.file_path)
            unique_skills = set(e.skill_name for e in pattern.events if e.skill_name)
            
            # If error is in 1-2 files and 1-3 skills, it's probably fixable
            is_localized = len(unique_files) <= 2 and len(unique_skills) <= 3
            
            return is_localized
        
        return False


class AutomatedDiagnosisEngine:
    """Uses LLM to diagnose error patterns and propose root causes.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        logger.info("AutomatedDiagnosisEngine initialized")
    
    async def diagnose_pattern(self, pattern: ErrorPattern) -> Dict[str, Any]:
        """Generate diagnosis for an error pattern.
        
        Args:
            pattern: Error pattern to diagnose
            
        Returns:
            Diagnosis dictionary with hypotheses and suggested tests

        """
        logger.info("Diagnosing pattern %s (%d occurrences)", pattern.fingerprint, pattern.occurrences)
        
        # Build diagnostic prompt
        prompt = self._build_diagnostic_prompt(pattern)
        
        # Get LLM analysis
        try:
            thought = await self.brain.think(prompt, priority=0.1)
            diagnosis = self._parse_diagnosis(thought.content if hasattr(thought, 'content') else str(thought))
            
            logger.info("Generated %d hypotheses", len(diagnosis.get('hypotheses', [])))
            return diagnosis
            
        except Exception as e:
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
    
    def _parse_diagnosis(self, response: str) -> Dict[str, Any]:
        """Parse LLM diagnosis response"""
        # Try to extract JSON
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        
        try:
            diagnosis = json.loads(response)
            diagnosis["ok"] = True
            return diagnosis
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
    
    def __init__(self, cognitive_engine, log_dir: Optional[str] = None):
        self.logger_system = StructuredErrorLogger(log_dir)
        self.analyzer = ErrorPatternAnalyzer(self.logger_system)
        self.diagnostics = AutomatedDiagnosisEngine(cognitive_engine)
        
        logger.info("ErrorIntelligenceSystem fully initialized")
    
    async def on_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        skill_name: Optional[str] = None,
        goal: Optional[str] = None
    ) -> ErrorEvent:
        """Handle an error occurrence (Async)"""
        return await self.logger_system.log_error(error, context, skill_name, goal)
    
    def on_execution(
        self,
        skill_name: str,
        goal: Dict[str, Any],
        result: Dict[str, Any],
        duration: float
    ):
        """Handle a successful execution"""
        self.logger_system.log_execution(skill_name, goal, result, duration)
    
    async def find_bugs_to_fix(self) -> List[Dict[str, Any]]:
        """Find bugs that should be fixed autonomously.
        
        Returns:
            List of bugs with diagnoses, sorted by priority

        """
        # Analyze recent errors
        patterns = self.analyzer.analyze_recent(window=200)
        
        # Filter to fixable patterns
        fixable = [p for p in patterns if self.analyzer.should_trigger_fix(p)]
        
        # Generate diagnoses
        bugs_with_diagnosis = []
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
    
    def get_status(self) -> Dict[str, Any]:
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
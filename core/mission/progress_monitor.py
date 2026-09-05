"""core/mission/progress_monitor.py — Mission Progress Monitor.

Tracks per-campaign progress with timestamps, metrics, forecasts,
and abort condition detection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("Aura.ProgressMonitor")


@dataclass
class ProgressEntry:
    campaign_id: str
    milestone_id: str
    event: str
    timestamp: float = field(default_factory=time.time)
    details: str = ""


class MissionProgressMonitor:
    """Records and analyzes mission progress across campaigns."""

    def __init__(self) -> None:
        self.entries: List[ProgressEntry] = []
        self._forecasts: Dict[str, Dict[str, Any]] = {}

    def record_progress(self, campaign_id: str, milestone_id: str, event: str) -> None:
        entry = ProgressEntry(campaign_id=campaign_id, milestone_id=milestone_id, event=event)
        self.entries.append(entry)
        logger.debug("📈 Progress [%s] %s: %s", campaign_id, milestone_id, event)

    def get_campaign_progress(self, campaign_id: str) -> List[ProgressEntry]:
        return [e for e in self.entries if e.campaign_id == campaign_id]

    def compute_velocity(self, campaign_id: str) -> float:
        """Compute milestones completed per minute for a campaign."""
        events = self.get_campaign_progress(campaign_id)
        completed = [e for e in events if e.event in ("completed", "Step completed successfully.")]
        if len(completed) < 2:
            return 0.0
        duration = completed[-1].timestamp - completed[0].timestamp
        if duration <= 0:
            return 0.0
        return len(completed) / (duration / 60.0)

    def forecast_completion(self, campaign_id: str, remaining_milestones: int) -> Dict[str, Any]:
        """Estimate time to completion based on current velocity."""
        velocity = self.compute_velocity(campaign_id)
        if velocity <= 0:
            forecast = {"estimated_minutes": float("inf"), "confidence": 0.0, "velocity": 0.0}
        else:
            est_minutes = remaining_milestones / velocity
            forecast = {
                "estimated_minutes": round(est_minutes, 1),
                "confidence": min(0.9, 0.3 + velocity * 0.1),
                "velocity": round(velocity, 3),
            }
        self._forecasts[campaign_id] = forecast
        return forecast

    def detect_stall(self, campaign_id: str, stall_threshold_s: float = 300.0) -> bool:
        """Detect if a campaign has stalled (no progress for threshold seconds)."""
        events = self.get_campaign_progress(campaign_id)
        if not events:
            return False
        last_event_time = max(e.timestamp for e in events)
        return (time.time() - last_event_time) > stall_threshold_s

    def summary(self, campaign_id: str) -> Dict[str, Any]:
        events = self.get_campaign_progress(campaign_id)
        return {
            "campaign_id": campaign_id,
            "total_events": len(events),
            "velocity": self.compute_velocity(campaign_id),
            "stalled": self.detect_stall(campaign_id),
            "forecast": self._forecasts.get(campaign_id),
        }

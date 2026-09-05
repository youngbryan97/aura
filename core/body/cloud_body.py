"""core/body/cloud_body.py — Governed Cloud Body.

Manages remote compute nodes, job queues, embeddings jobs, and model servers
with strict budget limits, cost ceilings, and automatic shutdown policies.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.CloudBody")


@dataclass
class CloudJob:
    job_id: str
    resource_type: str  # worker_node, embedding_node, model_server
    allocated_cores: int
    cost_per_hour: float
    started_at: float
    max_duration_hours: float
    shutdown_scheduled_at: float
    status: str = "active"  # active, completed, terminated


class CloudBody:
    """Governs the allocation and teardown of remote cloud resources."""

    def __init__(self, hourly_budget_limit: float = 100.0) -> None:
        self.hourly_budget_limit = hourly_budget_limit
        self.budget_limit = 100.0  # Total budget limit
        self.total_cost_incurred = 0.0
        self.active_jobs: dict[str, CloudJob] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self._job_counter = 0
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Governed Cloud Body initialized.")

    def register_node(self, node_id: str, region: str, cost_per_hour: float) -> None:
        """Register a remote compute provider node."""
        self.nodes[node_id] = {
            "node_id": node_id,
            "region": region,
            "cost_per_hour": cost_per_hour,
        }
        logger.info("☁️  CloudBody: registered node '%s' ($%.2f/hr) in %s", node_id, cost_per_hour, region)

    def request_compute_allocation(
        self,
        task_name: str,
        estimated_hours: float,
        node_id: str,
    ) -> bool:
        """Request a compute allocation, checking overall cost budgets."""
        node = self.nodes.get(node_id)
        if not node:
            logger.error("🚫 CloudBody: node '%s' not registered", node_id)
            return False

        cost = node["cost_per_hour"] * estimated_hours
        if (self.total_cost_incurred + cost) > self.budget_limit:
            logger.warning("🚫 CloudBody: allocation refused. Cost $%0.2f exceeds remaining budget limit.", cost)
            return False

        self.total_cost_incurred += cost
        logger.info("☁️  CloudBody: allocation approved for task '%s' cost=$%0.2f (total=$%0.2f)",
                    task_name, cost, self.total_cost_incurred)
        return True

    def request_compute_node(
        self,
        resource_type: str,
        cores: int = 4,
        max_duration_hours: float = 1.0,
    ) -> dict[str, Any] | None:
        """Request a remote worker node under strict cost checks."""
        # Calculate expected cost
        cost_rates = {"worker_node": 0.50, "embedding_node": 0.25, "model_server": 1.50}
        rate = cost_rates.get(resource_type, 0.50) * (cores / 4)
        projected_cost = rate * max_duration_hours

        # Check budget limits
        current_hourly_cost = sum(j.cost_per_hour for j in self.active_jobs.values() if j.status == "active")
        if (current_hourly_cost + rate) > self.hourly_budget_limit:
            logger.warning("🚫 CloudBody: request denied. Hourly cost ceiling exceeded.")
            return None

        self._job_counter += 1
        job_id = f"cloud_job_{self._job_counter}_{int(time.time())}"
        now = time.time()

        job = CloudJob(
            job_id=job_id,
            resource_type=resource_type,
            allocated_cores=cores,
            cost_per_hour=rate,
            started_at=now,
            max_duration_hours=max_duration_hours,
            shutdown_scheduled_at=now + (max_duration_hours * 3600.0),
        )
        self.active_jobs[job_id] = job
        logger.info("☁️  CloudBody: provisioned '%s' cores=%d, hourly_rate=$%.2f/hr", job_id, cores, rate)

        return {
            "job_id": job_id,
            "status": "provisioned",
            "cost_rate_per_hour": rate,
            "auto_shutdown_in": f"{max_duration_hours} hours",
        }

    def enforce_shutdown_policy(self) -> list[str]:
        """Tears down any active instances that exceed their lease duration."""
        now = time.time()
        terminated = []

        for job in list(self.active_jobs.values()):
            if job.status == "active" and now >= job.shutdown_scheduled_at:
                job.status = "terminated"
                duration = (now - job.started_at) / 3600.0
                cost = duration * job.cost_per_hour
                self.total_cost_incurred += cost
                terminated.append(job.job_id)
                logger.warning("⏰ CloudBody: Auto-shutdown triggered for job %s. Restoring budget.", job.job_id)

        return terminated

    def terminate_all(self) -> None:
        """Teardown every cloud worker instantly."""
        logger.warning("🚨 CloudBody: Emergency teardown of all cloud compute nodes.")
        for job in self.active_jobs.values():
            if job.status == "active":
                job.status = "terminated"
        self.active_jobs.clear()

"""Signed local-network bridge for multi-node Aura operation."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MycelialPacket:
    sender: str
    topic: str
    payload: dict[str, Any]
    timestamp: float
    signature: str

    def body(self) -> dict[str, Any]:
        return {"sender": self.sender, "topic": self.topic, "payload": self.payload, "timestamp": self.timestamp}

    def to_json(self) -> str:
        return json.dumps({**self.body(), "signature": self.signature}, sort_keys=True)


class MycelialBridge:
    """Signs distilled artifacts for safe local federation."""

    def __init__(self, node_id: str, secret: str) -> None:
        self.node_id = node_id
        self.secret = secret.encode("utf-8")

    def sign(self, topic: str, payload: dict[str, Any]) -> MycelialPacket:
        timestamp = time.time()
        body = {"sender": self.node_id, "topic": topic, "payload": payload, "timestamp": timestamp}
        signature = hmac.new(self.secret, json.dumps(body, sort_keys=True).encode("utf-8"), hashlib.sha256).hexdigest()
        return MycelialPacket(self.node_id, topic, payload, timestamp, signature)

    def verify(self, packet: MycelialPacket) -> bool:
        expected = hmac.new(self.secret, json.dumps(packet.body(), sort_keys=True).encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, packet.signature)

    def safe_payload(self, success_trace: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_type": success_trace.get("task_type"),
            "verifier_score": success_trace.get("score"),
            "adapter_delta_hash": success_trace.get("adapter_delta_hash"),
            "raw_memory_included": False,
        }


__all__ = ["MycelialPacket", "MycelialBridge"]

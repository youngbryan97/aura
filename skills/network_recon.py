"""Legacy bounded network-awareness compatibility skill.

This module no longer performs ARP, ping, subnet scanning, or stealth network
discovery. Active network work belongs behind the canonical governed network
gateway and the `sovereign_network` skill.
"""

from __future__ import annotations

import socket
from typing import Any

from infrastructure import BaseSkill


class NetworkReconSkill(BaseSkill):
    name = "network_recon"
    description = "Report local host network identity without scanning or probing devices."
    effect_scope = "external_io"

    async def execute(
        self,
        goal: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hostname = socket.gethostname()
        local_addresses: list[str] = []
        try:
            infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        except (OSError, RuntimeError, ValueError):
            infos = []
        for info in infos:
            address = str(info[4][0])
            if address not in local_addresses:
                local_addresses.append(address)

        return {
            "ok": True,
            "status": "local_identity_only",
            "local_hostname": hostname,
            "local_addresses": local_addresses,
            "devices": [],
            "summary": "NetworkRecon is constrained to local identity; no network scan or ARP probe was performed.",
        }

"""Boot catalog for optional concrete Reality Reach connector families."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, replace
from typing import Any
from core.runtime.flags import env_str

_OPENHAB_CONNECTOR_ID = "openhab.local"


@dataclass(frozen=True, slots=True)
class ConnectorBootStatus:
    connector_id: str
    configured: bool
    registered: bool
    state: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "configured": self.configured,
            "registered": self.registered,
            "state": self.state,
            "error": self.error,
        }


class RealityConnectorCatalog:
    """Own boot-time connector construction without retaining credentials."""

    def __init__(
        self,
        connectors: tuple[Any, ...],
        statuses: tuple[ConnectorBootStatus, ...],
    ) -> None:
        self._connectors = connectors
        self._statuses = statuses

    @property
    def connectors(self) -> tuple[Any, ...]:
        return self._connectors

    def register_with(self, broker: Any) -> None:
        register = getattr(broker, "register_connector", None)
        if not callable(register):
            raise TypeError("broker must expose register_connector")
        registered: set[str] = set()
        for connector in self._connectors:
            register(connector)
            registered.add(connector.connector_id)
        self._statuses = tuple(
            replace(
                status,
                registered=status.connector_id in registered,
                state=("registered" if status.connector_id in registered else status.state),
            )
            for status in self._statuses
        )

    def status(self) -> dict[str, Any]:
        entries = [status.to_dict() for status in self._statuses]
        return {
            "alive": True,
            "ready": not any(item["state"] == "invalid" for item in entries),
            "configured": sum(bool(item["configured"]) for item in entries),
            "registered": sum(bool(item["registered"]) for item in entries),
            "connectors": entries,
        }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])

    async def stop(self) -> None:
        for connector in reversed(self._connectors):
            stop = getattr(connector, "stop", None)
            if not callable(stop):
                continue
            result = stop()
            if inspect.isawaitable(result):
                await result


def build_configured_reality_connector_catalog() -> RealityConnectorCatalog:
    """Build configured connectors; absence is valid, partial config is explicit."""

    connectors: list[Any] = []
    statuses: list[ConnectorBootStatus] = []
    url = str(env_str("AURA_OPENHAB_URL", description="openHAB URL", owner="core.embodiment.reality_connectors") or "").strip()
    token = str(env_str("AURA_OPENHAB_TOKEN", description="openHAB token", owner="core.embodiment.reality_connectors") or "").strip()
    if not url and not token:
        statuses.append(
            ConnectorBootStatus(
                connector_id=_OPENHAB_CONNECTOR_ID,
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not url or not token:
        statuses.append(
            ConnectorBootStatus(
                connector_id=_OPENHAB_CONNECTOR_ID,
                configured=True,
                registered=False,
                state="invalid",
                error=(
                    "AURA_OPENHAB_URL is missing" if not url else "AURA_OPENHAB_TOKEN is missing"
                ),
            )
        )
    else:
        try:
            from core.embodiment.openhab_connector import (
                OpenHABConnector,
                OpenHABTransport,
            )

            connector = OpenHABConnector(OpenHABTransport())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id=_OPENHAB_CONNECTOR_ID,
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    mqtt_url = str(env_str("AURA_MQTT_BROKER_URL", description="MQTT broker URL", owner="core.embodiment.reality_connectors") or "").strip()
    mqtt_manifest = str(env_str("AURA_MQTT_RESOURCES_JSON", description="MQTT resources JSON", owner="core.embodiment.reality_connectors") or "").strip()
    mqtt_installation = str(env_str("AURA_MQTT_INSTALLATION_ID", description="MQTT installation id", owner="core.embodiment.reality_connectors") or "").strip()
    mqtt_present = bool(mqtt_url or mqtt_manifest or mqtt_installation)
    if not mqtt_present:
        statuses.append(
            ConnectorBootStatus(
                connector_id="mqtt.manifest",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not mqtt_url or not mqtt_manifest or not mqtt_installation:
        missing = [
            name
            for name, value in (
                ("AURA_MQTT_BROKER_URL", mqtt_url),
                ("AURA_MQTT_RESOURCES_JSON", mqtt_manifest),
                ("AURA_MQTT_INSTALLATION_ID", mqtt_installation),
            )
            if not value
        ]
        statuses.append(
            ConnectorBootStatus(
                connector_id="mqtt.manifest",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.mqtt_connector import (
                build_configured_mqtt_connector,
            )

            connector = build_configured_mqtt_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="mqtt.manifest",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    opcua_endpoint = str(env_str("AURA_OPCUA_ENDPOINT", description="OPC UA endpoint", owner="core.embodiment.opcua") or "").strip()
    opcua_manifest = str(env_str("AURA_OPCUA_RESOURCES_JSON", description="OPC UA resources JSON", owner="core.embodiment.opcua") or "").strip()
    opcua_installation = str(env_str("AURA_OPCUA_INSTALLATION_ID", description="OPC UA installation id", owner="core.embodiment.opcua") or "").strip()
    opcua_present = bool(opcua_endpoint or opcua_manifest or opcua_installation)
    if not opcua_present:
        statuses.append(
            ConnectorBootStatus(
                connector_id="opcua.manifest",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not opcua_endpoint or not opcua_manifest or not opcua_installation:
        missing = [
            name
            for name, value in (
                ("AURA_OPCUA_ENDPOINT", opcua_endpoint),
                ("AURA_OPCUA_RESOURCES_JSON", opcua_manifest),
                ("AURA_OPCUA_INSTALLATION_ID", opcua_installation),
            )
            if not value
        ]
        statuses.append(
            ConnectorBootStatus(
                connector_id="opcua.manifest",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.opcua_connector import (
                build_configured_opcua_connector,
            )

            connector = build_configured_opcua_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="opcua.manifest",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    rosbridge_url = str(env_str("AURA_ROSBRIDGE_URL", description="rosbridge URL", owner="core.embodiment.reality_connectors") or "").strip()
    rosbridge_manifest = str(env_str("AURA_ROSBRIDGE_NODE_MANIFEST_JSON", description="rosbridge node manifest JSON", owner="core.embodiment.reality_connectors") or "").strip()
    rosbridge_installation = str(env_str("AURA_ROSBRIDGE_INSTALLATION_ID", description="rosbridge installation id", owner="core.embodiment.reality_connectors") or "").strip()
    rosbridge_present = bool(rosbridge_url or rosbridge_manifest or rosbridge_installation)
    if not rosbridge_present:
        statuses.append(
            ConnectorBootStatus(
                connector_id="ros2.rosbridge",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not rosbridge_url or not rosbridge_manifest or not rosbridge_installation:
        missing = [
            name
            for name, value in (
                ("AURA_ROSBRIDGE_URL", rosbridge_url),
                ("AURA_ROSBRIDGE_NODE_MANIFEST_JSON", rosbridge_manifest),
                ("AURA_ROSBRIDGE_INSTALLATION_ID", rosbridge_installation),
            )
            if not value
        ]
        statuses.append(
            ConnectorBootStatus(
                connector_id="ros2.rosbridge",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.ros2_connector import (
                build_configured_ros2_connector,
            )

            connector = build_configured_ros2_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="ros2.rosbridge",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    scpi_endpoint = str(env_str("AURA_SCPI_ENDPOINT", description="SCPI endpoint", owner="core.embodiment.reality_connectors") or "").strip()
    scpi_manifest = str(env_str("AURA_SCPI_RESOURCES_JSON", description="SCPI resources JSON", owner="core.embodiment.reality_connectors") or "").strip()
    scpi_installation = str(env_str("AURA_SCPI_INSTALLATION_ID", description="SCPI installation id", owner="core.embodiment.reality_connectors") or "").strip()
    scpi_expected_idn = str(env_str("AURA_SCPI_EXPECTED_IDN_SHA256", description="SCPI expected IDN sha256", owner="core.embodiment.reality_connectors") or "").strip()
    scpi_present = bool(scpi_endpoint or scpi_manifest or scpi_installation or scpi_expected_idn)
    if not scpi_present:
        statuses.append(
            ConnectorBootStatus(
                connector_id="scpi.manifest",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not all((scpi_endpoint, scpi_manifest, scpi_installation, scpi_expected_idn)):
        missing = [
            name
            for name, value in (
                ("AURA_SCPI_ENDPOINT", scpi_endpoint),
                ("AURA_SCPI_RESOURCES_JSON", scpi_manifest),
                ("AURA_SCPI_INSTALLATION_ID", scpi_installation),
                ("AURA_SCPI_EXPECTED_IDN_SHA256", scpi_expected_idn),
            )
            if not value
        ]
        statuses.append(
            ConnectorBootStatus(
                connector_id="scpi.manifest",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.scpi_connector import (
                build_configured_scpi_connector,
            )

            connector = build_configured_scpi_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="scpi.manifest",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    azure_dt_values = {
        "AURA_AZURE_DT_ENDPOINT": str(env_str("AURA_AZURE_DT_ENDPOINT", description="Azure digital-twin endpoint", owner="core.embodiment.reality_connectors") or "").strip(),
        "AURA_AZURE_DT_INSTANCE_ID": str(env_str("AURA_AZURE_DT_INSTANCE_ID", description="Azure digital-twin instance id", owner="core.embodiment.reality_connectors") or "").strip(),
        "AURA_AZURE_DT_RESOURCES_JSON": str(
            env_str("AURA_AZURE_DT_RESOURCES_JSON", description="Azure digital-twin resources JSON", owner="core.embodiment.reality_connectors") or ""
        ).strip(),
        "AURA_AZURE_DT_TENANT_ID": str(env_str("AURA_AZURE_DT_TENANT_ID", description="Azure digital-twin tenant id", owner="core.embodiment.reality_connectors") or "").strip(),
        "AURA_AZURE_DT_CLIENT_ID": str(env_str("AURA_AZURE_DT_CLIENT_ID", description="Azure digital-twin client id", owner="core.embodiment.reality_connectors") or "").strip(),
        "AURA_AZURE_DT_CLIENT_SECRET": str(env_str("AURA_AZURE_DT_CLIENT_SECRET", description="Azure digital-twin client secret", owner="core.embodiment.reality_connectors") or "").strip(),
    }
    if not any(azure_dt_values.values()):
        statuses.append(
            ConnectorBootStatus(
                connector_id="azure.digital_twins",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not all(azure_dt_values.values()):
        missing = [name for name, value in azure_dt_values.items() if not value]
        statuses.append(
            ConnectorBootStatus(
                connector_id="azure.digital_twins",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.azure_digital_twins_connector import (
                build_configured_azure_digital_twins_connector,
            )

            connector = build_configured_azure_digital_twins_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="azure.digital_twins",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    aws_twinmaker_required = {
        "AURA_AWS_TWINMAKER_WORKSPACE_ARN": str(
            env_str("AURA_AWS_TWINMAKER_WORKSPACE_ARN", description="AWS TwinMaker workspace ARN", owner="core.embodiment.reality_connectors") or ""
        ).strip(),
        "AURA_AWS_TWINMAKER_RESOURCES_JSON": str(
            env_str("AURA_AWS_TWINMAKER_RESOURCES_JSON", description="AWS TwinMaker resources JSON", owner="core.embodiment.reality_connectors") or ""
        ).strip(),
        "AWS_ACCESS_KEY_ID": str(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        "AWS_SECRET_ACCESS_KEY": str(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    }
    aws_twinmaker_intent = bool(
        aws_twinmaker_required["AURA_AWS_TWINMAKER_WORKSPACE_ARN"]
        or aws_twinmaker_required["AURA_AWS_TWINMAKER_RESOURCES_JSON"]
    )
    if not aws_twinmaker_intent:
        statuses.append(
            ConnectorBootStatus(
                connector_id="aws.iot_twinmaker",
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not all(aws_twinmaker_required.values()):
        missing = [name for name, value in aws_twinmaker_required.items() if not value]
        statuses.append(
            ConnectorBootStatus(
                connector_id="aws.iot_twinmaker",
                configured=True,
                registered=False,
                state="invalid",
                error=f"missing configuration: {','.join(missing)}",
            )
        )
    else:
        try:
            from core.embodiment.aws_twinmaker_connector import (
                build_configured_aws_twinmaker_connector,
            )

            connector = build_configured_aws_twinmaker_connector()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id="aws.iot_twinmaker",
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    return RealityConnectorCatalog(tuple(connectors), tuple(statuses))


__all__ = [
    "ConnectorBootStatus",
    "RealityConnectorCatalog",
    "build_configured_reality_connector_catalog",
]

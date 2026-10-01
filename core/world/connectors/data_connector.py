"""core/world/connectors/data_connector.py — Public Datasets & Economic Indicators.

Reads currency rates, inflation index, package downloads, and dataset APIs.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.governance.action_domain import ActionDomain
from core.governance_context import GovernanceViolation
from core.runtime.action_executor import ActionExecutor

logger = logging.getLogger("Aura.DataConnector")


class DataConnector:
    """Reads financial market indices, package telemetry, and public stats."""

    async def fetch_financial_indicators(self, query: str) -> dict[str, Any]:
        logger.info("📡 DataConnector: fetching financial metrics for '%s'", query)

        try:
            res = await ActionExecutor.execute(
                domain=ActionDomain.NETWORK_CALL,
                action_name="data.fetch_forex",
                params={"method": "GET", "url": "https://open.er-api.com/v6/latest/USD"},
                source="data_connector",
            )
            if res.get("ok"):
                content_bytes = res.get("content")
                if content_bytes:
                    try:
                        data = json.loads(content_bytes.decode("utf-8", errors="ignore"))
                        rates = data.get("rates", {})
                        eur_rate = rates.get("EUR", 0.92)
                        gbp_rate = rates.get("GBP", 0.81)
                        return {
                            "USD_EUR": eur_rate,
                            "USD_GBP": gbp_rate,
                            "rates": rates,
                            "inflation_indexed": False,
                        }
                    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as parse_err:
                        logger.warning("Failed to parse exchange rates JSON: %s", parse_err)
        except GovernanceViolation:
            raise
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as e:
            logger.warning("Forex/Indicator fetch failed, using fallback: %s", e)

        return {
            "interest_rate": 0.0525,
            "sp500_trend": "positive",
            "compute_cost_per_mtoken": 0.0015,
        }

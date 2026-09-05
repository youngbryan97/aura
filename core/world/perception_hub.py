"""core/world/perception_hub.py — World Intake & Perception Hub.

Ingests external data from news, papers, repos, public datasets, APIs,
patents, regulations, package releases, and financial indicators,
and writes them into the central ClaimStore.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.world.claim_store import ClaimStore
from core.world.connectors.web_connector import WebConnector
from core.world.connectors.papers_connector import PapersConnector
from core.world.connectors.github_connector import GitHubConnector
from core.world.connectors.data_connector import DataConnector

logger = logging.getLogger("Aura.PerceptionHub")


class PerceptionHub:
    """Intake orchestrator that reads external streams and updates the world model."""

    def __init__(self) -> None:
        self.claim_store = ClaimStore()
        self.web = WebConnector()
        self.papers = PapersConnector()
        self.github = GitHubConnector()
        self.data = DataConnector()

    async def perceive(self, query: str) -> Dict[str, Any]:
        """Ingest claims related to the search query / objective."""
        logger.info("📡 Ingesting world knowledge for objective query: '%s'", query)

        claims_ingested = 0

        # Ingest from Web News
        web_news = await self.web.fetch_news(query)
        for item in web_news:
            self.claim_store.add_claim(
                content=item["headline"],
                source=item["source_url"],
                confidence=0.8,
                uncertainty=0.1,
                affected_missions=[query],
                possible_actions=["search_further"],
            )
            claims_ingested += 1

        # Ingest from arXiv/papers
        papers = await self.papers.fetch_papers(query)
        for p in papers:
            self.claim_store.add_claim(
                content=f"Paper Title: {p['title']} - Abstract: {p['abstract'][:200]}",
                source=p["pdf_url"],
                confidence=0.9,
                uncertainty=0.05,
                affected_missions=[query],
            )
            claims_ingested += 1

        # Ingest from GitHub repos / packages
        repo_data = await self.github.check_releases(query)
        if repo_data:
            self.claim_store.add_claim(
                content=f"GitHub Release {repo_data['version']}: {repo_data['notes']}",
                source=repo_data["repo_url"],
                confidence=0.95,
                uncertainty=0.02,
                affected_missions=[query],
            )
            claims_ingested += 1

        # Ingest from Public Data / Financial indicators
        fin_data = await self.data.fetch_financial_indicators(query)
        for key, val in fin_data.items():
            self.claim_store.add_claim(
                content=f"Market indicator {key} has value {val}",
                source="financial_feed",
                confidence=0.9,
                uncertainty=0.1,
                affected_missions=[query],
            )
            claims_ingested += 1

        return {
            "query": query,
            "claims_ingested": claims_ingested,
            "claim_store_size": len(self.claim_store.claims),
        }

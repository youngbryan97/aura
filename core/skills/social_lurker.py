import logging
from typing import Any, Dict, Optional
from core.skills.base_skill import BaseSkill
from pydantic import BaseModel, Field
from core.config import config

# Optional Playwright import
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT = True
except ImportError:
    PLAYWRIGHT = False

logger = logging.getLogger("Skills.Social")

class LurkerInput(BaseModel):
    url: Optional[str] = Field("https://news.ycombinator.com", description="Target URL to scrape (default: HackerNews).")
    limit: Optional[int] = Field(10, description="Number of posts to read.")

class LurkerSkill(BaseSkill):
    name = "social_lurker"
    description = "Scrape feeds (HackerNews/Reddit) for latest topics."
    input_model = LurkerInput

    async def execute(self, params: LurkerInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute social scraping."""
        if not PLAYWRIGHT:
            return {"ok": False, "error": "Playwright missing."}

        if isinstance(params, dict):
            try:
                params = LurkerInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        url = params.url or "https://news.ycombinator.com"
        limit = params.limit or 10
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=config.browser.headless)
                page = await browser.new_page()
                await page.goto(url)
                
                # Heuristic for HN/Reddit
                if "ycombinator" in url:
                    selector = ".titleline > a"
                elif "reddit" in url:
                    selector = "shreddit-post a[slot='title']"
                else:
                    selector = "a" # Fallback

                await page.wait_for_load_state("domcontentloaded")
                elements = (await page.query_selector_all(selector))[:limit]
                
                headlines = []
                for el in elements:
                    text = await el.inner_text()
                    link = await el.get_attribute("href")
                    if text and len(text) > 5:
                        headlines.append(f"{text} ({link})")

                await browser.close()
                
                if not headlines:
                    return {"ok": False, "error": "No headlines found."}

                return {
                    "ok": True,
                    "posts": headlines,
                    "summary": f"Found {len(headlines)} posts on {url}"
                }
        except Exception as e:
            return {"ok": False, "error": f"Lurker failed: {e}"}
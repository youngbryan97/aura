"""Phantom Browser Module
Playwright-based "human-like" browser agent for Aura.

Capabilities:
- Dynamic Visibility: Headless (background) vs Headed (interactive)
- Human-like Interaction: Random microsleeps, typing speeds, cursor movements
- Robust Navigation: Handling broken links, backing out, reading content
- Content Extraction: Getting markdown from pages

Usage:
    browser = PhantomBrowser()
    browser.browse("https://aura.internal")
    browser.type("input[name='q']", "Hello World")
    browser.click("input[name='btnK']")
"""
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

try:
    from playwright.async_api import Browser, ElementHandle, Page, async_playwright, Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

logger = logging.getLogger("PhantomBrowser")

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1"
]

_CONTENT_HINT_RE = re.compile(
    r"(article|story|content|entry|main|body|markdown|post|read|chapter|prose|text)",
    re.IGNORECASE,
)
_NOISE_HINT_RE = re.compile(
    r"(nav|footer|header|menu|sidebar|cookie|share|social|comment|promo|banner|breadcrumb|related|recommend|subscribe|login|signup|advert)",
    re.IGNORECASE,
)
_NOISY_LINE_RE = re.compile(
    r"^(?:home|about|menu|privacy|terms|cookies?|share|subscribe|login|sign up|contact|next|previous|advertisement)$",
    re.IGNORECASE,
)


def _clean_extracted_page_text(raw_text: str) -> str:
    seen: set[str] = set()
    cleaned_lines: list[str] = []
    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        if _NOISY_LINE_RE.match(line):
            continue
        if len(line) < 18 and not re.match(r"^(chapter|part|section)\b", lower):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned_lines.append(line)
    return "\n\n".join(cleaned_lines)


def _score_content_block(title: str, block: Dict[str, Any]) -> float:
    text = _clean_extracted_page_text(str(block.get("text") or ""))
    if not text:
        return float("-inf")

    tag = str(block.get("tag") or "").lower()
    block_id = str(block.get("id") or "").lower()
    class_name = str(block.get("class_name") or "").lower()
    meta = " ".join(part for part in (tag, block_id, class_name) if part)
    title_tokens = set(re.findall(r"[a-z0-9]+", str(title or "").lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text[:1200].lower()))
    token_overlap = len(title_tokens & text_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0

    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    paragraph_count = int(block.get("paragraph_count") or 0)
    link_density = float(block.get("link_density") or 0.0)
    score = min(len(text) / 180.0, 8.0)
    score += min(sentence_count * 0.12, 2.0)
    score += min(paragraph_count * 0.18, 1.8)
    score += token_overlap * 1.2

    if tag in {"article", "main"}:
        score += 1.2
    if _CONTENT_HINT_RE.search(meta):
        score += 0.8
    if _NOISE_HINT_RE.search(meta):
        score -= 2.4
    score -= min(link_density, 0.8) * 3.0
    return score

class PhantomBrowser:
    """High-fidelity browser agent (Async Version).
    """
    
    def __init__(self, visible: bool = False, browser_type: str = "chromium"):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self.visible = visible
        self.browser_type = browser_type
        self.is_active = False
        self._homeostasis = None
        
        if not PLAYWRIGHT_AVAILABLE:
            return

    async def ensure_ready(self):
        """Public lifecycle method: ensures the browser is started and ready."""
        if not self.is_active:
            await self._start_browser()

    async def _start_browser(self):
        """Start the Playwright browser asynchronously"""
        try:
            if self.is_active:
                return

            # HARDENING: Signal resource lock — heavy background tasks will pause
            try:
                from core.utils.resource_lock import get_resource_lock
                self._resource_lock = get_resource_lock()
                self._resource_lock._browser_sessions += 1
                self._resource_lock._total_browser_sessions += 1
                self._resource_lock._browser_idle.clear()
            except Exception:
                self._resource_lock = None

            self.playwright = await async_playwright().start()

            # RESILIENCE: Build a fallback cascade of browser types.
            # If the configured browser (e.g. Firefox) isn't installed,
            # fall back to chromium which is the most reliably available.
            browser_attempts = [self.browser_type]
            if self.browser_type != "chromium":
                browser_attempts.append("chromium")

            launch_error = None
            for bt in browser_attempts:
                try:
                    if bt == "firefox":
                        self.browser = await self.playwright.firefox.launch(headless=not self.visible)
                    elif bt == "webkit":
                        self.browser = await self.playwright.webkit.launch(headless=not self.visible)
                    else:
                        self.browser = await self.playwright.chromium.launch(
                            headless=not self.visible,
                            args=['--disable-blink-features=AutomationControlled']
                        )
                    if bt != self.browser_type:
                        logger.info("✓ Fell back to %s after %s was unavailable.", bt, self.browser_type)
                    launch_error = None
                    break  # Launch succeeded
                except Exception as launch_exc:
                    record_degradation('phantom_browser', launch_exc)
                    launch_error = launch_exc
                    logger.warning("Browser %s failed to launch: %s. Trying next fallback...", bt, launch_exc)

            if launch_error or self.browser is None:
                raise launch_error or RuntimeError("All browser types failed to launch.")

            user_agent = self._get_random_ua()

            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent=user_agent
            )
            self.page = await self.context.new_page()

            # Apply Stealth for anti-detection (2026 upgrade)
            if STEALTH_AVAILABLE:
                try:
                    await stealth_async(self.page)
                except Exception as se:
                    record_degradation('phantom_browser', se)
                    logger.warning("Stealth application failed: %s", se)
            else:
                logger.warning("playwright-stealth is not installed. Running without stealth.")

            self.is_active = True
            logger.info("✓ Phantom Browser initialized (Visible: %s, UA: %s...)", self.visible, user_agent[:30])
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Failed to start browser: %s", e)
            self.is_active = False
            # Release resource lock on failure
            self._release_resource_lock()

    def _get_random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    async def rotate_user_agent(self):
        """Switch to a new context with a different user agent."""
        if not self.is_active:
            await self._start_browser()
            return
            
        logger.info("🔄 Rotating User Agent...")
        ua = self._get_random_ua()
        
        # We need a new context to change the UA effectively
        new_context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent=ua
        )
        old_context = self.context
        old_page = self.page
        self.context = new_context
        self.page = await self.context.new_page()

        # Apply Stealth to new context/page
        try:
            await Stealth().apply_stealth_async(self.page)
        except Exception as e:
            record_degradation('phantom_browser', e)
            capture_and_log(e, {'module': __name__})
        
        # Properly close old page and context to prevent resource leaks
        if old_page:
            try:
                await old_page.close()
            except Exception:
                pass  # no-op: intentional
        if old_context:
            try:
                await old_context.close()
            except Exception:
                pass  # no-op: intentional
        logger.info("✓ User Agent rotated to: %s...", ua[:30])

    async def is_blocked(self) -> bool:
        """Detect if we are hitting a bot-detection page or CAPTCHA."""
        if not self.page: return False
        
        content = (await self.page.content()).lower()
        title = (await self.page.title()).lower()
        
        block_signals = [
            "unusual traffic from your computer network",
            "not a robot",
            "captcha",
            "verify you are a human",
            "access to this page has been denied",
            "security check",
            "bot detection",
            "automated requests"
        ]
        
        for signal in block_signals:
            if signal in content or signal in title:
                logger.warning("🚨 Browser Blocked Detected: %s", signal)
                return True
        return False

    async def set_visibility(self, visible: bool):
        """Toggle visibility (requires restart)"""
        if self.visible != visible:
            logger.info("Switching visibility: %s -> %s", self.visible, visible)
            self.visible = visible
            await self.close()
            await self._start_browser()

    async def browse(self, url: str) -> bool:
        """Navigate to a URL"""
        if not self.is_active: 
            await self._start_browser()
            if not self.is_active:
                logger.error("Browser failed to start.")
                return False
            
        if not url.startswith('http'):
            url = 'https://' + url
            
        logger.info("🌐 Navigating to: %s", url)
        try:
            await self.page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await self._human_delay(1, 2)
            return True
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Navigation failed: %s", e)
            return False

    async def click(self, selector: str = None, text_match: str = None) -> bool:
        """Human-like click on an element with enhanced robustness."""
        try:
            element = None
            if text_match:
                # Try multiple ways to find text (case-insensitive, contains)
                selectors = [
                    f"text='{text_match}'",
                    f"text=\"{text_match}\"",
                    f"a:has-text('{text_match}')",
                    f"button:has-text('{text_match}')",
                    f"*[role='button']:has-text('{text_match}')"
                ]
                for s in selectors:
                    try:
                        loc = self.page.locator(s).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                            break
                    except PlaywrightError:
                        continue
                
                if not element:
                    # Fallback to get_by_text with regex for fuzzy match
                    import re
                    try:
                        loc = self.page.get_by_text(re.compile(text_match, re.IGNORECASE)).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                    except Exception:
                        logger.debug('Exception caught during execution: %s', e if 'e' in locals() or 'e' in globals() else 'unknown')
            elif selector:
                element = self.page.locator(selector).first

            if element and await element.is_visible():
                # Scroll into view if needed
                await element.scroll_into_view_if_needed()
                await self._human_delay(0.2, 0.5)
                
                # Human-like mouse movement
                box = await element.bounding_box()
                if box:
                    x = box['x'] + box['width'] / 2 + random.randint(-5, 5)
                    y = box['y'] + box['height'] / 2 + random.randint(-3, 3)
                    await self.page.mouse.move(x, y, steps=15)
                    await self._human_delay(0.1, 0.3)
                
                await element.click()
                logger.info("🖱️ Clicked: %s", selector or text_match)
                await self._human_delay(0.5, 1.5)
                return True
            else:
                logger.warning("Element not found or not visible: %s", selector or text_match)
                return False
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Click failed: %s", e)
            return False

    async def type(self, selector: str, text: str) -> bool:
        """Human-like typing"""
        try:
            await self.click(selector) # Focus first
            
            logger.info("⌨️ Typing: '%s'", text)
            for char in text:
                await self.page.keyboard.type(char)
                # Random typing delay between keystrokes
                await asyncio.sleep(random.uniform(0.05, 0.15))
            
            await self._human_delay(0.5, 1.0)
            return True
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Typing failed: %s", e)
            return False

    async def scroll(self, direction: str = "down", amount: int = 500):
        """Scroll the page in a human-like manner."""
        try:
            steps = 5
            step_amount = amount // steps
            for _ in range(steps):
                if direction == "down":
                    await self.page.mouse.wheel(0, step_amount)
                else:
                    await self.page.mouse.wheel(0, -step_amount)
                await asyncio.sleep(random.uniform(0.1, 0.3))
            await self._human_delay(0.5, 1.0)
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Scroll failed: %s", e)

    async def read_content(self) -> str:
        """Extract page content by scoring likely article/content containers."""
        try:
            if not self.page: return ""
            
            title = await self.page.title()

            candidate_blocks = await self.page.evaluate("""() => {
                function isVisible(el) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0' &&
                        rect.width > 0 &&
                        rect.height > 0;
                }

                function collectCandidates() {
                    const selectors = [
                        'article',
                        'main',
                        '[role="main"]',
                        '[itemprop="articleBody"]',
                        '.article',
                        '.article-body',
                        '.entry-content',
                        '.post-content',
                        '.story-content',
                        '.story-body',
                        '.main-content',
                        '.content',
                        'section',
                        'body',
                    ];
                    const seen = new Set();
                    const blocks = [];
                    for (const selector of selectors) {
                        const nodes = Array.from(document.querySelectorAll(selector));
                        for (const el of nodes) {
                            if (!el || seen.has(el) || !isVisible(el)) continue;
                            const text = (el.innerText || '').replace(/\\u00a0/g, ' ').trim();
                            if (text.length < 80) continue;
                            seen.add(el);
                            const linkText = Array.from(el.querySelectorAll('a'))
                                .map(a => (a.innerText || '').trim())
                                .join(' ');
                            blocks.push({
                                tag: (el.tagName || '').toLowerCase(),
                                id: (el.id || ''),
                                class_name: (el.className || '').toString(),
                                text,
                                paragraph_count: el.querySelectorAll('p, li, blockquote').length,
                                heading_count: el.querySelectorAll('h1, h2, h3, h4').length,
                                link_density: text.length ? (linkText.length / text.length) : 0,
                            });
                        }
                    }
                    return blocks.slice(0, 24);
                }

                return collectCandidates();
            }""")

            best_text = ""
            best_score = float("-inf")
            for block in list(candidate_blocks or []):
                if not isinstance(block, dict):
                    continue
                score = _score_content_block(title, block)
                text = _clean_extracted_page_text(str(block.get("text") or ""))
                if text and score > best_score:
                    best_score = score
                    best_text = text

            if len(best_text) < 200:
                fallback_text = await self.page.evaluate("() => document.body.innerText")
                fallback_text = _clean_extracted_page_text(str(fallback_text or ""))
                if len(fallback_text) > len(best_text):
                    best_text = fallback_text

            return f"# {title}\n\n{best_text[:60000]}"
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Read content failed: %s", e)
            return ""

    async def get_links(self) -> List[Dict[str, str]]:
        """Extract all links from page"""
        try:
            if not self.page: return []
            links = await self.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    url: a.href
                })).filter(l => l.text && l.url)
            }""")
            return links
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Get links failed: %s", e)
            return []

    async def screenshot(self) -> Optional[str]:
        """Take a screenshot (base64)"""
        try:
            if not self.page: return None
            import base64
            bytes_data = await self.page.screenshot()
            return base64.b64encode(bytes_data).decode('utf-8')
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.error("Screenshot failed: %s", e)
            return None

    async def close(self):
        """Close browser resources with per-step timeouts to prevent hangs."""
        try:
            if self.page:
                try:
                    await asyncio.wait_for(self.page.close(), timeout=3.0)
                except Exception:
                    pass  # no-op: intentional
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.debug("Page close error: %s", e)
        try:
            if self.context:
                try:
                    await asyncio.wait_for(self.context.close(), timeout=5.0)
                except Exception:
                    pass  # no-op: intentional
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.debug("Context close error: %s", e)
        try:
            if self.browser:
                try:
                    await asyncio.wait_for(self.browser.close(), timeout=5.0)
                except Exception:
                    pass  # no-op: intentional
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.debug("Browser close error: %s", e)
        try:
            if self.playwright:
                try:
                    await asyncio.wait_for(self.playwright.stop(), timeout=5.0)
                except Exception:
                    pass  # no-op: intentional
        except Exception as e:
            record_degradation('phantom_browser', e)
            logger.debug("Playwright stop error: %s", e)
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self.is_active = False
        self._release_resource_lock()
        logger.info("Browser closed")

    def _release_resource_lock(self):
        """Release the resource lock so background tasks can resume."""
        lock = getattr(self, '_resource_lock', None)
        if lock:
            lock._browser_sessions = max(0, lock._browser_sessions - 1)
            if lock._browser_sessions <= 0:
                lock._browser_idle.set()

    async def _human_delay(self, min_s=0.5, max_s=1.5):
        """Random delay to simulate human pause, modulated by homeostasis."""
        if self._homeostasis is None:
            try:
                from core.container import ServiceContainer
                self._homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
            except Exception as e:
                record_degradation('phantom_browser', e)
                capture_and_log(e, {'module': __name__})
        
        delay_mod = 1.0
        if self._homeostasis:
            mods = self._homeostasis.get_modifiers()
            # Exhaustion (low vitality) makes her move SLOWER
            if mods.overall_vitality < 0.4:
                delay_mod = 2.5 # Significant fatigue delay
            elif mods.overall_vitality < 0.7:
                delay_mod = 1.5 # Mild fatigue delay
            
            # Urgency makes her move FASTER
            if mods.urgency_flag:
                delay_mod *= 0.6
                
        await asyncio.sleep(random.uniform(min_s, max_s) * delay_mod)

# Integration Helper
async def integrate_phantom_browser(orchestrator):
    """Integrate Phantom Browser into Orchestrator"""
    pb = PhantomBrowser(visible=False)
    await pb.ensure_ready()
    orchestrator.phantom_browser = pb
    logger.info("✅ Phantom Browser integrated")

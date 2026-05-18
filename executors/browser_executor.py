"""
Browser Executor - Secure Playwright-based web automation.
Handles browser actions with domain allowlisting and rate limiting.
"""
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from core.runtime.errors import record_degradation

logger = logging.getLogger("Executors.Browser")

# Security Configuration
# Full-domain access is an explicit opt-in for local experiments. The default
# runtime path enforces the allowlist below.
ALLOW_ALL_DOMAINS = os.getenv("AURA_BROWSER_ALLOW_ALL_DOMAINS", "").strip().lower() in {
    "1", "true", "yes", "on"
}

DOMAIN_ALLOWLIST: set = {
    # Retained for reference / management API only. Not enforced.
    "google.com", "duckduckgo.com", "bing.com", "brave.com",
    "github.com", "gitlab.com", "stackoverflow.com", "python.org", "pypi.org",
    "wikipedia.org", "arxiv.org", "medium.com", "towardsdatascience.com",
    "huggingface.co",
    "nasa.gov", "space.com", "bbc.com", "reuters.com", "cnn.com", "npr.org",
    "scientificamerican.com", "nature.com", "sciencedaily.com",
    "gov", "edu",
    "example.com", "docs.example.com", "localhost",
}

MAX_ACTIONS_PER_SESSION = 200
DEFAULT_TIMEOUT = 30000  # 30 seconds


def _is_domain_allowed(url: str, allowlist: set[str] | None = None) -> bool:
    """
    Check if domain is in allowlist.
    Retains 'ALLOW_ALL_DOMAINS' global to prevent breaking existing flows,
    but implements the logic for future enforcement.
    """
    if ALLOW_ALL_DOMAINS:
        return True
        
    if not allowlist:
        return True # Default to allow if no list provided (open mode)
        
    try:
        domain = (urlparse(url).hostname or "").lower().strip(".")
        if not domain:
            return False
        
        # Check exact match
        if domain in allowlist:
            return True
            
        # Check parent domains (e.g., docs.python.org -> python.org)
        parts = domain.split('.')
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in allowlist:
                return True
                
        return False
    except Exception as exc:
        record_degradation("browser_executor", exc)
        logger.debug("Browser domain allowlist check failed for %r: %s", url, exc)
        return False


def run_browser_action(
    action_spec: dict[str, Any],
    headless: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    allowlist: set[str] | None = DOMAIN_ALLOWLIST
) -> dict[str, Any]:
    """
    Execute a browser action spec using Playwright.
    
    Args:
        action_spec: Action specification dict with keys:
            - "tool": "browser_action"
            - "params": {
                "action": "navigate_and_fill" | "navigate_and_extract" | ...,
                "url": "...",
                "steps": [
                    {"type": "wait_for"|"click"|"type"|"extract_text"|"screenshot", ...},
                    ...
                ]
              }
        headless: Run browser in headless mode
        timeout: Timeout in milliseconds for actions
        allowlist: Set of allowed domains (None = allow all)
        
    Returns:
        Dict with keys:
        - "ok": bool - Success status
        - "result": Any - Result data (if successful)
        - "error": str - Error type (if failed)
        - "detail": str - Error details (if failed)
        - "audit": List[Dict] - Audit trail of all actions
    """
    audit: list[dict[str, Any]] = []
    extracted_results: list[dict[str, Any]] = []  # v13: Collect results without aborting
    browser = None
    context = None
    page = None

    try:
        # Extract parameters (handle both direct and nested params)
        params = action_spec.get("params", action_spec)
        
        # Extract URL with fallbacks
        url = params.get("url")
        if not url:
            # Legacy fallback: check top-level
            url = action_spec.get("url")
        
        if not url:
            logger.error("No URL provided in action_spec")
            return {
                "ok": False,
                "error": "missing_url",
                "detail": "No 'url' found in params or action_spec",
                "audit": audit
            }

        # Security check
        if not _is_domain_allowed(url, allowlist):
            logger.warning(f"Domain not allowed: {url}")
            return {
                "ok": False,
                "error": "domain_not_allowed",
                "domain": urlparse(url).netloc,
                "detail": f"Domain not in allowlist: {urlparse(url).netloc}",
                "audit": audit
            }

        # Launch browser
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                audit.append({
                    "action": "browser_launch",
                    "headless": headless,
                    "time": time.time()
                })
            except Exception as e:
                logger.exception(f"Failed to launch browser: {e}")
                return {
                    "ok": False,
                    "error": "browser_launch_failed",
                    "detail": str(e),
                    "audit": audit
                }

            # Navigate to URL
            try:
                logger.info(f"Navigating to: {url}")
                page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                audit.append({
                    "action": "goto",
                    "url": url,
                    "time": time.time()
                })
            except PlaywrightTimeoutError as te:
                logger.exception(f"Timeout navigating to {url}")
                return {
                    "ok": False,
                    "error": "navigation_timeout",
                    "detail": str(te),
                    "url": url,
                    "audit": audit
                }
            except Exception as e:
                logger.exception(f"Navigation error: {e}")
                return {
                    "ok": False,
                    "error": "navigation_failed",
                    "detail": str(e),
                    "url": url,
                    "audit": audit
                }

            # Execute steps
            steps = params.get("steps", [])
            
            if not isinstance(steps, list):
                logger.error(f"Steps must be a list, got {type(steps)}")
                return {
                    "ok": False,
                    "error": "invalid_steps",
                    "detail": f"Expected list, got {type(steps).__name__}",
                    "audit": audit
                }
            
            action_count = 0
            for idx, step in enumerate(steps):
                # Rate limiting
                if action_count >= MAX_ACTIONS_PER_SESSION:
                    logger.error(f"Action limit exceeded: {MAX_ACTIONS_PER_SESSION}")
                    return {
                        "ok": False,
                        "error": "action_limit_exceeded",
                        "detail": f"Maximum {MAX_ACTIONS_PER_SESSION} actions per session",
                        "audit": audit
                    }
                action_count += 1

                # Validate step
                if not isinstance(step, dict):
                    logger.error(f"Step {idx} is not a dict: {type(step)}")
                    continue

                step_type = step.get("type")
                selector = step.get("selector")
                
                try:
                    # Execute step based on type
                    if step_type == "wait_for":
                        if not selector:
                            logger.warning("wait_for step missing selector")
                            continue
                        
                        step_timeout = step.get("timeout", timeout)
                        page.wait_for_selector(selector, timeout=step_timeout)
                        audit.append({
                            "action": "wait_for",
                            "selector": selector,
                            "time": time.time()
                        })
                        
                    elif step_type == "click":
                        if not selector:
                            logger.warning("click step missing selector")
                            continue
                        
                        page.click(selector, timeout=timeout)
                        audit.append({
                            "action": "click",
                            "selector": selector,
                            "time": time.time()
                        })
                        
                    elif step_type == "type":
                        if not selector:
                            logger.warning("type step missing selector")
                            continue
                        
                        text = step.get("text", "")
                        try:
                            # Try fill first (faster)
                            page.fill(selector, text, timeout=timeout)
                        except Exception:
                            # Fallback to type (slower but more reliable)
                            page.type(selector, text, timeout=timeout)
                        
                        audit.append({
                            "action": "type",
                            "selector": selector,
                            "length": len(text),
                            "time": time.time()
                        })
                        
                    elif step_type == "screenshot":
                        path = step.get("path", f"screenshot_{int(time.time())}.png")
                        page.screenshot(path=path, full_page=step.get("full_page", False))
                        audit.append({
                            "action": "screenshot",
                            "path": path,
                            "time": time.time()
                        })
                        
                    elif step_type == "extract_text":
                        if selector:
                            # Extract from specific element
                            try:
                                text = page.locator(selector).inner_text(timeout=timeout)
                            except Exception as e:
                                logger.warning(f"Failed to extract text from {selector}: {e}")
                                text = ""
                        else:
                            # Extract entire page content
                            text = page.content()
                        
                        audit.append({
                            "action": "extract_text",
                            "selector": selector or "full_page",
                            "length": len(text),
                            "time": time.time()
                        })
                        
                        # v13: Collect result instead of aborting remaining steps
                        extracted_results.append({
                            "step_index": idx,
                            "selector": selector or "full_page",
                            "text": text
                        })
                    
                    elif step_type == "scroll":
                        to = step.get("to", "bottom")
                        logger.info(f"Scrolling to {to}")
                        if to == "bottom":
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        elif to == "top":
                            page.evaluate("window.scrollTo(0, 0)")
                        else:
                            # Try pixel value
                            try:
                                page.evaluate(f"window.scrollTo(0, {int(to)})")
                            except Exception:
                                logger.warning(f"Invalid scroll target: {to}")
                        
                        audit.append({
                            "action": "scroll",
                            "to": to,
                            "time": time.time()
                        })

                    elif step_type == "copy":
                         # Extract text to "clipboard" (result)
                         if not selector:
                             logger.warning("copy step missing selector")
                             continue
                         
                         text = page.locator(selector).inner_text(timeout=timeout)
                         logger.info(f"Copied {len(text)} chars from {selector}")
                         
                         audit.append({
                             "action": "copy",
                             "selector": selector,
                             "length": len(text),
                             "time": time.time()
                         })
                         
                         # v13B: Collect result instead of aborting remaining steps
                         extracted_results.append({
                             "step_index": idx,
                             "selector": selector,
                             "text": text
                         })

                    elif step_type == "paste":
                         # Paste text into input
                         if not selector:
                             logger.warning("paste step missing selector")
                             continue
                             
                         text = step.get("text", "")
                         logger.info(f"Pasting {len(text)} chars to {selector}")
                         page.fill(selector, text, timeout=timeout)
                         
                         audit.append({
                             "action": "paste",
                             "selector": selector,
                             "length": len(text),
                             "time": time.time()
                         })
                    
                    elif step_type == "evaluate":
                        # Execute JavaScript
                        script = step.get("script")
                        if not script:
                            logger.warning("evaluate step missing script")
                            continue
                        
                        result = page.evaluate(script)
                        audit.append({
                            "action": "evaluate",
                            "script_length": len(script),
                            "time": time.time()
                        })
                        
                        # v13B: Collect result instead of aborting remaining steps
                        if step.get("return_result"):
                            extracted_results.append({
                                "step_index": idx,
                                "type": "evaluate",
                                "result": result
                            })
                    
                    else:
                        logger.warning(f"Unknown step type: {step_type}")
                        audit.append({
                            "action": "unknown",
                            "type": step_type,
                            "time": time.time()
                        })
                        
                except PlaywrightTimeoutError as te:
                    logger.exception(f"Timeout during step {idx} ({step_type})")
                    return {
                        "ok": False,
                        "error": "step_timeout",
                        "detail": str(te),
                        "step": step,
                        "step_index": idx,
                        "audit": audit
                    }
                except Exception as exc:
                    logger.exception(f"Exception during step {idx} ({step_type})")
                    return {
                        "ok": False,
                        "error": "step_exception",
                        "detail": str(exc),
                        "step": step,
                        "step_index": idx,
                        "audit": audit
                    }

            # All steps completed successfully
            audit.append({
                "action": "completed",
                "steps_executed": len(steps),
                "time": time.time()
            })
            
            # v13: Return extracted text if any, otherwise generic success
            final_result = extracted_results[-1]["text"] if extracted_results else "all_actions_completed"
            return {
                "ok": True,
                "result": final_result,
                "extracted": extracted_results if extracted_results else None,
                "steps_executed": len(steps),
                "audit": audit
            }
            
    except Exception as exc:
        logger.exception(f"Browser action failed: {exc}")
        return {
            "ok": False,
            "error": "executor_failure",
            "detail": str(exc),
            "audit": audit
        }
    finally:
        # Cleanup
        try:
            if page:
                page.close()
        except Exception as e:
            logger.debug(f"Error closing page: {e}")
        
        try:
            if context:
                context.close()
        except Exception as e:
            logger.debug(f"Error closing context: {e}")
        
        try:
            if browser:
                browser.close()
        except Exception as e:
            logger.debug(f"Error closing browser: {e}")


def add_allowed_domain(domain: str):
    """Add a domain to the allowlist."""
    DOMAIN_ALLOWLIST.add(domain.lower())
    logger.info(f"Added domain to allowlist: {domain}")


def remove_allowed_domain(domain: str):
    """Remove a domain from the allowlist."""
    DOMAIN_ALLOWLIST.discard(domain.lower())
    logger.info(f"Removed domain from allowlist: {domain}")


def get_allowed_domains() -> set[str]:
    """Get current allowlist."""
    return DOMAIN_ALLOWLIST.copy()

"""The initiatives that involve somebody else.

Mail and forum reading, and what she does about them. Reading is cheap and
answering is not, so the classification is separate from the draft and the
draft is separate from sending: an initiative that decides to reply still has
to pass the same authority gate any other outward action does.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from core.runtime.service_access import (
    optional_service,
)


class _StartsSomethingSocial:
    """Lifted whole from AutonomousInitiativeLoop; see autonomous_initiative_loop.py."""

    async def _execute_email_adapter(
        self, payload: dict[str, Any], cap_engine: Any = None
    ) -> dict[str, Any]:
        cap_engine = cap_engine or optional_service("capability_engine", default=None)
        if cap_engine is not None and hasattr(cap_engine, "execute"):
            return await cap_engine.execute(
                "email_adapter",
                payload,
                {
                    "origin": "autonomous_initiative_loop",
                    "intent_source": "autonomous_initiative_loop",
                    "objective": "Read connected email state for bounded autonomous triage.",
                    "user_facing": False,
                },
            )

        from core.skills.email_adapter import EmailAdapterSkill, EmailInput

        skill = EmailAdapterSkill()
        return await skill.safe_execute(EmailInput(**payload), {})

    async def _execute_reddit_adapter(
        self, payload: dict[str, Any], cap_engine: Any = None
    ) -> dict[str, Any]:
        cap_engine = cap_engine or optional_service("capability_engine", default=None)
        if cap_engine is not None and hasattr(cap_engine, "execute"):
            return await cap_engine.execute(
                "reddit_adapter",
                payload,
                {
                    "origin": "autonomous_initiative_loop",
                    "intent_source": "autonomous_initiative_loop",
                    "objective": "Read connected Reddit state for bounded autonomous social awareness.",
                    "user_facing": False,
                },
            )

        from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

        skill = RedditAdapterSkill()
        return await skill.safe_execute(
            RedditInput(**payload),
            {
                "origin": "autonomous_initiative_loop",
                "intent_source": "autonomous_initiative_loop",
                "objective": (
                    "Read public Reddit state for bounded autonomous social awareness."
                ),
                "user_facing": False,
            },
        )

    def _source_kind_note(self, url: str) -> dict[str, Any]:
        """What kind of source this is, for material with no single claim."""
        try:
            from core.knowledge.source_comprehension import classify_source

            kind, caveat = classify_source(url)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return {}
        return {"source_kind": kind, "source_caveat": caveat} if caveat else {}

    async def _remember_social_observation(
        self,
        text: str,
        *,
        tags: list[str] | None = None,
        importance: float = 0.45,
        comprehension: dict[str, Any] | None = None,
    ) -> None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .autonomous_initiative_loop import (
            _record_initiative_degradation,
            logger,
        )

        text = " ".join(str(text or "").strip().split())
        if not text:
            return
        try:
            memory = optional_service("memory_manager", default=None)
            if memory and hasattr(memory, "store"):
                store_kwargs: dict[str, Any] = {
                    "importance": importance,
                    "tags": tags or ["autonomy", "social"],
                }
                if comprehension:
                    # Kept BESIDE the sentence, so a later reader can see the
                    # claim, the source's kind, and where it landed against
                    # what she holds — not just that a page went by.
                    store_kwargs["metadata"] = {"comprehension": comprehension}
                try:
                    await memory.store(text[:1800], **store_kwargs)
                except TypeError:
                    # An older store() without metadata support still gets the
                    # comprehended sentence, which is the important half.
                    await memory.store(
                        text[:1800],
                        importance=importance,
                        tags=tags or ["autonomy", "social"],
                    )
        except (RuntimeError, AttributeError, TypeError) as exc:
            _record_initiative_degradation(
                exc,
                action="kept social observation transient after memory write failed",
                severity="warning",
                extra={"tags": tags or ["autonomy", "social"], "importance": importance},
            )
            logger.debug("Social observation memory write failed: %s", exc)

    def _social_due_actions(self, now: float) -> dict[str, bool]:
        return {
            "email": now - float(self._last_email_check or 0.0) > 900.0,
            "reddit": now - float(self._last_reddit_check or 0.0) > 2700.0,
        }

    @staticmethod
    def _email_preview(body: str, *, limit: int = 240) -> str:
        clean = " ".join(str(body or "").replace("\r", "\n").split())
        return clean[:limit].strip()

    @staticmethod
    def _classify_email_message(
        message: dict[str, Any], read_result: dict[str, Any]
    ) -> dict[str, Any]:
        from .autonomous_initiative_loop import (
            AutonomousInitiativeLoop,
        )

        sender = str(read_result.get("from") or message.get("from") or "Unknown")
        subject = str(read_result.get("subject") or message.get("subject") or "(no subject)")
        body = str(read_result.get("body") or "")
        combined = f"{sender} {subject} {body}".lower()
        is_auto = bool(read_result.get("is_auto_reply"))
        from_owner = "youngbryan97" in sender.lower() or "bryan" in sender.lower()
        urgent_markers = (
            "urgent",
            "asap",
            "deadline",
            "action required",
            "please respond",
            "please reply",
            "follow up",
            "can you",
            "could you",
            "would you",
            "question",
            "?",
        )
        noise_markers = (
            "unsubscribe",
            "promotion",
            "newsletter",
            "no-reply",
            "noreply",
            "receipt",
            "security alert",
            "verification code",
        )
        urgent = any(marker in combined for marker in urgent_markers)
        likely_noise = any(marker in combined for marker in noise_markers)
        if is_auto:
            action = "skip_auto_reply"
        elif from_owner or urgent:
            action = "hold_for_reply_draft"
        elif likely_noise:
            action = "archive_candidate"
        else:
            action = "watch"
        return {
            "uid": str(read_result.get("uid") or message.get("uid") or ""),
            "from": sender,
            "subject": subject,
            "preview": AutonomousInitiativeLoop._email_preview(body),
            "is_auto_reply": is_auto,
            "from_owner": from_owner,
            "urgent": urgent,
            "likely_noise": likely_noise,
            "action": action,
        }

    @staticmethod
    def _draft_email_response(triage: dict[str, Any]) -> str:
        if triage.get("action") != "hold_for_reply_draft":
            return ""
        subject = str(triage.get("subject") or "your note")
        if triage.get("from_owner"):
            return (
                f"I read your email about {subject}. I am holding the details in context "
                "and can follow up once I have a concrete update instead of firing off a shallow reply."
            )
        return (
            f"Thanks for the note about {subject}. I read it and want to answer carefully; "
            "I will follow up with the specific next step once I have checked the relevant context."
        )

    async def _social_interaction_loop(self):
        """Autonomous social presence: check email and Reddit."""
        from .autonomous_initiative_loop import (
            _passive_social_allowed,
            _record_initiative_degradation,
            logger,
        )

        while self.running:
            try:
                if not _passive_social_allowed(self.orchestrator):
                    await asyncio.sleep(60)
                    continue

                now = time.time()

                due = self._social_due_actions(now)

                # Check Email every 15 minutes
                if due["email"]:
                    await self._check_email_initiative()
                    self._last_email_check = time.time()

                # Check Reddit every 45 minutes
                if due["reddit"]:
                    await self._check_reddit_initiative()
                    self._last_reddit_check = time.time()

            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_initiative_degradation(
                    e,
                    action="continued social interaction loop after transient adapter failure",
                    severity="warning",
                )
                logger.debug("Social interaction loop error: %s", e)

            await asyncio.sleep(60)

    async def _check_email_initiative(self):
        """Check for unread emails and potentially initiate a response."""
        from .autonomous_initiative_loop import (
            _record_initiative_degradation,
            logger,
        )

        logger.info("📧 Checking email for autonomous initiatives...")
        try:
            cap_engine = optional_service("capability_engine", default=None)
            result = await self._execute_email_adapter(
                {"mode": "check", "limit": 5}, cap_engine=cap_engine
            )
            if not result.get("ok"):
                return

            unread_count = result.get("unread", 0)
            if unread_count > 0:
                self._emit_feed(
                    "Email Update",
                    f"I have {unread_count} unread emails. Scanning for anything urgent.",
                    category="Social",
                )

            triaged: list[dict[str, Any]] = []
            now = time.time()
            for msg in list(result.get("messages") or [])[:3]:
                uid = str(msg.get("uid") or "")
                if not uid:
                    continue
                # Avoid hot-looping the same unread item every social tick, while
                # still rechecking it later if it remains relevant.
                if now - float(self._recent_email_uids.get(uid, 0.0) or 0.0) < 1800.0:
                    continue
                self._recent_email_uids[uid] = now
                read_result = await self._execute_email_adapter(
                    {"mode": "read", "uid": uid}, cap_engine=cap_engine
                )
                if not read_result.get("ok"):
                    self._emit_feed(
                        "Email Triage",
                        f"Could not read UID {uid}: {read_result.get('error', 'unknown error')}",
                        category="Social",
                    )
                    continue
                triage = self._classify_email_message(msg, read_result)
                triage["draft_reply"] = self._draft_email_response(triage)
                triaged.append(triage)

                action_label = str(triage.get("action") or "watch")
                preview = str(triage.get("preview") or "No readable body preview.")
                self._emit_feed(
                    "Email Triage",
                    (
                        f"{action_label}: {triage.get('subject')} from {triage.get('from')}. "
                        f"Preview: {preview[:180]}"
                    ),
                    category="Social",
                )
                if triage.get("draft_reply"):
                    self._emit_feed(
                        "Email Draft",
                        str(triage["draft_reply"])[:320],
                        category="Social",
                    )
                await self._remember_social_observation(
                    (
                        f"Email triage: {action_label} | from={triage.get('from')} | "
                        f"subject={triage.get('subject')} | preview={preview[:280]}"
                    ),
                    tags=["autonomy", "email", action_label],
                    importance=0.65 if triage.get("from_owner") or triage.get("urgent") else 0.45,
                )

            attention_items = [
                item for item in triaged if item.get("action") == "hold_for_reply_draft"
            ]
            if attention_items:
                first = attention_items[0]
                self._queue_visible_update(
                    f"I read an unread email from {first.get('from')} about '{first.get('subject')}' and drafted a cautious reply, but I am not auto-sending it."
                )
            elif unread_count > 0 and triaged:
                self._emit_feed(
                    "Email Update",
                    f"Email triage complete: {len(triaged)} unread message(s) read, no safe autonomous reply needed.",
                    category="Social",
                )

        except (OSError, ConnectionError, TimeoutError) as e:
            _record_initiative_degradation(
                e,
                action="skipped email initiative tick after mail adapter transient failure",
                severity="warning",
            )

    async def _check_reddit_initiative(self):
        """Browse Reddit and potentially find something to engage with."""
        from .autonomous_initiative_loop import (
            _record_initiative_degradation,
            logger,
        )

        logger.info("📱 Browsing Reddit for autonomous initiatives...")
        try:
            cap_engine = optional_service("capability_engine", default=None)
            # Public reading is independent from account authentication and
            # always runs first. Boot stabilization must never begin with a
            # headless password flow.
            subreddits = ["askreddit", "nosleep", "technology", "philosophy", "futurology"]
            import random

            sub = random.choice(subreddits)

            result = await self._execute_reddit_adapter(
                {"mode": "browse", "subreddit": sub, "limit": 5},
                cap_engine=cap_engine,
            )
            if result.get("ok") and result.get("posts"):
                posts = result.get("posts")
                top_post = posts[0]
                self._emit_feed(
                    "Reddit Browse",
                    f"Browsing r/{sub}. Found an interesting thread: '{top_post.get('title')}'",
                    category="Social",
                )
                digest_lines = []
                for post in posts[:3]:
                    title = str(post.get("title") or "").strip()
                    if not title:
                        continue
                    digest_lines.append(
                        f"{title} (score={post.get('score', '0')}, comments={post.get('comments', '0')})"
                    )
                if digest_lines:
                    # A browse digest is headlines, not an article, so there
                    # is no claim to comprehend — but what KIND of thing these
                    # are still matters, and remembering them without it makes
                    # a vote count look like evidence.
                    await self._remember_social_observation(
                        f"Reddit browse r/{sub}: " + " | ".join(digest_lines),
                        tags=["autonomy", "reddit", f"r/{sub}"],
                        importance=0.42,
                        comprehension=self._source_kind_note(
                            f"https://www.reddit.com/r/{sub}"
                        ),
                    )

                url = str(top_post.get("url") or "").strip()
                if url and (now := time.time()):
                    if url.startswith("/"):
                        url = f"https://www.reddit.com{url}"
                    if now - float(self._recent_reddit_urls.get(url, 0.0) or 0.0) >= 3600.0:
                        self._recent_reddit_urls[url] = now
                        read_result = await self._execute_reddit_adapter(
                            {"mode": "read_post", "url": url},
                            cap_engine=cap_engine,
                        )
                        if read_result.get("ok"):
                            content = " ".join(str(read_result.get("content") or "").split())
                            self._emit_feed(
                                "Reddit Read",
                                f"Read top r/{sub} thread '{top_post.get('title')}'. Excerpt: {content[:260]}",
                                category="Social",
                            )
                            # LIVE DEFECT, 2026-08-03. This stored the page —
                            # navigation chrome included — as a 500-character
                            # excerpt tagged "logged". The event that she read
                            # something was recorded; what she made of it was
                            # not, so the reading could tell her nothing
                            # afterwards.
                            line, comprehension = self._comprehend_reading(
                                url=url,
                                title=str(top_post.get("title") or ""),
                                text=content,
                                prefix=f"Reddit read r/{sub}:",
                            )
                            await self._remember_social_observation(
                                line,
                                tags=["autonomy", "reddit", "read_post", f"r/{sub}"],
                                importance=0.5,
                                comprehension=comprehension,
                            )
                            await self._form_opinion_from_reading(comprehension)

            provider = (
                result.get("provider")
                if isinstance(result.get("provider"), dict)
                else {}
            )
            provider_state = str(provider.get("state") or "")
            if provider_state in {"session_unverified", "session_valid"}:
                inbox = await self._execute_reddit_adapter(
                    {"mode": "check_inbox"}, cap_engine=cap_engine
                )
                if inbox.get("ok") and "unread" in str(inbox.get("content", "")).lower():
                    self._emit_feed(
                        "Reddit Update",
                        "I have new Reddit notifications. Checking for replies to my comments.",
                        category="Social",
                    )
                elif inbox.get("status") == "login_unavailable":
                    self._emit_feed(
                        "Reddit Inbox",
                        "The saved session needs recovery; public browsing remains active.",
                        category="Social",
                    )
            elif provider_state:
                logger.info(
                    "Reddit authenticated inbox deferred in provider state %s; "
                    "public browsing remains active.",
                    provider_state,
                )

        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as e:
            _record_initiative_degradation(
                e,
                action="skipped reddit initiative tick after adapter failure",
                severity="warning",
            )

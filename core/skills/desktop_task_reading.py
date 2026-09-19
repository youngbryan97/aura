"""What the objective asks for, and what is pulled out of it.

Lifted whole out of `desktop_task`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
import time


class _ReadsTheObjective:
    """Lifted whole out of DesktopTaskSkill; see desktop_task.py."""

    @staticmethod
    def _extract_folder_name(objective: str) -> str:
        text = str(objective or "")
        # Quoted names may contain possessive apostrophes ("Aura's
        # Journal"); the close-quote is the one followed by a boundary,
        # not the first internal apostrophe (which truncated the name to
        # "Aura" and broke the journal demo's folder).
        match = re.search(
            r"\b(?:folder|directory)\b[^.\n]{0,80}?\b(?:named|called|titled)\s+"
            r"(?:'((?:[^']|'(?=\w))+)'(?=[\s.,;)]|$)"
            r"|\"([^\"]+)\""
            r"|([^.,;\n]+?)(?=\s+(?:in|inside|under|on)\s+(?:my\s+)?\w|[.,;\n]|$))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            name = str(match.group(1) or match.group(2) or match.group(3) or "").strip()
            return name.strip("'\"., ")[:100]
        # Name-first phrasing: "the 'Aura's Journal' folder" — quoted name
        # immediately before the word folder/directory.
        name_first = re.search(
            r"(?:'((?:[^']|'(?=\w))+)'|\"([^\"]+)\")\s+(?:folder|directory)\b",
            text,
            flags=re.IGNORECASE,
        )
        if name_first:
            name = str(name_first.group(1) or name_first.group(2) or "").strip()
            if name:
                return name.strip("'\"")[:100]
        return f"Aura Desktop Task {int(time.time())}"

    @staticmethod
    def _extract_root_hint(objective: str) -> str:
        """Honor the user's stated artifact root.

        Live proof rounds wrote to the Desktop default while the user
        said 'in my Documents folder' — parameter fidelity is general
        capability, not pattern-matching: extract what was actually
        asked.
        """
        lowered = str(objective or "").lower()
        for token, root in (
            ("documents folder", "~/Documents"),
            ("my documents", "~/Documents"),
            ("documents directory", "~/Documents"),
            ("downloads folder", "~/Downloads"),
            ("my downloads", "~/Downloads"),
            ("desktop folder", "~/Desktop"),
            ("my desktop", "~/Desktop"),
        ):
            if token in lowered:
                return root
        return ""

    @staticmethod
    def _extract_explicit_filename(objective: str) -> str:
        """The user's stated filename wins over generated stems."""
        match = re.search(
            r"\bfile\b[^.\n]{0,60}?\b(?:named|called|titled)\s+"
            r"['\"]?([\w][\w .-]{0,80}?\.(?:txt|md|markdown|rtf|text))['\"]?",
            str(objective or ""),
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        # Manner phrases say WHERE to look, not WHAT to look for, and they are
        # removed before the topic patterns run — otherwise "on the internet"
        # becomes the topic and "orcas online" searches for a wireless ISP.
        from .desktop_task import (
            DesktopTaskSkill,
        )

        text = DesktopTaskSkill._SEARCH_MANNER_ANYWHERE_RE.sub(
            " ", str(objective or "")
        )
        text = " ".join(text.split()).strip()
        count_word = r"(?:\d+|one|two|three|four|five)"
        patterns = (
            rf"\bfind\s+(?:me\s+)?(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            rf"\b(?:summari[sz]e|write\s+(?:a\s+)?summary\s+of)\s+(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\b(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\bsearch\s+(?:for\s+)?([^.;\n]+)",
            r"\blook\s+up\s+([^.;\n]+)",
            r"\bgoogle\s+([^.;\n]+)",
            r"\bopen\s+(?:a\s+)?(?:browser\s+)?tab\s+(?:on\s+google\s+)?(?:for\s+)?([^.;\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" ,")
                if query:
                    if re.match(
                        r"^(?:doc|docs|document|drive|sheet|sheets|slide|slides|chrome|safari|browser)\b",
                        query,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    if query.lower() in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                        # Resolve the coreference pronoun to preceding topic in context
                        m = re.search(r"\b(?:read|find|search)\s+(?:about|on|for)\s+([^.;\n,]+)", text, flags=re.IGNORECASE)
                        if m:
                            candidate = m.group(1).strip(" ,")
                            if candidate.lower() not in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                                    return candidate[:240]
                    else:
                        return DesktopTaskSkill._strip_search_manner(query)[:240]
        if "news" in text.lower():
            return DesktopTaskSkill._strip_search_manner(text)[:240]
        return ""

    @staticmethod
    def _objective_requests_recent_sources(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:recent|latest|current|newly published|new reporting)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _objective_requests_source_reading(cls, objective: str) -> bool:
        if cls._objective_requests_authored_synthesis(objective):
            return True
        return bool(
            re.search(
                r"\b(?:read|review|study|inspect|compare|evaluate|assess|look through)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _objective_requests_research_document(objective: str) -> bool:
        # Classify the PROSE, not the paths in it.
        #
        # LIVE, 2026-08-10: "count how many .py files are in
        # /Users/bryan/.aura/live-source/core/introspection, then write that
        # number ... into ~/Documents/aura_probe_count.txt" was classified as a
        # research-document objective, so completion required research SOURCES
        # and the turn reported "semantic completion incomplete:
        # requested_source_count_found" over a filesystem task that had
        # succeeded.
        #
        # The marker was "source", matched inside "live-source". A path is a
        # name, not prose, and every marker here is a substring test over
        # whatever the user happened to type — so any objective naming a path
        # with "report", "news", "article" or "source" in it inherits a
        # contract about citations.
        from .desktop_task import (
            _PATHS_IN_TEXT_RE,
            mentions_object_class,
        )

        lowered = _PATHS_IN_TEXT_RE.sub(" ", str(objective or "")).lower()
        has_source_markers = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "report",
                "reports",
            )
        )
        visual_reference_only = (
            mentions_object_class(lowered, "image") and not has_source_markers
        )
        if visual_reference_only:
            return False
        wants_research = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "look up",
                "search",
                "find",
            )
        )
        wants_written_output = any(
            marker in lowered
            for marker in (
                "summarize",
                "summary",
                "write",
                "document",
                "doc",
                "essay",
                "report",
                "note",
                "pdf",
                "type",
            )
        )
        return wants_research and wants_written_output

    @staticmethod
    def _extract_image_query(objective: str) -> str:
        from .desktop_task import (
            _VISUAL_ASSET_RE,
            extract_object_description,
        )

        text = str(objective or "").strip()
        described_object = extract_object_description(
            text,
            "image",
            action_phrases=("find", "search", "look up", "get", "download", "fetch"),
        )
        patterns = (
            rf"\b{_VISUAL_ASSET_RE}\s+of\s+([^.;\n]+)",
            rf"\b(?:find|search|look\s+up)\s+(?:an?\s+)?{_VISUAL_ASSET_RE}\s+(?:of\s+)?([^.;\n]+)",
            rf"\b(?:find|search(?:\s+for)?|look\s+up|get)\b\s+"
            rf"(?:an?\s+|some\s+)?([^.;\n]{{2,120}}?)\s+{_VISUAL_ASSET_RE}\b",
            rf"\b([^.;\n]{{2,120}}?)\s+{_VISUAL_ASSET_RE}\b",
        )
        candidates = [described_object]
        candidates.extend(
            match.group(1)
            for pattern in patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE)) is not None
        )
        for candidate in candidates:
            if candidate:
                query = re.sub(r"\b(?:and|then|also)\b.*$", "", candidate, flags=re.IGNORECASE)
                query = re.sub(
                    r"\bfrom\s+(?:online|the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
                query = re.sub(
                    r"\b(?:online|on\s+the\s+(?:internet|web)|from\s+the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
                query = re.sub(r"^(?:a|an|the)\s+", "", query.strip(" ,"), flags=re.IGNORECASE)
                query = query.strip(" ,")
                if query:
                    return query[:240]
        return ""

    @staticmethod
    def _extract_apps(objective: str) -> list[str]:
        from .desktop_task import (
            _DESKTOP_TASK_RECOVERABLE_ERRORS,
            _QUOTED_SPAN_RE,
            _without_filenames,
            logger,
        )

        text = str(objective or "").lower()
        apps: list[str] = []
        app_markers = {
            "notes": "Notes",
            "calculator": "Calculator",
            "finder": "Finder",
            "preview": "Preview",
            "safari": "Safari",
            "chrome": "Google Chrome",
            "browser": "Safari",
            "textedit": "TextEdit",
            "pages": "Pages",
            "microsoft word": "Microsoft Word",
            "ms word": "Microsoft Word",
        }
        # Word-boundary matching: the bare substring scan opened
        # Microsoft Word because the objective said "in your own words"
        # — a fatal launch on Macs without Word. Apps must be NAMED.
        #
        # A word boundary is not enough on its own, because "." is one:
        # "add a line to the end of notes.txt" matched \bnotes\b and routed a
        # file edit into the Notes app, where the file on disk was never
        # touched. A name carrying a file extension is a FILENAME — same
        # failure as "in your own words", one punctuation mark along.
        named = _without_filenames(text)
        for marker, app in app_markers.items():
            if re.search(rf"\b{re.escape(marker)}\b", named) and app not in apps:
                if marker == "browser" and "chrome" in text:
                    continue
                apps.append(app)

        # ...and then everything else that is actually on this machine.
        #
        # The table above is eleven names, so "open Reminders" named no app at
        # all and the work fell through to a text file on disk. The machine
        # already knows what is installed; an app it has is an app she can be
        # asked for, without anyone adding a row.
        #
        # A NAMED app needs a verb, and must not be inside quotes.
        #
        # The eleven-name table above could match loosely because those names
        # rarely appear by accident. Ninety-one cannot: "a new folder called
        # 'Aura's Journal'" contains two installed app names, and matching
        # them opened two applications to make a folder. This is the same
        # failure as "in your own words" launching Microsoft Word, one list
        # further along — so the general form carries the general guard.
        quoted = " ".join(_QUOTED_SPAN_RE.findall(text))
        try:
            from core.perception.app_dictionary import installed_apps

            for name in installed_apps():
                lowered_name = name.lower()
                if name in apps or len(name) < 4:
                    continue
                if re.search(rf"\b{re.escape(lowered_name)}\b", quoted):
                    continue  # It is the name of a thing, not a request.
                if re.search(
                    rf"\b(?:open|launch|start|run|use|using|switch\s+to|"
                    rf"in|into|inside|with|via|from)\s+"
                    rf"(?:up\s+)?(?:my\s+|the\s+|a\s+)?"
                    rf"{re.escape(lowered_name)}\b",
                    named,
                ):
                    apps.append(name)
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            logger.debug("Could not enumerate installed applications: %s", exc)
        return apps[:4]

    @staticmethod
    def _objective_requests_opinion(objective: str) -> bool:
        """Does the objective ask Aura for her own view, not just a summary?"""
        lowered = str(objective or "").lower()
        return bool(
            re.search(r"\b(?:your|my|her|own)\s+(?:opinion|view|views|take|thoughts|assessment|perspective|stance)\b", lowered)
            or re.search(r"\bform\s+(?:your|an?|my)\s+(?:own\s+)?opinion\b", lowered)
            or "what you think" in lowered
            or "what do you think" in lowered
        )

    @staticmethod
    def _extract_declared_document_content(text: str) -> str:
        """Pull authored content out of model preambles like "write this content:"."""
        from .desktop_task import (
            DesktopTaskSkill,
        )

        value = str(text or "").strip()
        if not value:
            return ""
        patterns = (
            r"(?:following\s+)?content\s*[:：]\s*[-–—]*\s*(.+)$",
            r"\bhere\s+(?:it\s+is|is\s+the\s+(?:paragraph|note|document|content))\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:note|paragraph|document)\s+(?:text|body)\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:write|type|insert)\s+(?:this\s+)?(?:text|paragraph|content)\s*[:：]\s*[-–—]*\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            body = str(match.group(1) or "").strip(" \n\r\t-–—")
            if body:
                return DesktopTaskSkill._strip_artifact_action_tail(body)[:9000]
        return ""

    @classmethod
    def _objective_supplies_literal_document_body(cls, objective: str) -> bool:
        return bool(cls._literal_document_body_from_objective(objective))

    @staticmethod
    def _objective_requests_timestamp(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:timestamp|time stamp|date stamp|current date|current time|date and time|dated)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _objective_needs_authored_content(cls, objective: str) -> bool:
        """True only when SHE has to supply the words.

        "Create a note from the clipboard" names its own content source; writing
        something new there would ignore the request. "Write a note with three
        sentences about orcas" does not, and that is the case where the
        deterministic composer produced a note describing what a note should
        contain instead of containing it.
        """

        text = str(objective or "")
        if not text.strip():
            return False
        if cls._CONTENT_SOURCE_RE.search(text):
            return False
        if cls._objective_supplies_literal_document_body(text):
            return False
        return bool(
            cls._objective_requests_freeform_written_content(text)
            or cls._objective_requests_written_artifact(text)
        )

    @staticmethod
    def _extract_requested_writing_topic(objective: str) -> str:
        """Extract the subject of a requested note/document when possible."""
        from .desktop_task import (
            DesktopTaskSkill,
        )

        text = " ".join(str(objective or "").strip().split())
        if not text:
            return ""
        patterns = (
            # "a new note WITH THREE SENTENCES about humpback whales" — the
            # qualifier between the noun and "about" defeated this, so the
            # topic came back empty and the note was written about "the
            # requested subject". Measured live 2026-07-28.
            r"\b(?:write|draft|compose|type|create)\s+(?:me\s+)?(?:a\s+|an\s+)?"
            r"(?:new\s+|short\s+|full\s+|one\s+)?"
            r"(?:paragraph|note|document|essay|summary|report|journal\s+entry)"
            r"(?:\s+(?:with|containing|of|that\s+has)\s+[^.]{0,60}?)?"
            r"\s+(?:about|on|describing|explaining|covering)\s+(.+)$",
            r"\b(?:write|draft|compose|type)\s+(.+?)\s+(?:in|into|to)\s+(?:notes|google docs|docs|a note|the note)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            topic = match.group(1).strip(" .,:;?!\"'")
            # A following instruction is not part of the subject: "about
            # humpback whales. Actually do it" is about humpback whales.
            topic = re.split(r"(?<=[a-z])[.!?]\s+\S", topic)[0].strip(" .,:;?!\"'")
            topic = re.split(
                r"\b(?:and then|then|after that|also|export|save|create a folder|make a folder)\b",
                topic,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;?!\"'")
            # WHERE the writing goes is not WHAT it is about. "a note about
            # orcas in the Notes app" is about orcas; the destination rode
            # along and became the title "Orcas In The Notes App". Same shape
            # as "orcas online" searching for a wireless ISP — a trailing
            # adjunct read as part of the subject.
            topic = DesktopTaskSkill._DESTINATION_TAIL_RE.sub("", topic).strip(" .,:;?!\"'")
            if topic:
                return topic[:180]
        return ""

    @classmethod
    def _objective_requests_freeform_written_content(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        if cls._objective_requests_self_summary(objective) or cls._objective_requests_research_document(objective):
            return False
        return bool(
            re.search(
                # Any verb that introduces authored content, not a chosen few.
                #
                # The list held "make up" and not "make", so "make a file on my
                # Desktop with one sentence in it about what you did tonight"
                # fell through to the deterministic composer and the file held
                # "Notes on the requested subject: The requested subject is the
                # focus of this note." Correctly created, correctly saved, and
                # empty of content — for the third time, reached by a phrasing
                # nobody had listed.
                #
                # Verbs of production are a small closed class and the same one
                # every request uses; literary forms are open and there is
                # always another. LIVE 2026-08-26.
                r"\b(?:write|writing|written|draft|compose|type|create|make|made|put|"
                r"add|save|jot|record|tell|generate|produce|fill)\b.{0,80}?"
                # The form has to be used AS a form. Several of these words are
                # verbs in the same breath — "report the paths", "record the
                # session", "caption the photo" — and matching the bare word
                # read every one of them as a request for a document. LIVE
                # 2026-08-30: "put it into Notes ... and report the paths" was
                # authored as a report. A determiner in front, or a plural, is
                # what tells a noun from a verb here, and it is the same signal
                # a person reads.
                r"(?:\b(?:a|an|the|one|two|three|four|five|six|several|some|any|"
                r"another|each|this|that|my|your|his|her|our|their|\d+)\b"
                r"(?:\s+\w+){0,1}\s+)"
                r"(?:paragraph|sentence|sentences|line|lines|note|document|essay|"
                r"summary|report|journal entry|"
                # Creative forms are freeform writing too. Without them, "write
                # a haiku to a file called poem.txt" fell through to the
                # deterministic composer and the file held "Notes on the
                # requested subject: The requested subject is the focus of this
                # note" — the same empty template this predicate exists to
                # avoid, reached by a request nobody had listed.
                r"haiku|poem|poetry|verse|limerick|sonnet|song|lyric|story|"
                r"tale|joke|riddle|letter|speech|toast|eulogy|caption)\b",
                lowered,
            )
            # A bare plural is a noun on its own and needs no determiner in
            # front of it: "write sentences about the harbour".
            or cls._asks_for_plural_writing(lowered)
            # A subject with no named form is a writing request only behind a
            # verb that means writing. "about" is a preposition, not a literary
            # form, and counting it as one made every request with a subject in
            # it into a document: LIVE 2026-08-30, "play 2048 for me ... tell me
            # what you learn ABOUT the game" was authored as a note, failed to
            # author, and reported that no file had been created — to a request
            # that never mentioned a file.
            or re.search(
                r"\b(?:write|writing|written|draft|compose|pen|jot)\b"
                r"(?:\s+\w+){0,6}?\s+(?:about|describing|explaining)\b",
                lowered,
            )
        )

    @classmethod
    def _objective_requests_written_artifact(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        return bool(
            cls._objective_requests_freeform_written_content(objective)
            or cls._objective_requests_self_summary(objective)
            or cls._objective_requests_research_document(objective)
            or (
                re.search(r"\b(?:write|draft|compose|type|create|make|save|export)\b", lowered)
                and re.search(r"\b(?:note|notes|document|doc|file|pdf|paragraph|summary|report|journal)\b", lowered)
            )
        )

    @classmethod
    def _objective_requests_self_summary(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        direct_self_request = any(
            marker in lowered
            for marker in (
                "who you are",
                "what you are",
                "who or what you are",
                "about yourself",
                "describe yourself",
                "describing yourself",
                "self-summary",
                "self summary",
            )
        )
        if direct_self_request:
            return True
        if "in your own words" not in lowered:
            return False
        return bool(
            re.search(
                r"\b(?:you|yourself|aura)\b.{0,80}\b(?:are|identity|self|being|system|architecture)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _objective_requests_authored_synthesis(cls, objective: str) -> bool:
        return bool(cls._AUTHORED_SYNTHESIS_RE.search(str(objective or "")))







    @staticmethod
    def _objective_requests_observation_only(
        objective: str, previous_user_request: str = ""
    ) -> bool:
        """Delegates to the one shared definition.

        This was a literal-substring list, and it disagreed with the regex the
        router already used. "Can you see what's on the screen and tell me what
        you see?" said "the screen" where the list said "my screen", so a read
        escalated into os_automation and came back refused for having no
        observable acceptance contract. See looks_like_screen_observation.

        ``previous_user_request`` restores the antecedent for a follow-up. Live,
        Bryan asked for a screen read, was refused, and then said "Can you do it
        now?" and "Yes you can lol" — neither contains a screen noun, so both
        classified as not-an-observation and he was refused twice more. "It" was
        the screen read; the request was in the previous turn.
        """
        from .desktop_task import (
            looks_like_screen_observation,
        )

        if looks_like_screen_observation(objective):
            return True
        if not previous_user_request:
            return False
        from core.runtime.referential_continuation import effective_message

        resolved = effective_message(
            objective, previous_user_request=previous_user_request
        )
        return resolved.resolved and looks_like_screen_observation(resolved.text)

    @staticmethod
    def _objective_needs_general_os_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|focus|select|switch|close|"
                r"minimi[sz]e|maximi[sz]e|organize|click|press|type|paste|"
                r"enter|fill|choose)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _objective_requires_true_window_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize|tile|snap)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )


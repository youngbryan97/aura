"""Evidence provider — the ReAct layer that grounds reasoning in real data.

This is the difference between "prompt the model harder" and actually *knowing*.
Before the amplifier reasons about a verifiable question it should gather real
evidence by acting on the world it can read:

* repo / architecture / self-claim questions → search the actual codebase
  (ripgrep when available, else an in-process scan), then **read the matching
  source spans** (``path:line: code``). The model then answers *from* real spans,
  and the repo/citation truth engines check the answer *against* the same spans.
* factual / memory questions → recall from Aura's live memory facade.

So generation is conditioned on retrieved fact, not vibes, and verification has
something concrete to check. Pure read-only: it never mutates anything.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.paths import PROJECT_ROOT

logger = logging.getLogger("Aura.EvidenceProvider")

_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
_PATH_RE = re.compile(r"\b([A-Za-z_][\w./-]*\.(?:py|md|json|toml|yaml|yml))\b")
_CAMEL_OR_SNAKE = re.compile(r"^(?:[A-Z][a-z0-9]+){2,}$|^[a-z][a-z0-9]*_[a-z0-9_]+$")
_STOP = frozenset(
    "the and that this with from which would should there their about into your where when what "
    "does done make made work works using used your you aura have here only just like also both "
    "explain describe implement function module method class".split()
)
_REFERENCE_LEAD_RE = re.compile(
    r"^\s*(?:(?:can|could|would)\s+you\s+|please\s+)?"
    r"(?:explain|define|describe|summarize|outline|teach\s+me\s+about|what\s+is)\s+",
    re.IGNORECASE,
)
_REFERENCE_FORMAT_TAIL_RE = re.compile(
    r"\b(?:in\s+(?:one|a)\s+complete\s+response|in\s+detail|step\s+by\s+step|"
    r"for\s+me|include\s*:?).*$",
    re.IGNORECASE | re.DOTALL,
)
_REFERENCE_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*(?:-[A-Za-z0-9'’-]+)*")
_REFERENCE_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*[A-Z][A-Za-z0-9_.-]{1,48}\s+here\s*[.:!?-]+\s*",
)
_SKIP_DIRS = {".venv", "__pycache__", "node_modules", ".git", "archive", "dev_archive", ".mypy_cache",
              ".ruff_cache", ".pytest_cache", "dist", "build",
              # Worktrees are COPIES of this repository. Measured 2026-08-06:
              # 326,885 of the 334,558 scannable .py files were under
              # .claude/worktrees — 98% — so the 4,000-file scan cap was spent
              # almost entirely on duplicates and core/ was never reached. Two
              # separate harms: evidence gathering silently found nothing for
              # most symbols, and anything it did find could cite
              # `.claude/worktrees/<branch>/core/x.py:42`, which is a real line
              # in a file that is not the running code.
              ".claude", "worktrees",
              # Generated evidence bundles, not source. Aura citing her own
              # past output as evidence for a claim about her source is a
              # circularity nobody asked for.
              "artifacts",
              # Self-modification working copies: .aura_architect holds
              # `candidate/core/...` and `original/core/...` snapshots of real
              # modules. Citing one is citing a version of herself that was
              # considered and not adopted, at a line number that looks
              # authentic.
              ".aura_architect"}

#: A hit scoring at least this much is admitted no matter how full the
#: candidate bucket is: it means the file is named after the symbol, or the
#: line defines it. Those are exactly the hits a cap must never discard —
#: without this, asking about SubprocessGateway filled up on aura_main.py and
#: interface/server.py and never reached core/runtime/subprocess_gateway.py.
_STRONG_EVIDENCE_SCORE = 5.0

#: Which file is ABOUT the thing is a ranking problem with a standard answer,
#: and hand-weighting it kept trading one failure for another: counting terms
#: put aura_main.py first because the entrypoint mentions every subsystem
#: once, and counting density put a small file first because two lines of it
#: matched. BM25 is the published resolution of exactly that tension — term
#: frequency saturates, length is normalised, and a term appearing in most
#: files counts for little. k1 and b are the literature's defaults.
_BM25_K1 = 1.5
_BM25_B = 0.75

#: How much the file-level subject score moves a line's rank.
_SUBJECT_WEIGHT = 4.0

#: Lines quoted from any one file. An evidence pack of twelve lines from one
#: module answers nothing a single line would not.
_MAX_SPANS_PER_FILE = 2

#: Files opened per repo search. Listing paths is cheap and reading them is
#: not, so this bounds the reads while `_search_order` decides which ones.
_MAX_FILES_READ = 4000


@dataclass(frozen=True)
class ReferenceEvidence:
    """Corpus spans, and whether the corpus could be read at all."""

    spans: list[EvidenceSpan]
    retrieval_failed: bool


@dataclass
class EvidenceSpan:
    source: str          # "repo" | "memory"
    ref: str             # "core/x.py:42" or memory id
    text: str

    def render(self) -> str:
        return f"{self.ref}: {self.text}" if self.ref else self.text


def snake_case(term: str) -> str:
    """SubprocessGateway -> subprocess_gateway, so the file named after a
    symbol can be recognised as its home."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(term or "")).lower().strip("_")


def _filename_candidates(objective: str) -> set[str]:
    """Filenames the question itself spells out.

    A module is often named for the WORDS someone used rather than for any one
    of them: "how does the file write gateway work" points at
    core/runtime/file_write_gateway.py. Salient-term ranking drops "file" as
    too common, so the name could not be reassembled from the ranked terms —
    it has to come from the run of words as spoken.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(objective or ""))]
    out: set[str] = set()
    for width in range(2, 5):
        for start in range(len(words) - width + 1):
            out.add("_".join(words[start : start + width]))
    return out


@lru_cache(maxsize=1)
def _low_information_terms() -> frozenset[str]:
    """Words this project uses so often they cannot point at a file.

    "you have a lock ordering system ... what happens if two subsystems take
    locks in opposite order" reduces to six terms, of which "system",
    "happens" and "opposite" say nothing about WHICH file. Counting them as
    coverage put aura_main.py first for every such question, because the
    entrypoint names every subsystem there is.

    Measured against the project's own prose rather than a guessed stoplist: a
    word the documentation writes constantly is a word that does not
    discriminate. Read once per process from the docs already on disk, and an
    unreadable docs tree simply yields no stoplist.
    """
    counts: Counter[str] = Counter()
    try:
        sources = list((PROJECT_ROOT / "docs").rglob("*.md")) + list(
            PROJECT_ROOT.glob("*.md")
        )
        for path in sources[:200]:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
            counts.update(re.findall(r"[a-z][a-z-]{3,}", text))
    except (OSError, ValueError):
        return frozenset()
    if not counts:
        return frozenset()
    # The head of the distribution, not an arbitrary count: terms appearing
    # more often than the 200th most common one carry no file-level signal.
    ranked = counts.most_common(200)
    floor = ranked[-1][1]
    return frozenset(word for word, uses in counts.items() if uses >= floor)


def _salient_terms(objective: str, *, limit: int = 6) -> list[str]:
    """Pull the identifier-shaped, content-bearing terms worth searching for."""
    terms: list[str] = []
    # Explicit code-ish identifiers (CamelCase / snake_case) rank first.
    for m in _IDENT_RE.finditer(objective or ""):
        w = m.group(1)
        if w.lower() in _STOP:
            continue
        if _CAMEL_OR_SNAKE.match(w) and w not in terms:
            terms.append(w)
    if len(terms) < limit:
        for m in _IDENT_RE.finditer(objective or ""):
            w = m.group(1)
            if len(w) >= 5 and w.lower() not in _STOP and w not in terms:
                terms.append(w)
            if len(terms) >= limit:
                break
    return terms[:limit]


def reference_query_candidates(objective: str) -> list[str]:
    """Return bounded subject queries, excluding answer-format instructions.

    FTS over a long multipart request falls back to a huge any-term query. The
    subject usually appears in the opening clause; requested formatting and
    worked-example constraints do not identify the reference article. This
    extraction changes retrieval only. It does not tell the decoder how to
    answer.
    """

    raw = " ".join(str(objective or "").split()).strip()
    if not raw:
        return []
    # A visible speaker label is conversation routing metadata, not the topic.
    # Live desktop requests commonly begin ``ChatGPT here.`` so Aura knows who
    # is testing her.  Splitting on the first sentence before removing that
    # label searched the local corpus for ``ChatGPT here`` and discarded the
    # actual Dijkstra sentence.  Strip only the narrow self-identification
    # form; ordinary subjects that happen to contain "here" are untouched.
    raw = _REFERENCE_SPEAKER_PREFIX_RE.sub("", raw, count=1).strip()
    if not raw:
        return []
    opening = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0]
    opening = _REFERENCE_LEAD_RE.sub("", opening).strip(" .?!,:;")
    opening = _REFERENCE_FORMAT_TAIL_RE.sub("", opening).strip(" .?!,:;")
    opening = re.sub(r"^(?:the|a|an)\s+", "", opening, flags=re.IGNORECASE)
    terms = _REFERENCE_TERM_RE.findall(opening)[:10]
    subject = " ".join(terms).strip()

    candidates: list[str] = []
    named_subject = ""
    if subject:
        candidates.append(subject)
        # Possessive algorithm/theorem names often index under the shorter
        # title even when the request contains modifiers such as "shortest-path".
        named = re.search(
            r"\b([A-Z][A-Za-z0-9'’-]*)\b.*?\b"
            r"(algorithm|theorem|protocol|method|model|effect|law|equation|system)\b",
            subject,
        )
        if named:
            named_subject = f"{named.group(1)} {named.group(2)}"
    if not candidates:
        fallback_terms = [
            term
            for term in _REFERENCE_TERM_RE.findall(raw)
            if term.lower() not in _STOP
        ][:8]
        if fallback_terms:
            candidates.append(" ".join(fallback_terms))
    evidence_subject = named_subject or subject
    lowered = raw.lower()
    normalized_lowered = lowered.replace("-", " ")
    # Add only facets the person explicitly asked about. These are retrieval
    # queries, not decoder instructions: their purpose is to expose the exact
    # reference sections needed to verify a compound technical explanation.
    if evidence_subject and "negative" in lowered and "weight" in lowered:
        candidates.append(f"{evidence_subject} negative weight")
    if evidence_subject and "complexity" in normalized_lowered and (
        "heap" in normalized_lowered or "array" in normalized_lowered
    ):
        if "heap" in normalized_lowered:
            heap = "binary heap" if "binary" in normalized_lowered else "heap"
            candidates.append(
                f"{evidence_subject} {heap} complexity"
            )
        if "array" in normalized_lowered:
            candidates.append(f"{evidence_subject} array complexity")
    return list(dict.fromkeys(candidates))[:4]


class EvidenceProvider:
    def __init__(self, root: Path | None = None, *, memory_facade: Any | None = None) -> None:
        self._root = Path(root or PROJECT_ROOT)
        self._memory = memory_facade

    # -------------------------------------------------------------- public
    async def gather(
        self,
        objective: str,
        *,
        task_type: str,
        limit: int = 6,
    ) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        tt = (task_type or "generic").lower()
        if tt in {"repo", "repo_audit", "architecture", "code_audit", "self_claim", "code"}:
            spans.extend(await self._repo_evidence(objective, limit=limit))
        if tt == "factual":
            spans.extend(await self.reference_evidence(objective, limit=limit))
        if tt in {"factual", "self_claim", "generic", "architecture"} or not spans:
            spans.extend(await self._memory_evidence(objective, limit=max(2, limit - len(spans))))
        # De-dupe by rendered text, keep order.
        seen: set[str] = set()
        out: list[EvidenceSpan] = []
        for s in spans:
            key = s.render()[:160]
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        return out[:limit]

    async def reference_evidence(
        self,
        objective: str,
        *,
        limit: int = 4,
    ) -> list[EvidenceSpan]:
        """Read topic evidence from the offline corpus within chat latency bounds.

        The spans only. Callers that must tell a BROKEN corpus from an empty
        one want `reference_evidence_result` instead.
        """

        return (await self.reference_evidence_result(objective, limit=limit)).spans

    async def reference_evidence_result(
        self,
        objective: str,
        *,
        limit: int = 4,
    ) -> ReferenceEvidence:
        """The same read, with retrieval failure kept separate from absence.

        Returning a bare list made the two identical to every caller. A corpus
        that raised and a corpus that held nothing both arrived as `[]`, so the
        citation verifier reported "unconfirmed by local corpus" — an
        epistemic claim about the world — when the truth was that the corpus
        had not been read at all.
        """

        queries = reference_query_candidates(objective)
        if not queries:
            return ReferenceEvidence(spans=[], retrieval_failed=False)
        try:
            from core.knowledge.local_corpus import (
                CONVERSATION_SEARCH_DEADLINE_S,
                get_local_corpus_store,
            )

            corpus = get_local_corpus_store()
            search_groups = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        corpus.search,
                        query,
                        max(6, limit * 2),
                        deadline_s=CONVERSATION_SEARCH_DEADLINE_S,
                    )
                    for query in queries
                )
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "evidence_reference",
                exc,
                severity="warning",
                action="continued factual reasoning without offline reference evidence",
            )
            return ReferenceEvidence(spans=[], retrieval_failed=True)

        query_terms = {
            term.lower().replace("’", "'").removesuffix("'s")
            for term in _REFERENCE_TERM_RE.findall(queries[0])
            if len(term) > 2
        }

        def rank(hit: Any) -> tuple[float, float, int]:
            title_terms = {
                term.lower().replace("’", "'").removesuffix("'s")
                for term in _REFERENCE_TERM_RE.findall(str(getattr(hit, "title", "")))
            }
            overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            return (-overlap, float(getattr(hit, "rank", 0.0) or 0.0), len(title_terms))

        # Rank each query's results by subject-title agreement, then interleave
        # the groups. A broad title hit must not crowd out the focused snippet
        # about a requested limitation or complexity bound. Exact duplicate
        # snippets collapse; different sections of the same article remain.
        ranked_groups = [sorted(list(group or []), key=rank) for group in search_groups]
        ordered: list[Any] = []
        seen: set[tuple[str, str, str]] = set()
        width = max((len(group) for group in ranked_groups), default=0)
        for position in range(width):
            for group in ranked_groups:
                if position >= len(group):
                    continue
                hit = group[position]
                key = (
                    str(getattr(hit, "source", "")),
                    str(getattr(hit, "title", "")),
                    str(getattr(hit, "snippet", "")),
                )
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                ordered.append(hit)
        return ReferenceEvidence(
            spans=[
                EvidenceSpan(
                    "reference",
                    f"{hit.source}:{hit.title}",
                    str(hit.snippet or "")[:1_200],
                )
                for hit in ordered[:limit]
                if str(hit.snippet or "").strip()
            ],
            retrieval_failed=False,
        )

    async def render_pack(self, objective: str, *, task_type: str, limit: int = 6) -> list[str]:
        return [s.render() for s in await self.gather(objective, task_type=task_type, limit=limit)]

    # -------------------------------------------------------------- repo
    async def _repo_evidence(self, objective: str, *, limit: int) -> list[EvidenceSpan]:
        terms = _salient_terms(objective)
        # Always honor explicitly named paths.
        paths = [m.group(1) for m in _PATH_RE.finditer(objective or "")]
        spans: list[EvidenceSpan] = []
        for ref in paths[:3]:
            spans.extend(self._read_named_path(ref))
        if not terms:
            return spans[:limit]
        try:
            hits = await self._ripgrep(terms, limit=limit) if shutil.which("rg") else await asyncio.to_thread(
                self._inprocess_search, terms, limit, _filename_candidates(objective)
            )
            spans.extend(hits)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("evidence_repo", exc)
        return spans[:limit]

    def _read_named_path(self, ref: str) -> list[EvidenceSpan]:
        candidate = self._root / ref
        if not candidate.exists():
            matches = list(self._root.rglob(Path(ref).name))[:1]
            if not matches:
                return []
            candidate = matches[0]
        try:
            lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        except (OSError, ValueError):
            return []
        rel = candidate.relative_to(self._root) if candidate.is_relative_to(self._root) else candidate
        # First few non-trivial lines as a span (signature/docstring region).
        out: list[EvidenceSpan] = []
        for i, ln in enumerate(lines[:40], start=1):
            if ln.strip() and (ln.lstrip().startswith(("def ", "class ", "async def ")) or i == 1):
                out.append(EvidenceSpan("repo", f"{rel}:{i}", ln.strip()[:200]))
            if len(out) >= 4:
                break
        return out

    async def _ripgrep(self, terms: list[str], *, limit: int) -> list[EvidenceSpan]:
        """Search terms in priority order (most specific identifier first).

        ``terms`` is already ranked by :func:`_salient_terms`, so searching them in
        order — rather than OR-ing equally — surfaces the *defining* file of the most
        informative symbol before generic keyword hits crowd it out.
        """
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        gateway = get_subprocess_gateway()
        spans: list[EvidenceSpan] = []
        seen: set[str] = set()
        # Prefer a definition line for the lead identifier when there is one.
        for prefer_def in (True, False):
            for term in terms:
                if len(spans) >= limit:
                    return spans
                pattern = rf"(?:def|class)\s+{re.escape(term)}\b" if prefer_def else re.escape(term)
                argv = (
                    "rg", "--no-heading", "--line-number", "--max-count", "2",
                    "--glob", "*.py",
                    "--glob", "!**/{.venv,__pycache__,archive,dev_archive,node_modules}/**",
                    "-e", pattern, str(self._root),
                )
                try:
                    res = await gateway.run_async(
                        argv, timeout=10.0, read_only=True, source="evidence_provider:ripgrep",
                        # "none", not "auto". ripgrep is a text search and uses
                        # no accelerator, and "auto" could not infer that: the
                        # gateway raised
                        # subprocess_accelerator_capability_unresolved on EVERY
                        # call, the handler below swallowed it, and this
                        # provider returned zero spans. Silently. Every ReAct
                        # answer that was supposed to be grounded in real source
                        # spans has been grounded in nothing.
                        accelerator_capability="none",
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    # Recorded, not swallowed. The failure that hid here
                    # disabled evidence grounding entirely while every caller
                    # kept reporting success, which is the exact shape this
                    # codebase keeps having to dig out.
                    record_degradation(
                        "evidence_repo",
                        exc,
                        severity="degraded",
                        action="repo evidence search failed; answering without source spans",
                    )
                    continue
                for line in (res.stdout or "").splitlines():
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    path, lineno, code = parts
                    try:
                        rel = Path(path).relative_to(self._root)
                    except ValueError:
                        rel = Path(path).name
                    ref = f"{rel}:{lineno}"
                    if ref in seen:
                        continue
                    seen.add(ref)
                    spans.append(EvidenceSpan("repo", ref, code.strip()[:200]))
                    if len(spans) >= limit:
                        return spans
        return spans

    def _search_order(
        self,
        needles: list[str],
        snake: dict[str, str],
        filename_candidates: set[str] | None = None,
    ) -> list[Path]:
        """Which files are worth spending the read budget on, best first.

        The budget used to be spent in directory-walk order, and the walk is
        longer than the budget: 6,299 scannable files against 4,000 reads, with
        core/runtime/subprocess_gateway.py at position 4,152. Asked where
        SubprocessGateway lives, the answer could only ever be incidental
        mentions in aura_main and interface/server, because the file named
        after the symbol was never opened — and a third of the repository was
        unreachable as evidence for the same reason, silently, whatever was
        asked.

        Listing paths is cheap; reading them is not. So the order is decided
        first — the file named after the symbol, then implementation, then
        tooling, then tests — and the budget bounds the reads.
        """
        specific = set(filename_candidates or ())
        # Only the TOP-RANKED term claims a filename. _salient_terms orders by
        # how informative a term is, and letting every term claim one let the
        # least informative decide: "you have a lock ordering system" opened
        # core/consciousness/system.py, because a file is named system.py and
        # "system" was in the list. The rest of the terms still match content;
        # they just do not get to say which file is read first.
        lead = needles[0].lower() if needles else ""
        wanted = ({lead, snake.get(needles[0], lead)} if lead else set()) | specific
        scored: list[tuple[int, str, Path]] = []
        # Prune while walking, not after. rglob yields every path under
        # .venv, node_modules and the worktree copies and leaves the caller to
        # discard them, which cost 1.9s per lookup — more than reading the
        # files. os.walk can be told not to descend at all.
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            here = Path(dirpath)
            parts = set(here.parts)
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                py = here / filename
                stem = py.stem.lower()
                # These buckets are the READ order, and the scoring below
                # is the RANK. They have to agree: with both named-after
                # kinds in one bucket, path order decided which was opened
                # first, and stopping on "enough strong hits" could stop
                # before the better file — core/consciousness/contract.py
                # answered "how does the health contract work" while
                # core/runtime/health_contract.py went unread.
                if stem in specific:
                    bucket = 0
                elif stem in wanted:
                    bucket = 1
                elif parts & {"tests", "test"} or stem.startswith("test_"):
                    bucket = 4
                elif parts & {"tools", "scripts", "training"}:
                    bucket = 3
                else:
                    bucket = 2
                scored.append((bucket, str(py), py))
        scored.sort()
        return [py for _bucket, _key, py in scored]

    def _inprocess_search(
        self,
        terms: list[str],
        limit: int,
        filename_candidates: set[str] | None = None,
    ) -> list[EvidenceSpan]:
        """Priority scan: collect hits for the most specific term first.

        Mirrors the ripgrep priority so the defining file of the lead identifier
        surfaces before generic keyword matches. Reads every candidate file once
        and buckets matched lines by which term hit, then drains buckets in the
        ranked order of ``terms``.
        """
        needles = [t for t in terms if t]
        if not needles:
            return []
        import re as _re

        def_re = {t: _re.compile(rf"\b(?:def|class)\s+{_re.escape(t)}\b") for t in needles}
        # SubprocessGateway -> subprocess_gateway, so a file named after the
        # symbol can be recognised as its home.
        snake = {t: snake_case(t) for t in needles}
        # (score, term_rank, tie) -> span. Candidates are SCORED and sorted
        # rather than taken in walk order: the previous version filled a small
        # per-term bucket first-come-first-served, so six test files could
        # crowd out the module that defines the symbol. Measured: asking about
        # SubprocessGateway returned six test spans and not
        # core/runtime/subprocess_gateway.py.
        candidates: list[tuple[float, int, int, EvidenceSpan]] = []
        per_term_cap = max(2, limit)
        per_term_candidates = max(per_term_cap * 8, 40)
        counts = {t: 0 for t in needles}
        #: Which of the question's terms each file actually contains. A file
        #: carrying most of what was asked about is the subject; a file
        #: carrying one common word is a coincidence. Without this, "you have
        #: a lock ordering system ... locks in opposite order" ranked
        #: aura_main.py above core/runtime/lockdep.py, which is the file that
        #: detects exactly that.
        covered: dict[str, set[str]] = {}
        #: Matching lines per file, and how many lines the file has. Presence
        #: is not evidence of subject: aura_main.py mentions every subsystem
        #: once, so it "covered" more of a question about lock ordering than
        #: core/runtime/lockdep.py, which is the file that implements it.
        #: What separates them is DENSITY — how much of the file is about the
        #: thing — and both numbers are already in hand during the scan.
        frequency: dict[str, Counter[str]] = {}
        length: Counter[str] = Counter()
        low_information = _low_information_terms()
        scanned = 0
        tie = 0
        strong = 0
        for py in self._search_order(needles, snake, filename_candidates):
            if scanned >= _MAX_FILES_READ:
                break
            # The order puts the file named after the symbol first, so once
            # enough hits that strong exist, nothing later can outrank them and
            # the remaining reads are latency spent for no change in answer.
            if strong >= limit:
                break
            scanned += 1
            try:
                rel = py.relative_to(self._root)
                parts = set(rel.parts)
                stem = py.stem.lower()
                # What a person does when asked where a symbol lives: look at
                # the file named after it, in the implementation tree, at the
                # line that defines it.
                is_test = bool(parts & {"tests", "test"}) or stem.startswith("test_")
                is_tooling = bool(parts & {"tools", "scripts", "training"})
                path_score = 0.0
                if not is_test and not is_tooling:
                    path_score += 2.0
                elif is_tooling:
                    path_score += 0.5
                file_lines = py.read_text(encoding="utf-8", errors="ignore").splitlines()
                length[str(rel)] = len(file_lines)
                for i, ln in enumerate(file_lines, start=1):
                    for rank, t in enumerate(needles):
                        if t not in ln:
                            continue
                        # Count the term BEFORE any admission cap. The cap
                        # bounds how many quotable lines are kept; it must not
                        # decide how much a file is about the subject, or the
                        # ranking sees zero for every file reached after the
                        # buckets fill — which is every file past the first
                        # dozen, including the one that implements the answer.
                        if t.lower() not in low_information:
                            covered.setdefault(str(rel), set()).add(t)
                            frequency.setdefault(str(rel), Counter())[t] += 1
                        score = path_score
                        if def_re[t].search(ln):
                            score += 4.0
                        # The ordering and the scoring have to agree about
                        # what "the file named after this" means, or the right
                        # file is opened first and then outscored by an
                        # incidental mention elsewhere.
                        # Same rule as the read order above: a filename bonus
                        # is only granted to the lead term or to a multi-word
                        # name the question spells out. Letting any term grant
                        # it put system.py at the top for a question about
                        # lock ordering.
                        named_file = False
                        if stem in (filename_candidates or ()):
                            # A stem matching several of the asked words is a
                            # more specific claim than one matching a single
                            # common term: "how does the health contract work"
                            # named core/consciousness/contract.py ahead of
                            # core/runtime/health_contract.py on a tie.
                            score += 6.0
                            named_file = True
                        elif rank == 0 and (stem == snake[t] or stem == t.lower()):
                            score += 5.0
                            named_file = True
                        # A cap that admits candidates in walk order still lets
                        # early files crowd out the answer: with the cap alone,
                        # asking about SubprocessGateway filled up on aura_main
                        # and interface/server before ever reaching
                        # core/runtime/subprocess_gateway.py. A hit in the file
                        # named after the symbol, or one that defines it, is
                        # admitted regardless of how full the bucket is —
                        # that is precisely the hit worth displacing others.
                        if score < _STRONG_EVIDENCE_SCORE and counts[t] >= per_term_candidates:
                            break
                        counts[t] += 1
                        tie += 1
                        # Only a hit in the file NAMED after the subject can
                        # end the scan early. Any def line in any
                        # implementation file scores "strong", so counting
                        # those stopped the search after ten files: asked
                        # about lock ordering, core/runtime/lockdep.py sits at
                        # position 2213 and was never opened.
                        if named_file and score >= _STRONG_EVIDENCE_SCORE:
                            strong += 1
                        candidates.append(
                            (
                                -score,
                                rank,
                                tie,
                                EvidenceSpan("repo", f"{rel}:{i}", ln.strip()[:200]),
                            )
                        )
                        break
            except (OSError, ValueError):
                continue

        # Coverage outranks a single strong line — but as a PROPORTION of
        # what was asked, not as a count. Rewarding the raw number put
        # aura_main.py first for every question, because the entrypoint
        # mentions every subsystem in the system and therefore "covers" more
        # of any question than the file that actually implements the answer.
        informative = [n for n in needles if n.lower() not in low_information]
        documents = [f for f in length if length[f] > 0]
        average_length = (
            sum(length[f] for f in documents) / len(documents) if documents else 1.0
        )
        document_frequency = Counter(
            term
            for term in informative
            for source_file in frequency
            if frequency[source_file].get(term)
        )
        total_documents = max(1, len(documents))

        def _subject_bonus(ref: str) -> float:
            """BM25 over the files this scan actually read."""
            source_file = ref.split(":", 1)[0]
            counts = frequency.get(source_file)
            if not counts:
                return 0.0
            file_length = max(1, length[source_file])
            score = 0.0
            for term in informative:
                term_frequency = counts.get(term, 0)
                if not term_frequency:
                    continue
                seen_in = document_frequency.get(term, 0)
                idf = math.log(
                    1 + (total_documents - seen_in + 0.5) / (seen_in + 0.5)
                )
                saturated = term_frequency * (_BM25_K1 + 1) / (
                    term_frequency
                    + _BM25_K1
                    * (1 - _BM25_B + _BM25_B * file_length / average_length)
                )
                score += idf * saturated
            return _SUBJECT_WEIGHT * score

        candidates = [
            (score - _subject_bonus(span.ref), rank, tie, span)
            for score, rank, tie, span in candidates
        ]
        candidates.sort()
        ordered: list[EvidenceSpan] = []
        seen: set[str] = set()
        per_file: dict[str, int] = {}
        for _score, _rank, _tie, span in candidates:
            if span.ref in seen:
                continue
            # Evidence from one file is one piece of evidence. Twelve lines of
            # aura_main.py is not a survey of the codebase, and it crowded out
            # every other file that had something to say.
            source_file = span.ref.split(":", 1)[0]
            if per_file.get(source_file, 0) >= _MAX_SPANS_PER_FILE:
                continue
            per_file[source_file] = per_file.get(source_file, 0) + 1
            seen.add(span.ref)
            ordered.append(span)
            if len(ordered) >= limit:
                break
        return ordered

    # -------------------------------------------------------------- memory
    def _ensure_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        try:
            from core.container import ServiceContainer

            self._memory = ServiceContainer.get("memory_facade", default=None)
        except (ImportError, RuntimeError, AttributeError):
            self._memory = None
        return self._memory

    async def _memory_evidence(self, objective: str, *, limit: int) -> list[EvidenceSpan]:
        facade = self._ensure_memory()
        if facade is None or not hasattr(facade, "search"):
            return []
        try:
            results = await facade.search(objective, limit=limit)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("evidence_memory", exc)
            return []
        spans: list[EvidenceSpan] = []
        for item in list(results or [])[:limit]:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                ref = str(item.get("id", "") or "memory")
            else:
                content = str(item or "").strip()
                ref = "memory"
            if content:
                spans.append(EvidenceSpan("memory", ref, content[:240]))
        return spans


def get_evidence_provider() -> EvidenceProvider:
    return EvidenceProvider()

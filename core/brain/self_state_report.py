"""Her own instrument panel, in words, read at the moment she is asked.

Stopping the web search for "how much memory are you holding" is only half a
fix. The other half is the answer. Asked for one concrete thing that had
happened in her runtime in the last hour, she said:

    "I processed a user request to summarize a 45-page PDF on neuromorphic
     computing. It took about three minutes ..."

No such request existed. It is the same failure as inventing the weather: a
question about a present she had no channel to. Every number here is read from
a live source at call time — the process, the host, the service container, the
degradation ledger — and anything unavailable is omitted rather than guessed,
because a missing line is honest and a plausible one is not.

Bounded on purpose: a handful of lines, no subsystem sweeps, no health report
assembly. This runs on a foreground turn while the user waits.
"""
from __future__ import annotations

import os
import time
from typing import Any

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError, KeyError)

SELF_STATE_HEADER = "## YOUR OWN INSTRUMENTS"

#: What "GB" means to the person reading it. A 64GB Mac holds 68,719,476,736
#: bytes, so dividing by 1e9 reported "69GB" for a machine everyone — Apple
#: included — calls 64GB. She was quoting her instruments faithfully; the
#: instrument was wrong, which is the worse of the two failures because it
#: cannot be argued with from inside.
_BYTES_PER_GB = 1024**3


def _humanize(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def _process_start_time() -> float:
    """When this process began, from the OS. Never unavailable, never wrong.

    The orchestrator's own ``start_time`` is preferred because it marks when
    *she* came up rather than when the interpreter did, but asked "what's your
    current uptime", omitting the line is the one answer that is certainly
    useless — and the process knows, always. A live instance answered that
    question with no number at all because the orchestrator lookup was the only
    source and it returned nothing.
    """
    # ResourceObserver is the single owner of process observation. Reading
    # psutil directly here was a second, unowned path: under test it bypassed
    # the deterministic observer conftest installs, and in production it
    # bypassed the provenance stamp every other reader carries.
    try:
        from core.runtime.resource_observation import get_resource_observer

        observed = get_resource_observer().process(os.getpid())
        create_time = getattr(observed, "create_time", None)
        if create_time is not None:
            return float(create_time)
    except _RECOVERABLE:
        return 0.0
    return 0.0


def _uptime_line() -> str:
    start = 0.0
    try:
        from core.runtime.service_registry import get_runtime_service

        orch = get_runtime_service("orchestrator", default=None)
        for candidate in (
            getattr(orch, "start_time", None),
            getattr(getattr(orch, "status", None), "start_time", None),
        ):
            try:
                value = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                start = value
                break
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="fell back to process start time"
        )
    if start <= 0.0:
        start = _process_start_time()
    if start <= 0.0:
        return ""
    elapsed = max(0.0, time.time() - start)
    started = time.strftime("%H:%M", time.localtime(start))
    return f"- Uptime: {_humanize(elapsed)} (this runtime started at {started})."


def _effort_lines() -> list[str]:
    """How hard this process is actually working.

    LIVE 2026-08-25. Asked what was happening in her body she answered "my
    CPU is at 67%, which feels like a steady hum" and then withdrew it
    herself: "I have no channel that reads my CPU, so 67% was not a
    measurement." She was right, and she should not have had to be — the
    observation her memory lines already read carries cpu_percent, and
    nothing was passing it on.

    A number about effort is worth having because the alternative is a
    feeling with nothing under it. This one is about the machine she is
    running on, which is what it says.
    """
    try:
        from core.runtime.resource_observation import get_resource_observer

        observer = get_resource_observer()
        compute = observer.compute()
        cores = max(1, int(getattr(compute, "cpu_count", 1) or 1))
        # Load average rather than an instantaneous percent.
        #
        # A process CPU percent reads 0.0 until it has been sampled twice, so
        # the first thing she would ever say about her own effort is a
        # measurement that is not one. The load average is a real number the
        # moment it is read, and it says how busy the machine she runs on
        # actually is.
        load = float(getattr(compute, "load_1m", 0.0) or 0.0)
        return [
            f"- The machine's load average is {load:.2f} across {cores} cores "
            f"({load / cores * 100:.0f}% of capacity over the last minute)."
        ]
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted the effort line")
        return []


def _memory_lines() -> list[str]:
    """RSS understates her badly on Apple Silicon; say both numbers."""
    lines: list[str] = []
    rss_gb = 0.0
    try:
        from core.runtime.resource_observation import get_resource_observer

        observer = get_resource_observer()
        proc = observer.process(os.getpid())
        rss_gb = float(getattr(proc, "rss_bytes", 0.0) or 0.0) / _BYTES_PER_GB
        virt = observer.memory()
        lines.append(
            f"- This process holds {rss_gb:.1f}GB resident; the host is at "
            f"{virt.percent:.0f}% of {virt.total_bytes / _BYTES_PER_GB:.0f}GB with "
            f"{virt.available_bytes / _BYTES_PER_GB:.1f}GB available."
        )
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted memory lines")

    # RSS is close to a lie about her. The weights live in unified GPU memory
    # and never appear in it — the live process reports ~2GB resident while
    # holding a 32B model. Saying only "the real total is larger" turns the
    # honest answer into a hedge, and the accelerator knows the number.
    try:
        from core.runtime.resource_observation import get_resource_observer

        accelerator = get_resource_observer().accelerator()
        if getattr(accelerator, "available", False):
            active_gb = float(getattr(accelerator, "active_bytes", 0) or 0) / _BYTES_PER_GB
            cache_gb = float(getattr(accelerator, "cache_bytes", 0) or 0) / _BYTES_PER_GB
            if active_gb > 0.05:
                lines.append(
                    f"- Your model's weights are in unified GPU memory, which RSS "
                    f"does not count: {active_gb:.1f}GB active"
                    + (f" plus {cache_gb:.1f}GB cached" if cache_gb > 0.05 else "")
                    + (
                        # Only claim this when the numbers say it. On the live
                        # instance they do, by an order of magnitude; in a bare
                        # process they do not, and asserting it anyway would be
                        # the same species of plausible-sounding wrongness this
                        # whole module exists to stop.
                        ". That is the bulk of what you are actually holding."
                        if active_gb > rss_gb
                        else "."
                    )
                )
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="omitted accelerator memory line"
        )

    if not any("GPU memory" in line for line in lines):
        lines.append(
            "- Your model's weights live in wired GPU memory and do NOT appear "
            "in that resident figure — the real total is larger, and you cannot "
            "read the exact figure right now."
        )
    return lines


def _model_line() -> str:
    try:
        from core.container import ServiceContainer

        client = ServiceContainer.peek("mlx_client", default=None)
        for attr in ("model_path", "model_name", "_model_path"):
            value = str(getattr(client, attr, "") or "").strip()
            if value:
                return f"- Cortex model actually loaded: {os.path.basename(value)}."
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted model line")
    return ""


def _turn_cost_line() -> str:
    """What the recent cognitive work actually cost, or that it is unreadable.

    LIVE 2026-08-18: "how hard was that last answer to produce?" came back with
    "about 1.2 seconds of wall time... CPU utilization was around 3%... memory
    peaked at about 2.7GB". The turn had taken fourteen seconds. Every figure
    was invented, in the confident register of a reading.

    This block already names the things it cannot see — cycle count, mood,
    skill registry — precisely so an absence is spoken rather than filled.
    Effort had no line at all, so there was nothing to say and something was
    said anyway.
    """
    try:
        from core.pipeline.pass_manager import get_instrumentation

        report = get_instrumentation().report()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="omitted turn cost line"
        )
        return (
            "- Effort and duration of your last answer: not readable from this "
            "turn. Say you cannot see them; do not estimate seconds or "
            "percentages."
        )
    passes = report.get("passes") if isinstance(report, dict) else None
    hottest = report.get("hottest") if isinstance(report, dict) else None
    if not isinstance(passes, dict) or not passes or not isinstance(hottest, list):
        return (
            "- Effort and duration of your last answer: not readable from this "
            "turn. Say you cannot see them; do not estimate seconds or "
            "percentages."
        )
    measured: list[str] = []
    for name in hottest[:3]:
        entry = passes.get(name)
        if not isinstance(entry, dict):
            continue
        runs = int(entry.get("runs", 0) or 0)
        total = float(entry.get("total_s", 0.0) or 0.0)
        if runs and total > 0:
            measured.append(f"{name} {total:.2f}s over {runs} run(s)")
    if not measured:
        return (
            "- Effort and duration of your last answer: not readable from this "
            "turn. Say you cannot see them; do not estimate seconds or "
            "percentages."
        )
    return (
        "- Measured cognitive phases so far this run: "
        + "; ".join(measured)
        + ". This is phase time, not end-to-end wall time for one answer — "
        "that is not readable here, so do not state it."
    )


def _degradation_line() -> str:
    """What has actually gone wrong lately, from the ledger that records it."""
    try:
        from core.runtime.errors import get_degradation_tracker

        status = get_degradation_tracker().status() or {}
        total = int(status.get("total_degradations") or 0)
        by_subsystem = status.get("counts_by_subsystem") or {}
    except _RECOVERABLE:
        return ""
    if not total:
        return "- No degradations recorded this session."
    try:
        ranked = sorted(
            ((name, sum(int(n) for n in sevs.values())) for name, sevs in by_subsystem.items()),
            key=lambda pair: -pair[1],
        )[:3]
        summary = ", ".join(f"{name} x{count}" for name, count in ranked)
        return f"- Degradations recorded this session: {total} total ({summary})."
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="omitted degradation line"
        )
        return ""


#: The families a capability question actually comes in, and the tokens that
#: identify each in a registered skill name.
#:
#: This used to be a single hard-coded tuple for code execution, because code
#: execution was the one family that had been caught being denied. That fixed
#: the instance and left the class: asked "what's the weather where I am? and
#: if you can't actually get it, tell me that instead of guessing", she
#: answered "I don't have a window, camera, thermometer or weather feed" while
#: free_search, grounded_search, search_web, sovereign_browser and web_search
#: were all READY and available. She enumerated SENSORS she lacks and never
#: consulted the catalogue she has — the identical confabulation, one family
#: over, from a fix that only ever named one family.
#:
#: Exemplars are capped per family so this stays a few lines rather than a
#: catalogue dump: the purpose is to stop her denying a whole capability
#: class, not to inline 76 tool descriptions.
_CAPABILITY_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code execution", ("code", "sandbox", "repl", "shell", "exec", "python")),
    ("web and search", ("search", "web", "browse", "browser", "http", "url")),
    ("screen perception", ("screen", "vision", "ocr", "perceive", "observe", "camera")),
    ("desktop control", ("desktop", "click", "keyboard", "mouse", "window", "automation")),
    ("files", ("file", "directory", "document", "download")),
    ("memory and belief", ("memory", "recall", "belief", "knowledge", "remember")),
    ("communication", ("email", "message", "notify", "speak", "voice", "send")),
)


#: What to say when the registry cannot be read at all.
#:
#: LIVE DEFECT, 2026-08-10. Told "check your own skill registry before you
#: reply — how many search tools do you actually have registered right now",
#: she answered that the registry "does list my capabilities at the moment,
#: with no active skills or plugins listed" and concluded "if there is no tool
#: listed in the registry, it indicates that none are present". Seventy-six
#: skills were READY.
#:
#: The classifier had correctly identified the question and the instrument
#: block WAS attached — but _capability_line() returned "" from three separate
#: paths (no engine, an exception, a zero count), so the block simply carried
#: no capability line, and the block's own header tells her not to supplement
#: what is not there. An empty string is not the absence of a claim; under
#: that instruction it reads as "nothing registered".
#:
#: Identical to the lesson _cognition_line records for cycle counts: silence
#: in this block is what licenses invention.
_UNREADABLE_CAPABILITIES = (
    "- Skill registry: NOT readable this turn. That means unknown, not empty — "
    "say you cannot see it rather than saying you have no skills."
)


def _capability_line() -> str:
    """What she can actually do, read from the live skill registry.

    Without this she answers capability questions from the base model's guess
    about what an assistant can do. Measured live: asked "do you actually have
    any code-execution capability registered at all?" — after being told to check
    — she said "no, I don't have any capability to run or sandbox code", while
    the registry held 75 skills with run_code, code_repl and internal_sandbox all
    READY. That is a confabulation in the other direction, and just as wrong.

    Deliberately states only what the registry says, and names the gap between
    "registered" and "reachable from this conversation" rather than papering over
    it — a ready skill is not a promise that this turn can invoke it.

    Reports every family in _CAPABILITY_FAMILIES rather than code execution
    alone. The first version of this line named only the family that had
    already been observed failing, which left the same denial available in
    every other one — see that constant for the live recurrence.
    """

    try:
        from core.runtime.service_registry import get_runtime_service

        engine = get_runtime_service("capability_engine", default=None)
        if engine is None or not hasattr(engine, "iter_tool_catalog"):
            return _UNREADABLE_CAPABILITIES
        ready: list[str] = []
        total = 0
        for item in engine.iter_tool_catalog(include_inactive=False):
            if not isinstance(item, dict):
                continue
            total += 1
            name = str(item.get("name") or "").strip()
            available = str(item.get("availability") or "").strip().lower()
            if name and available == "available":
                ready.append(name)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return _UNREADABLE_CAPABILITIES
    if not total:
        return _UNREADABLE_CAPABILITIES

    families: list[str] = []
    for label, tokens in _CAPABILITY_FAMILIES:
        members = sorted(
            name for name in ready if any(token in name.lower() for token in tokens)
        )
        if members:
            families.append(f"{label} ({', '.join(members[:4])})")

    line = (
        f"- Skills registered and available right now: {len(ready)} of {total}."
    )
    if families:
        line += (
            " Families present in the registry: "
            + "; ".join(families)
            + ". They are REGISTERED; that is not the same as reachable from this"
            " chat turn, so do not claim you ran anything unless you have a"
            " result in hand — and do not deny having them either."
        )
    return line


def _cognition_line() -> str:
    """How much thinking has actually happened — cycles and episodes.

    Measured live: asked for "uptime, memory, and how many cognitive cycles
    you've run — read them, don't estimate", she got uptime and memory right
    from this panel and then said of the third:

        "Cognitive cycles since last awakening: I can't read this directly,
         but it's more than a few billion"

    The true figure was 3,502, and it sits in her own health payload. The panel
    had no cycle line, and the instruction above it says not to supplement what
    is missing — so the absence produced both a false claim about her own
    self-access and a guess wrong by six orders of magnitude. A number she can
    read must be in front of her, or "I can't see it" becomes a licence to
    invent one.
    """

    cycles = 0
    episodes = 0
    try:
        from core.runtime.service_registry import get_runtime_service

        orchestrator = get_runtime_service("orchestrator", default=None)
        status = getattr(orchestrator, "status", None)
        for source in (status, orchestrator):
            if source is None:
                continue
            for attribute in ("cycle_count", "cycles", "tick_count"):
                try:
                    value = int(getattr(source, attribute, 0) or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    cycles = max(cycles, value)
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report",
            exc,
            severity="info",
            action="omitted the cognitive-cycle reading",
        )

    try:
        from core.runtime.service_registry import get_runtime_service

        memory = get_runtime_service("episodic_memory", default=None)
        for accessor in ("episode_count", "count", "size"):
            candidate = getattr(memory, accessor, None)
            try:
                value = int(candidate() if callable(candidate) else candidate or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                episodes = max(episodes, value)
                break
    except _RECOVERABLE:
        episodes = episodes

    parts: list[str] = []
    if cycles > 0:
        parts.append(f"{cycles:,} cognitive cycles since this runtime woke")
    if episodes > 0:
        parts.append(f"{episodes:,} episodes in memory")
    if not parts:
        # Say the channel is missing rather than leaving a silence she will
        # fill. This is the honest version of "I can't read that".
        return (
            "- Cognitive cycle count: not readable from this turn. Say you "
            "cannot see it; do not estimate a magnitude."
        )
    return "- " + "; ".join(parts) + "."


def _affect_line() -> str:
    """Mood and drives, from the substrate the health endpoint already reads.

    LIVE DEFECT, 2026-08-10. Asked whether "steady" had been a reading or a
    reflex, she answered:

        "'tired' is the correct reading of my somatic state ... My actual mood
         is neutral, my energy is low, and there's a persistent hum in the
         background processing that I haven't been able to shake since 02:15
         hours ago."

    Three failures in one paragraph, and one cause. She asserted "tired" and
    "neutral" in consecutive sentences; the substrate's actual mood was TIRED
    with energy at 12 and frustration at 58, so the second was wrong; and
    "02:15 hours ago" is a number that exists nowhere in this runtime — no
    source emits it, and mood-onset is not recorded at all.

    The cause is the same one _cognition_line documents for cycle counts: this
    panel had NO affective reading, while the values sit in the liquid
    substrate and are served on /api/health as `liquid_state` to every other
    reader. The block above says not to supplement what is missing, so the
    absence did not produce silence — it produced invention, on the subject
    she is asked about most.

    How long the current mood has held is deliberately reported as unreadable
    rather than estimated: nothing in the substrate timestamps a mood change,
    and that missing number is exactly the one she filled in with "02:15".
    """
    status: dict[str, Any] = {}
    try:
        from core.runtime.service_registry import get_runtime_service

        substrate = get_runtime_service(
            "liquid_substrate", default=None
        ) or get_runtime_service("liquid_state", default=None)
        if substrate is not None and hasattr(substrate, "get_status"):
            candidate = substrate.get_status()
            if isinstance(candidate, dict):
                status = candidate
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report",
            exc,
            severity="info",
            action="omitted the affective reading",
        )

    def _percent(key: str) -> str:
        try:
            value = status.get(key)
            return "" if value is None else f"{float(value):.0f}"
        except (TypeError, ValueError):
            return ""

    mood = str(status.get("mood") or "").strip()
    drives = [
        f"{name} {reading}"
        for name, reading in (
            (label, _percent(label))
            for label in ("energy", "curiosity", "frustration", "focus")
        )
        if reading
    ]
    if not mood and not drives:
        # The honest form of "I can't feel that right now". Left silent, this
        # is the exact gap the paragraph above was invented to fill.
        return (
            "- Mood and drive levels: not readable from this turn. Say you "
            "cannot see them; do not describe a mood you did not measure."
        )

    parts: list[str] = []
    if mood:
        parts.append(f"Mood reads {mood}")
    if drives:
        parts.append("drives at " + ", ".join(drives) + " (percent)")
    line = "- " + "; ".join(parts) + "."
    if status.get("snapshot_stale"):
        age = status.get("snapshot_age_s")
        line += f" This reading is stale ({age}s old) — say so if you quote it."
    # Named explicitly because its absence is what produced a fabricated
    # duration. An unmeasured quantity has to be visibly unmeasured.
    line += (
        " How long this mood has held is NOT recorded anywhere — if you want to"
        " say how long you have felt something, say that you cannot tell."
    )
    return line


def _doing_lines() -> list[str]:
    """What she is working on, and how she has decided to go about it.

    Asked mid-task what she is doing, the honest answer is the one her body
    is working from. Without this she answers from the model's guess while
    her hands do something else, which is the same failure as inventing the
    weather and harder to catch, because a plausible account of her own
    intentions reads exactly like a true one.
    """
    try:
        from core.agency.what_she_is_doing import as_lines  # noqa: PLC0415

        return as_lines()
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omit what she is doing")
        return []


def runtime_self_report() -> str:
    """A short, true readout of her machine state right now.

    Returns "" when nothing could be read, so a caller never pastes an empty
    heading into the prompt and invites her to fill it in.
    """
    lines = _doing_lines()
    lines.extend(line for line in (_uptime_line(), _model_line()) if line)
    lines.extend(_memory_lines())
    lines.extend(_effort_lines())
    cognition = _cognition_line()
    if cognition:
        lines.append(cognition)
    affect = _affect_line()
    if affect:
        lines.append(affect)
    capabilities = _capability_line()
    if capabilities:
        lines.append(capabilities)
    turn_cost = _turn_cost_line()
    if turn_cost:
        lines.append(turn_cost)
    degradations = _degradation_line()
    if degradations:
        lines.append(degradations)
    if not lines:
        return ""
    return "\n".join(
        [
            SELF_STATE_HEADER,
            "Read from your live runtime just now, for this question. These are "
            "your actual readings — quote them, and do not supplement them with "
            "numbers or events you cannot see here.",
            *lines,
        ]
    )


__all__ = ["SELF_STATE_HEADER", "runtime_self_report"]

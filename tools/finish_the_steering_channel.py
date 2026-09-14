#!/usr/bin/env python3
"""Prepare measured steering authority without changing the live deployment.

Seal, measure, independently replay, adjudicate and issue. A successful exit
means an authority package was prepared, not that a live channel is serving.
Installation belongs to the controlled model-publication and runtime lifecycle
owners. Historical fusion certificates remain evidence for their own bases.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_unattended")

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402

OUT = REPO / "artifacts/migration/27b/recovery/unattended"
STATUS = OUT / "status.json"
TRAINED = REPO / "training/vectors/gradient-trained"
PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan_wide.json"
PYTHON = "/Users/bryan/.aura/live-source/.venv/bin/python"

#: Long enough for a 27B campaign that decodes 216 samples, and bounded so a
#: hung step is a reported failure rather than a wait nobody ends.
CAMPAIGN_TIMEOUT_S = 5 * 60 * 60
STEP_TIMEOUT_S = 45 * 60

#: The wait for training gets its own bound, and a far longer one. It borrowed
#: the campaign's and gave up at five hours on a run that needed six and a
#: quarter -- with three of five dimensions already trained and written. A wait
#: that ends before the thing it is waiting for is not a timeout, it is a
#: guess. Progress is checked as well as elapsed time, so a run that has
#: genuinely stopped writing still ends the wait.
TRAINING_WAIT_S = 18 * 60 * 60

#: A dimension writes its sixteen vectors when it FINISHES, and a dimension
#: takes about ninety minutes. So a ninety-minute stall threshold is the
#: interval it is meant to detect, and it fired forty-nine minutes before the
#: last dimension finished -- on a run that then completed all eighty vectors.
#: Three hours is twice the longest gap the run actually has.
TRAINING_STALL_S = 3 * 60 * 60


def say(stage: str, **facts: object) -> None:
    record = {"stage": stage, "at": time.time(), "at_iso": time.strftime("%Y-%m-%d %H:%M:%S"), **facts}
    history = []
    if STATUS.exists():
        try:
            history = json.loads(STATUS.read_text()).get("history", [])
        except ValueError:
            history = []
    write(STATUS, json.dumps({"latest": record, "history": [*history, record]}, indent=2) + "\n")
    print(f"[{record['at_iso']}] {stage}: {facts}", flush=True)


def write(path: Path, content: str) -> None:
    with local_internal_governed_scope("steering.preparation", domain="file_write"):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="steering.preparation")
        gateway.write_text(path, content, source="steering.preparation")


def run(name: str, args: list[str], timeout: int = STEP_TIMEOUT_S) -> subprocess.CompletedProcess:
    say(f"{name}:start", argv=" ".join(args[1:4]))
    done = subprocess.run(  # noqa: S603
        args, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
    )
    write(OUT / f"{name}.log", done.stdout + "\n--- stderr ---\n" + done.stderr)
    say(f"{name}:done", returncode=done.returncode, tail=done.stdout.strip().splitlines()[-1:])
    return done


def training_is_finished() -> bool:
    alive = subprocess.run(  # noqa: S603
        ["/usr/bin/pgrep", "-f", "train_steering_vectors_by_gradient"],
        capture_output=True, text=True, check=False,
    )
    return alive.returncode != 0


def wait_for_someone_elses_campaign(result: Path) -> bool:
    """Wait for a campaign this process did not start, and did not interrupt.

    The parallel agent restarted the campaign at 01:07 with install and restart
    switched off, so it ends at a result file and stops. Running a second
    campaign beside it would fight for the one GPU and measure neither, so this
    waits for theirs -- for the file AND for the process that writes it to be
    gone, because the file appears at the end of a long write and a half-read
    result is worse than no result.
    """
    say("waiting_for_campaign", result=str(result), started_by="not this process")
    waited = 0
    while waited < CAMPAIGN_TIMEOUT_S:
        running = subprocess.run(  # noqa: S603
            ["/usr/bin/pgrep", "-f", "run_caa_steering_campaign"],
            capture_output=True, text=True, check=False,
        ).returncode == 0
        if result.exists() and not running:
            size = result.stat().st_size
            time.sleep(20)
            if result.stat().st_size == size:
                say("campaign_result_arrived", bytes=size, waited_s=waited)
                return True
        if not running and not result.exists() and waited > 300:
            say("gave_up_waiting_for_campaign",
                why="nothing is running and no result was written", seconds=waited)
            return False
        time.sleep(60)
        waited += 60
    say("gave_up_waiting_for_campaign", why="wall clock", seconds=waited)
    return False


def main() -> int:
    result = REPO / "artifacts/migration/27b/recovery/campaign_result_trained.json"
    if "--from-campaign" in sys.argv:
        if not wait_for_someone_elses_campaign(result):
            return 1
        return carry_the_result(result)

    say("waiting_for_training")
    waited = 0
    last_change = (0, time.time())
    while not training_is_finished():
        time.sleep(60)
        waited += 60
        written = len(list(TRAINED.glob("*_layer*.npz")))
        if written != last_change[0]:
            last_change = (written, time.time())
            say("training_progress", vectors=written, waited_s=waited)
        elif time.time() - last_change[1] > TRAINING_STALL_S:
            say("gave_up_waiting_for_training",
                seconds=waited, vectors=written, why="no vector written in "
                f"{TRAINING_STALL_S // 60} minutes")
            return 1
        if waited > TRAINING_WAIT_S:
            say("gave_up_waiting_for_training", seconds=waited, why="wall clock")
            return 1
    report = TRAINED / "training_report.json"
    vectors = sorted(TRAINED.glob("*_layer*.npz"))
    say("training_finished", vectors=len(vectors), report=report.exists(), waited_s=waited)
    if len(vectors) < 80:
        say("stopped", why=f"training wrote {len(vectors)} vectors, expected 80")
        return 1

    if run("seal", [PYTHON, "tools/seal_caa_generation.py", "--plan", str(PLAN),
                    "--vectors", str(TRAINED)]).returncode != 0:
        say("stopped", why="the generation would not seal")
        return 1

    campaign = run(
        "campaign",
        [PYTHON, "tools/run_caa_steering_campaign.py", "--plan", str(PLAN),
         "--vectors", str(TRAINED), "--alpha", "0.2", "--trials", "4",
         "--out", str(result)],
        timeout=CAMPAIGN_TIMEOUT_S,
    )
    if not result.exists():
        say("stopped", why="the campaign wrote no result", returncode=campaign.returncode)
        return 1
    return carry_the_result(result)


def carry_the_result(result: Path) -> int:
    """Reopen this result and prepare authority; never install or restart."""
    import hashlib

    result_digest = hashlib.sha256(result.read_bytes()).hexdigest()
    output = OUT / result_digest
    with local_internal_governed_scope("steering.preparation", domain="file_write"):
        get_file_write_gateway().ensure_directory(output, source="steering.preparation")
    verdict_path = output / "campaign_verdict.json"
    if run("verdict", [PYTHON, "tools/read_the_steering_campaign.py", "--result", str(result),
                       "--out", str(verdict_path)]).returncode != 0:
        say("stopped", why="this result could not be replayed; no stale verdict consumed")
        return 1
    verdict = json.loads(verdict_path.read_text())
    say("campaign_read",
        causal=verdict.get("causal_effect_positive"),
        unmet=verdict.get("unmet_requirements"),
        layer_specificity=verdict.get("layer_assignment_specificity"))
    if not verdict.get("causal_effect_positive"):
        say("stopped", why="the trained vectors did not pass the campaign; the "
                           "running generation is untouched")
        return 2

    evidence = output / "independent_evidence.json"
    evaluation = output / "causal_evaluation.json"
    authority = output / "steering_authority.json"
    if run("verify", [PYTHON, "tools/verify_caa_steering_campaign.py",
                      "--result", str(result), "--metadata", str(TRAINED / "metadata.json"),
                      "--generation-dir", str(TRAINED), "--out", str(evidence)]).returncode != 0:
        say("stopped", why="independent replay refused")
        return 1
    if run("adjudicate", [PYTHON, "tools/adjudicate_caa_steering_campaign.py",
                          "--result", str(result), "--metadata", str(TRAINED / "metadata.json"),
                          "--independent-evidence", str(evidence),
                          "--out", str(evaluation)]).returncode != 0:
        say("stopped", why="adjudication refused")
        return 1
    if run("issue", [PYTHON, "tools/issue_cortex_migration_authority.py",
                     "--descriptor", "artifacts/migration/27b/active_descriptor.json",
                     "--out", str(authority), "steering",
                     "--metadata", str(TRAINED / "metadata.json"),
                     "--causal-evaluation", str(evaluation),
                     "--independent-evidence", str(evidence)]).returncode != 0:
        say("stopped", why="the authority would not issue")
        return 1

    if hashlib.sha256(result.read_bytes()).hexdigest() != result_digest:
        say("stopped", why="campaign result changed during preparation")
        return 1
    say("authority_prepared", authority=str(authority), result_sha256=result_digest,
        installed=False, runtime_restarted=False, fusion_measured=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

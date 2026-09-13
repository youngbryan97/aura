#!/usr/bin/env python3
"""tools/finish_the_steering_channel.py — carry the trained vectors the rest of the way.

Waits for the gradient training to finish, then does every step that stands
between a set of vectors and a channel a person's turn can see:

    seal -> campaign -> replay -> adjudicate -> issue -> install -> restart
    -> re-certify

FAIL-CLOSED at every step. The instance already serves with steering attached
at sixteen qualified layers; the only thing these trained vectors can do is
REPLACE that generation, and they only do it if the campaign passes on its own
terms. A campaign that does not pass leaves the running system exactly as it
is and writes down why.

Status lands in `artifacts/migration/27b/recovery/unattended/status.json` after
every step, so anybody -- a person, a later session -- can read where it got to
without reading the log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_unattended")

OUT = REPO / "artifacts/migration/27b/recovery/unattended"
STATUS = OUT / "status.json"
TRAINED = REPO / "training/vectors/gradient-trained"
PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan_wide.json"
POINTER = Path("/Users/bryan/.aura/live-source/training/fused-model/active.json")
DESCRIPTOR_SHA = "52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c"
CERTIFICATE = REPO / f"artifacts/fusion/{DESCRIPTOR_SHA}.json"
PYTHON = "/Users/bryan/.aura/live-source/.venv/bin/python"

#: Long enough for a 27B campaign that decodes 216 samples, and bounded so a
#: hung step is a reported failure rather than a wait nobody ends.
CAMPAIGN_TIMEOUT_S = 5 * 60 * 60
STEP_TIMEOUT_S = 45 * 60


def say(stage: str, **facts: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "at": time.time(), "at_iso": time.strftime("%Y-%m-%d %H:%M:%S"), **facts}
    history = []
    if STATUS.exists():
        try:
            history = json.loads(STATUS.read_text()).get("history", [])
        except ValueError:
            history = []
    STATUS.write_text(
        json.dumps({"latest": record, "history": [*history, record]}, indent=2) + "\n"
    )
    print(f"[{record['at_iso']}] {stage}: {facts}", flush=True)


def run(name: str, args: list[str], timeout: int = STEP_TIMEOUT_S) -> subprocess.CompletedProcess:
    say(f"{name}:start", argv=" ".join(args[1:4]))
    done = subprocess.run(  # noqa: S603
        args, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
    )
    (OUT / f"{name}.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    say(f"{name}:done", returncode=done.returncode, tail=done.stdout.strip().splitlines()[-1:])
    return done


def training_is_finished() -> bool:
    alive = subprocess.run(  # noqa: S603
        ["/usr/bin/pgrep", "-f", "train_steering_vectors_by_gradient"],
        capture_output=True, text=True, check=False,
    )
    return alive.returncode != 0


def main() -> int:
    say("waiting_for_training")
    waited = 0
    while not training_is_finished():
        time.sleep(60)
        waited += 60
        if waited > CAMPAIGN_TIMEOUT_S:
            say("gave_up_waiting_for_training", seconds=waited)
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

    result = REPO / "artifacts/migration/27b/recovery/campaign_result_trained.json"
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

    run("verdict", [PYTHON, "tools/read_the_steering_campaign.py", "--result", str(result)])
    verdict_path = REPO / "artifacts/migration/27b/recovery/campaign_verdict_trained.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
    say("campaign_read",
        causal=verdict.get("causal_effect_positive"),
        unmet=verdict.get("unmet_requirements"),
        layer_specificity=verdict.get("layer_assignment_specificity"))
    if not verdict.get("causal_effect_positive"):
        say("stopped", why="the trained vectors did not pass the campaign; the "
                           "running generation is untouched")
        return 2

    evidence = OUT / "independent_evidence.json"
    evaluation = OUT / "causal_evaluation.json"
    authority = OUT / "steering_authority.json"
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

    shutil.copy2(POINTER, OUT / "active.json.before-trained")
    install = run("install", [PYTHON, "-c", f'''
import json
from pathlib import Path
from core.governance_context import local_internal_governed_scope
from core.learning.cortex_generation_upgrade import build_migration_contract
from core.runtime.file_write_gateway import get_file_write_gateway
pointer = json.loads(Path({str(POINTER)!r}).read_text())
components = dict(pointer["migration_contract"]["components"])
components["steering"] = json.loads(Path({str(authority)!r}).read_text())
contract = build_migration_contract(pointer["artifact_descriptor"], components=components)
pointer["migration_contract"] = contract
payload = (json.dumps(pointer, indent=2, sort_keys=True) + "\\n").encode()
with local_internal_governed_scope("cortex_generation_upgrade"):
    get_file_write_gateway().write_bytes(Path({str(POINTER)!r}), payload, source="steering_authority_install")
print("installed", contract["migration_contract_sha256"][:16])
'''])
    if install.returncode != 0:
        say("stopped", why="the authority would not install; pointer unchanged")
        return 1

    CERTIFICATE.unlink(missing_ok=True)
    subprocess.run(["/usr/bin/pkill", "-f", "aura_main"], check=False)  # noqa: S603
    time.sleep(10)
    boot = OUT / "live_boot.log"
    with boot.open("wb") as handle:
        subprocess.Popen(  # noqa: S603
            ["/usr/bin/caffeinate", "-dims", PYTHON, "aura_main.py"],
            cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    say("instance_restarted", log=str(boot))

    # The worker measures once it has been idle long enough; give it room.
    for _ in range(40):
        time.sleep(30)
        if CERTIFICATE.exists():
            break
    if not CERTIFICATE.exists():
        say("finished", channel="no certificate measured yet; the worker writes "
                                "one when it has been idle long enough")
        return 0
    from core.consciousness.fusion_certificate import certificate_for

    certificate = certificate_for(DESCRIPTOR_SHA)
    say("finished",
        holds=bool(certificate and certificate.holds),
        why_not=None if certificate is None or certificate.holds else certificate.why_not(),
        alpha=getattr(certificate, "alpha", None),
        state_separation=getattr(certificate, "state_separation", None),
        margin_delta=getattr(certificate, "margin_delta", None),
        control_margin_delta=getattr(certificate, "control_margin_delta", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

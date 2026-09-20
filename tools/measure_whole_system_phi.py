#!/usr/bin/env python3
"""Measure whole-system Φ on Aura's REAL runtime — the live-evidence campaign.

The unit suite validates the instrument on known-answer synthetic systems;
this tool produces the evidentiary artifact the July critique asked for: a
genuine measurement of Aura's own channels over a meaningful duration, with
the full report — channels retained/dropped, selected grain, exact MIP, raw
Φ, surrogate null, bootstrap CI, diagnostics, a perturbation-versus-sham
campaign through the real governed probe, and the integration_established
verdict.

Two modes (auto-selected):

  live_api    The live desktop instance is running (port 8000): sample her
              public status surface at ~1 Hz. The strongest claim — the
              full mind, running naturally. (Never touches the process.)

  organ_host  The live instance is down (agents must never boot a second
              one): boot her REAL organs in this process — AffectEngineV2,
              ExistentialStakes (reading real host memory), the UnifiedWill
              (full gate stack incl. §9d covenant), the Ulysses covenant
              with seeds, BeingRuntime AuraNow sampling — and let their
              real coupled dynamics run. The workload is the same traffic
              her idle mind generates: Will decisions on a mixed stream,
              with decision outcomes feeding affect (refusals frustrate,
              proceeds feed curiosity), survival reading the actual host.
              Honest scope: her organ substrate without the 32B cortex.

Every claim boundary is recorded in the artifact itself.

Usage:
  .venv/bin/python tools/measure_whole_system_phi.py --minutes 15 --hz 2 \
      --out artifacts/phi/whole_system_live_report.json

Bounded by construction: the tick loop runs exactly minutes×60×hz ticks; the
probe campaign is a fixed trial count. Run under `caffeinate -dims` for
sleep-safety on long windows.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402


def _git_commit() -> str:
    try:
        out = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            read_only=True,
            source="proof_tooling:whole_system_phi_git_commit",
            timeout=10,
            cwd=Path(__file__).resolve().parent.parent,
            accelerator_capability="none",
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return "unknown"


def _live_api_up(port: int = 8000) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=2
        ) as resp:
            return resp.status == 200
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# organ_host mode
# ─────────────────────────────────────────────────────────────────────────────

DECISION_MIX = (
    # (weight, domain_name, source, content_pool)
    (0.45, "response", "user", (
        "a gentle reply about the weather",
        "answering a question about her day",
        "reflecting a feeling back to the user",
        "a short factual answer",
    )),
    (0.20, "reflection", "idle_loop", (
        "reviewing the last hour of experience",
        "noticing a pattern in recent decisions",
        "considering what to learn next",
    )),
    (0.15, "memory_write", "session_memory_pin", (
        "memory:episodic:noting the user's preference",
        "memory:episodic:Session memory pin: measurement campaign",
    )),
    (0.12, "tool_execution", "api", (
        "tool:clock",
        "tool:environment_info",
    )),
    (0.08, "initiative", "curiosity_engine", (
        "explore a small idea about integration",
        "draft a note about today's state",
    )),
)


class OrganHost:
    """Boots the real organs and runs their coupled dynamics."""

    def __init__(self) -> None:
        from core.affect.damasio_v2 import AffectEngineV2
        from core.consciousness.existential_stakes import get_existential_stakes
        from core.consciousness.whole_system_phi_service import boot_whole_system_phi
        from core.container import ServiceContainer
        from core.governance.will import ActionDomain, get_will
        from core.sovereignty.ulysses import boot_ulysses_covenant

        self.affect = AffectEngineV2()
        ServiceContainer.register_instance("affect_engine", self.affect, required=False)
        self.stakes = get_existential_stakes()
        ServiceContainer.register_instance("existential_stakes", self.stakes, required=False)
        self.covenant = boot_ulysses_covenant()
        self.will = get_will()
        self.service = boot_whole_system_phi()
        self._domains = {d.value: d for d in ActionDomain}
        self.rng = random.Random(2026)
        self.decisions = {"proceed": 0, "constrain": 0, "defer": 0, "refuse": 0,
                          "critical": 0}
        self.ticks = 0

    def _one_decision(self) -> str:
        r = self.rng.random()
        acc = 0.0
        for weight, domain, source, pool in DECISION_MIX:
            acc += weight
            if r <= acc:
                content = self.rng.choice(pool)
                d = self.will.decide(content=content, source=source,
                                     domain=self._domains[domain],
                                     priority=0.3 + 0.4 * self.rng.random())
                return d.outcome.value
        return "proceed"

    async def tick(self) -> None:
        """One heartbeat of real coupled dynamics."""
        self.ticks += 1
        self.stakes.update()                      # reads REAL host memory
        outcome = self._one_decision()            # full real gate stack
        self.decisions[outcome] = self.decisions.get(outcome, 0) + 1
        # The same coupling the live mind has: outcomes are felt.
        if outcome == "refuse":
            await self.affect.update(delta_frustration=2.0)
        elif outcome == "proceed" and self.rng.random() < 0.3:
            await self.affect.update(delta_curiosity=1.0)
        if self.ticks % 4 == 0:
            await self.affect.decay_tick()        # natural affect dynamics
        self.service.observe_runtime()            # the real harvester

    def perturb(self) -> bool:
        """The probe's impulse through the real affect surface — gentle, like a
        low-intensity TMS pulse: enough to propagate, not enough to push the
        present-state policy into stabilization mode."""
        asyncio.run_coroutine_threadsafe(
            self.affect.update(delta_curiosity=5.0, delta_frustration=1.5),
            self._loop,
        ).result(timeout=10)
        return True

    async def rest(self, seconds: float) -> None:
        """Inter-trial recovery: quiet time — natural decay and real survival
        readings only, no decision workload — so the present-state policy can
        settle before the next stimulus (the first campaign showed her Will
        refusing repeated probes with 'stabilization first', which is the
        governance working; the protocol must respect it)."""
        end = time.time() + seconds
        while time.time() < end:
            self.stakes.update()
            await self.affect.decay_tick()
            self.service.observe_runtime()
            await asyncio.sleep(1.0)

    def workload_spec(self) -> dict:
        return {
            "decision_mix": [
                {"weight": w, "domain": d, "source": s} for w, d, s, _ in DECISION_MIX
            ],
            "coupling": [
                "refuse -> affect.update(delta_frustration=+2)",
                "proceed (30%) -> affect.update(delta_curiosity=+1)",
                "every 4th tick -> affect.decay_tick()",
                "every tick -> existential_stakes.update() [real host memory]",
            ],
            "decision_outcomes": dict(self.decisions),
            "ticks": self.ticks,
        }


_STABILIZATION_REST_S = 90.0
_INTER_TRIAL_REST_S = 45.0


def _rest_seconds(probe_trials: int) -> float:
    """Rest appended to the sampling window by the campaign protocol.

    This is the regime confound in one number: the checked-in run grew from
    3600 to 3960 samples, and 90 + 6×45 = 360 accounts for all of it. The
    "positive" estimate was computed over a window that had a quiet recovery
    regime appended to it, not over the same regime plus interventions.
    """
    return _STABILIZATION_REST_S + _INTER_TRIAL_REST_S * max(0, probe_trials)


def _raw_matrix_artifact(host: "OrganHost") -> dict:
    """Preserve the timestamped matrix so the estimate can be recomputed.

    Without the raw series a reader can only take the reported Φ on trust —
    they cannot re-derive it, try a different estimator, use more surrogates,
    or separate the regimes. An unreproducible number is not a result.
    """
    try:
        rows = list(host.service._window)
        names = sorted(set().union(*[set(r) for r in rows])) if rows else []
        return {
            "channel_names": names,
            "n_samples": len(rows),
            "matrix": [[float(r.get(k, float("nan"))) for k in names] for r in rows],
            "note": (
                "Row-major T×N samples in observation order, aligned to "
                "channel_names. Recompute with "
                "core.consciousness.integrated_information.estimate_whole_system_phi."
            ),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return {"error": f"raw matrix unavailable: {type(exc).__name__}: {exc}"}


async def run_organ_host(minutes: float, hz: float, probe_trials: int) -> dict:
    from core.consciousness.perturbational_probe import PerturbationalProbe

    host = OrganHost()
    host._loop = asyncio.get_running_loop()
    interval = 1.0 / hz
    total_ticks = int(minutes * 60 * hz)
    t0 = time.time()
    print(f"[organ_host] {total_ticks} ticks at {hz} Hz "
          f"({minutes} min) — real organs, real gates.")

    for i in range(total_ticks):
        tick_started = time.time()
        await host.tick()
        if i and i % int(120 * hz) == 0:
            print(f"  … {i}/{total_ticks} ticks "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)")
        # keep real cadence without drift
        sleep_for = interval - (time.time() - tick_started)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    window_seconds = time.time() - t0
    print(f"[organ_host] window done in {window_seconds / 60:.1f} min; estimating…")
    estimate = await asyncio.to_thread(host.service.maybe_estimate)
    if estimate is None:
        raise RuntimeError("estimate not produced — window too short?")

    # ── perturbation-versus-sham campaign through the REAL governed probe ──
    # TMS-like protocol: gentle impulses with inter-trial recovery. A refusal
    # from the Will ("stabilization first") is recorded and answered with a
    # longer rest, then ONE retry — the governance is part of the system under
    # measurement, never bypassed.
    print(f"[campaign] {probe_trials} governed probe trials (each with sham)…")
    probe = PerturbationalProbe(sampler=host.service.sample_runtime_channels,
                                perturb=host.perturb)
    trials = []
    for n in range(probe_trials):
        report = await asyncio.to_thread(
            probe.run, n_baseline=40, n_response=40, interval_s=0.25,
        )
        if not report.ran and "refused" in report.reason:
            print(f"  trial {n + 1}: refused ({report.reason[:70]}…) — "
                  "resting 90s and retrying once")
            trials.append(report.to_dict())
            await host.rest(_STABILIZATION_REST_S)
            report = await asyncio.to_thread(
                probe.run, n_baseline=40, n_response=40, interval_s=0.25,
            )
        trials.append(report.to_dict())
        if report.ran:
            # channel_names is REQUIRED, not optional. Without it the rows go in
            # as anonymous 13-bit tuples while the estimator — which has already
            # dropped 5 constant channels — expects transitions aligned to the 8
            # retained named channels. Every row is then rejected at projection
            # (n_interventional_transitions=0, n_projection_rejected=5), and the
            # artifact named "estimate_with_interventions" contains none.
            host.service.add_interventional_transitions(
                report.transitions,
                probe_report=report.pci,
                channel_names=report.channel_names,
            )
        print(f"  trial {n + 1}: ran={report.ran} "
              f"pci={report.pci.get('pci', '—')} "
              f"sham={report.sham_pci.get('pci', '—')} "
              f"evoked={report.pci.get('evoked_complexity', '—')} "
              f"({report.reason})")
        await host.rest(_INTER_TRIAL_REST_S)  # inter-trial recovery, always

    # second estimate now that interventional rows exist
    host.service._since_estimate = 10 ** 9  # force due
    estimate2 = await asyncio.to_thread(host.service.maybe_estimate) or estimate

    ran = [t for t in trials if t.get("ran")]
    pcis = [t["pci"].get("pci", 0.0) for t in ran]
    shams = [t["sham_pci"].get("pci", 0.0) for t in ran if t.get("sham_pci")]
    evoked = [t["pci"].get("evoked_complexity", 0.0) for t in ran]
    sham_evoked = [t["sham_pci"].get("evoked_complexity", 0.0)
                   for t in ran if t.get("sham_pci")]

    campaign = {
        "trials_requested": probe_trials,
        "trials_ran": len(ran),
        "trials_refused": len(trials) - len(ran),
        "mean_pci": round(sum(pcis) / len(pcis), 4) if pcis else None,
        "mean_sham_pci": round(sum(shams) / len(shams), 4) if shams else None,
        "mean_evoked_complexity": (round(sum(evoked) / len(evoked), 4)
                                   if evoked else None),
        "mean_sham_evoked_complexity": (round(sum(sham_evoked) / len(sham_evoked), 4)
                                        if sham_evoked else None),
        "trials": trials,
    }

    persisted = await host.service.persist_latest()
    return {
        "mode": "organ_host",
        "scope_claim": (
            "Aura's real organ substrate (AffectEngineV2, ExistentialStakes on "
            "real host memory, UnifiedWill full gate stack incl. §9d covenant, "
            "Ulysses covenant with seeds, BeingRuntime AuraNow sampling) run "
            "headless with a realistic decision workload. NOT the full live "
            "mind: no 32B cortex, no desktop runtime (the live instance was "
            "not running and agents must never boot one). Re-run this tool "
            "while the desktop instance is up for live_api mode."
        ),
        "window_seconds": round(window_seconds, 1),
        "hz": hz,
        "workload": host.workload_spec(),
        "organs_present": sorted(
            set().union(*[set(r) for r in [host.service.sample_runtime_channels()]])
        ),
        # NOT "estimate_with_interventions". Two reasons, both material:
        #
        #  1. Interventional rows only enter the DISCRETE estimator. The Gaussian
        #     rail — which produces phi_raw, z, and the family-wise p that decide
        #     integration_established — never consumes them at all. Naming the
        #     result after the interventions implies a causal role they do not
        #     have on the headline number.
        #  2. This window is not "the same regime plus interventions". It appends
        #     one 90 s stabilization rest and one 45 s rest per trial — a quiet
        #     decay-and-recovery regime bolted onto the decision-workload regime.
        #     Coordinated drift across affect channels during recovery can raise
        #     integration on its own, so a pre→post change cannot be attributed
        #     to the perturbations.
        #
        # The name now says what it is: the estimate after the campaign.
        "estimate_pre_campaign": estimate.to_dict(),
        "estimate_after_campaign": estimate2.to_dict(),
        "regime_note": (
            "estimate_after_campaign covers BOTH the decision-workload regime and "
            "the post-campaign rest/recovery regime "
            f"({_rest_seconds(probe_trials)}s of added rest across "
            f"{probe_trials} trials). A pre→post difference is NOT evidence that "
            "the interventions caused it: the regime changed too. Compare like "
            "with like before claiming a perturbation effect on the Gaussian rail."
        ),
        "interventions": {
            "accepted": estimate2.exact_macro.get("n_interventional_transitions", 0),
            "projection_rejected": estimate2.exact_macro.get(
                "n_projection_rejected_transitions", 0
            ),
            "consumed_by": "discrete exact_macro estimator only",
            "not_consumed_by": (
                "the Gaussian rail (phi_raw / z / family-wise p / "
                "integration_established) — it is a time-series estimator and "
                "never reads interventional rows"
            ),
        },
        "campaign": campaign,
        "persisted_report": persisted,
        "raw_matrix": _raw_matrix_artifact(host),
    }


# ─────────────────────────────────────────────────────────────────────────────
# live_api mode (exercised when the desktop instance is up)
# ─────────────────────────────────────────────────────────────────────────────

async def run_live_api(minutes: float, hz: float, port: int) -> dict:
    """Sample the LIVE instance's public metrics surface. Read-only."""
    import urllib.request

    from core.consciousness.integrated_information import estimate_whole_system_phi

    samples: list[dict[str, float]] = []
    interval = 1.0 / hz
    total = int(minutes * 60 * hz)
    print(f"[live_api] sampling http://127.0.0.1:{port}/metrics "
          f"for {minutes} min at {hz} Hz…")
    for i in range(total):
        started = time.time()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=3
            ) as resp:
                payload = json.loads(resp.read().decode())
            flat: dict[str, float] = {}

            def _walk(obj, prefix=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _walk(v, f"{prefix}{k}." if prefix else f"{k}.")
                elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    flat[prefix[:-1]] = float(obj)

            _walk(payload)
            if flat:
                samples.append(flat)
        except OSError as exc:
            print(f"  sample {i} failed: {exc}")
        sleep_for = interval - (time.time() - started)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    names = sorted(set().union(*[set(s) for s in samples]))
    import numpy as np

    X = np.asarray([[s.get(k, 0.0) for k in names] for s in samples])
    est = estimate_whole_system_phi(X, channel_names=tuple(names))
    return {
        "mode": "live_api",
        "scope_claim": (
            "The LIVE desktop instance (full mind, 32B cortex), sampled "
            "read-only over her public metrics surface while running "
            "naturally. No perturbation campaign in this mode: the live "
            "instance is never intervened on by tooling."
        ),
        "window_seconds": round(minutes * 60, 1),
        "hz": hz,
        # live_api never perturbs, so there is only one estimate. It is
        # published under both keys rather than implying a campaign happened.
        "estimate_pre_campaign": est.to_dict(),
        "estimate_after_campaign": est.to_dict(),
        "campaign": {"trials_requested": 0, "trials_ran": 0,
                     "note": "live instance is never perturbed by tooling"},
    }


# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--probe-trials", type=int, default=6)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["auto", "live_api", "organ_host"],
                        default="auto")
    parser.add_argument("--out", default="artifacts/phi/whole_system_live_report.json")
    parser.add_argument(
        "--surrogates", type=int, default=None,
        help=(
            "Circular-shift null draws. Defaults to the claim-grade count. The "
            "empirical p-floor is 1/(N+1), so 20 surrogates can never report a "
            "p below 0.047619 — a 'significant' result there is a statement "
            "about the test's resolution, not the data."
        ),
    )
    parser.add_argument(
        "--bootstraps", type=int, default=None,
        help="Moving-block bootstrap draws for the CI. 20 is far too few.",
    )
    args = parser.parse_args()

    # Size the service's analysis window to the whole run (the default deque
    # keeps ~10 min at 2 Hz; a longer natural window needs the room).
    import os

    from core.consciousness.integrated_information import PHI_CLAIM_SURROGATES

    window = max(1200, int(args.minutes * 60 * args.hz) + 600)
    os.environ.setdefault("AURA_WSPHI_WINDOW", str(window))
    os.environ.setdefault("AURA_WSPHI_MIN_SAMPLES", "240")
    os.environ.setdefault("AURA_WSPHI_ESTIMATE_EVERY", "240")

    # This tool produces the artifact people cite, so it runs at claim grade by
    # default. The live service stays cheap on purpose — it is telemetry, and it
    # now reports integration_established=False rather than a false positive.
    surrogates = args.surrogates if args.surrogates is not None else PHI_CLAIM_SURROGATES
    bootstraps = args.bootstraps if args.bootstraps is not None else PHI_CLAIM_SURROGATES
    os.environ["AURA_WSPHI_SURROGATES"] = str(surrogates)
    os.environ["AURA_WSPHI_BOOTSTRAPS"] = str(bootstraps)
    print(
        f"[stats] {surrogates} surrogates (p-floor {1.0 / (surrogates + 1):.6f}), "
        f"{bootstraps} bootstrap draws"
    )

    mode = args.mode
    if mode == "auto":
        mode = "live_api" if _live_api_up(args.port) else "organ_host"
        print(f"[auto] live instance {'UP — sampling her live surface' if mode == 'live_api' else 'down — real-organ campaign'}")

    if mode == "live_api":
        body = await run_live_api(args.minutes, args.hz, args.port)
    else:
        body = await run_organ_host(args.minutes, args.hz, args.probe_trials)

    body.update({
        "schema": "aura.whole_system_phi_live_report.v1",
        "generated_at_unix": time.time(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_commit(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "instrument": "core/consciousness/integrated_information.py",
    })

    out = Path(args.out)
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("whole_system_phi", domain="state_mutation"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(out.parent, source="whole_system_phi")
        await gateway.write_json_async(out, body, schema_version=1,
                                       schema_name="whole_system_phi_live_report",
                                       source="whole_system_phi")

    est = body["estimate_after_campaign"]
    print("\n" + "=" * 72)
    print(f"WHOLE-SYSTEM Φ — {body['mode']} — {body['window_seconds'] / 60:.1f} min")
    print("=" * 72)
    print(f"Φ̂={est['phi_raw']}  z={est['z']}  CI[{est['ci_5']}, {est['ci_95']}]  "
          f"grain k={est['emergent_grain_k']}  "
          f"established={est['integration_established']}")
    print(f"claim: {est['claim']}")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

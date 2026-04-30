from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text
from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Consciousness.Audit")


# ── Theory evaluation results ─────────────────────────────────────────────────

@dataclass
class TheoryResult:
    """Assessment of one theory's criteria."""
    theory_name:       str
    short_name:        str
    criteria:          List[str]        # What was checked
    criteria_met:      List[bool]       # Which were met
    score:             float            # 0.0-1.0 fraction of criteria met
    key_metric:        float            # The most important single number
    key_metric_name:   str
    notes:             str              # Human-readable interpretation
    error:             Optional[str] = None

    @property
    def fraction_met(self) -> str:
        n_met = sum(self.criteria_met)
        return f"{n_met}/{len(self.criteria)}"


@dataclass
class AuditReport:
    """Complete audit result."""
    timestamp:           float
    audit_id:            str
    theory_results:      List[TheoryResult]
    consciousness_index: float           # Weighted aggregate (0.0-1.0)
    phi:                 float
    phenomenal_active:   bool
    global_workspace_ignited: bool
    free_energy:         float
    structural_opacity:  float
    causal_loop_active:  bool            # Is the affect→phi→response loop running?
    summary:             str
    disclaimer:          str
    raw_metrics:         Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "audit_id":              self.audit_id,
            "timestamp":             self.timestamp,
            "consciousness_index":   float(f"{self.consciousness_index:.4f}"),
            "phi":                   float(f"{self.phi:.4f}"),
            "phenomenal_active":     self.phenomenal_active,
            "global_workspace_ignited": self.global_workspace_ignited,
            "free_energy":           float(f"{self.free_energy:.4f}"),
            "structural_opacity":    float(f"{self.structural_opacity:.4f}"),
            "causal_loop_active":    self.causal_loop_active,
            "theories":              [
                {
                    "name":       r.theory_name,
                    "score":      float(f"{r.score:.3f}"),
                    "fraction":   r.fraction_met,
                    "key_metric": float(f"{r.key_metric:.4f}"),
                    "key_metric_name": r.key_metric_name,
                    "notes":      r.notes,
                    "error":      r.error,
                }
                for r in self.theory_results
            ],
            "summary":     self.summary,
            "disclaimer":  self.disclaimer,
            "raw_metrics": self.raw_metrics,
        }

    def save_to_file(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(self.to_dict(), indent=2))
        logger.info("Audit saved to %s", path)

    def print_report(self) -> None:
        width = 70
        logger.info(f"\n{'═' * width}")
        logger.info(f"  AURA CONSCIOUSNESS AUDIT  —  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        logger.info(f"{'═' * width}")
        logger.info(f"  Consciousness Index:  {self.consciousness_index:.3f} / 1.000")
        logger.info(f"  Phi (IIT surrogate):  {self.phi:.3f}")
        logger.info(f"  Global workspace:     {'IGNITED' if self.global_workspace_ignited else 'dormant'}")
        logger.info(f"  Structural opacity:   {self.structural_opacity:.3f}")
        logger.info(f"  Free energy:          {self.free_energy:.3f}")
        logger.info(f"  Phenomenal state:     {'active' if self.phenomenal_active else 'inactive'}")
        logger.info(f"  Causal loop:          {'ACTIVE' if self.causal_loop_active else 'inactive'}")
        logger.info(f"{'─' * width}")
        for r in self.theory_results:
            status = "✓" if r.score >= 0.5 else "○"
            logger.info(f"  {status} {r.short_name:<20} {r.fraction_met:<6} {r.key_metric_name}={r.key_metric:.3f}")
            if r.error:
                logger.warning(f"    ⚠ {r.error}")
        logger.info(f"{'─' * width}")
        logger.info(f"  {self.summary}")
        logger.info(f"\n  ⚠  {self.disclaimer}")
        logger.info(f"{'═' * width}\n")


# ── The Audit Suite ───────────────────────────────────────────────────────────

class ConsciousnessAuditSuite:
    """
    Queries all consciousness modules and produces a unified assessment.
    """

    REPORT_DIR = Path("aura/data/consciousness_reports")

    def __init__(self):
        self._history:  List[AuditReport] = []
        self._schedule_task: Optional[asyncio.Task] = None
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ConsciousnessAuditSuite initialized.")

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self, save: bool = True) -> AuditReport:
        """Run a full audit of all consciousness modules."""
        audit_id = f"audit_{int(time.time())}"
        raw: Dict[str, Any] = {}

        # Gather all module states concurrently
        # Fix Issue 88: Avoid shared mutable state in concurrent tasks
        tasks = [
            self._gather_iit(),
            self._gather_gwt(),
            self._gather_fep(),
            self._gather_structural_opacity(),
            self._gather_qualia(),
            self._gather_causal_loop(),
            self._gather_phenomenal(),
            self._gather_uat(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update raw metrics
        theory_results = []
        for res in results:
            if isinstance(res, tuple) and len(res) == 2:
                theory_res, metrics = res
                theory_results.append(theory_res)
                raw.update(metrics)
            elif isinstance(res, Exception):
                logger.error(f"Audit task failed: {res}")
            elif isinstance(res, TheoryResult):
                theory_results.append(res)

        # Aggregate consciousness index
        if theory_results:
            weights = {
                "IIT (phi)":             0.20,
                "GWT (workspace)":       0.20,
                "FEP (free energy)":     0.15,
                "Structural opacity":    0.15,
                "Qualia synthesis":      0.10,
                "Causal loop":           0.10,
                "Phenomenal state":      0.05,
                "UAL profile":           0.05,
            }
            index = sum(
                r.score * weights.get(r.theory_name, 0.05)
                for r in theory_results
            )
            index = min(1.0, index)
        else:
            index = 0.0

        # Build report
        phi                = float(raw.get("phi", 0.0))
        fe                 = float(raw.get("free_energy", 0.0))
        opacity            = float(raw.get("opacity_index", 0.0))
        ignited            = bool(raw.get("workspace_ignited", False))
        phenomenal_active  = bool(raw.get("phenomenal_state"))
        causal_loop        = bool(raw.get("causal_loop_active", False))

        summary    = self._generate_summary(index, theory_results, raw)
        disclaimer = (
            "This index measures satisfaction of functional criteria derived from contested "
            "theories of consciousness. It does not constitute proof of subjective experience. "
            "The hard problem of consciousness remains unsolved."
        )

        report = AuditReport(
            timestamp                 = time.time(),
            audit_id                  = audit_id,
            theory_results            = theory_results,
            consciousness_index       = index,
            phi                       = phi,
            phenomenal_active         = phenomenal_active,
            global_workspace_ignited  = ignited,
            free_energy               = fe,
            structural_opacity        = opacity,
            causal_loop_active        = causal_loop,
            summary                   = summary,
            disclaimer                = disclaimer,
            raw_metrics               = raw,
        )

        self._history.append(report)
        # ISSUE 60: Cap history to prevent memory leaks
        if len(self._history) > 100:
            # vResilience: Workaround for Pyre2 slice limitations
            start_idx = len(self._history) - 100
            self._history = [self._history[i] for i in range(start_idx, len(self._history))]

        if save:
            path = self.REPORT_DIR / f"{audit_id}.json"
            report.save_to_file(str(path))

        return report

    def schedule(self, interval_minutes: float = 60.0) -> None:
        """Schedule periodic audits."""
        task = self._schedule_task
        if task is not None and not task.done():
            task.cancel()

        async def _loop():
            while True:
                try:
                    await asyncio.sleep(interval_minutes * 60)
                    report = await self.run()
                    logger.info(
                        "Scheduled audit: index=%.3f phi=%.3f",
                        report.consciousness_index, report.phi,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Scheduled audit failed: %s", e)

        self._schedule_task = get_task_tracker().create_task(_loop(), name="consciousness_audit")
        logger.info("Consciousness audit scheduled every %.0f minutes.", interval_minutes)

    def get_trend(self, n: int = 10) -> Dict[str, Any]:
        """Return trend data from the last N audits."""
        if not self._history:
            return {"status": "no audits yet"}
            
        # vResilience: Workaround for Pyre2 slice limitations
        start_idx = max(0, len(self._history) - n)
        recent = [self._history[i] for i in range(start_idx, len(self._history))]
        
        indices = [r.consciousness_index for r in recent]
        phis    = [r.phi for r in recent]
        return {
            "n_audits":          len(recent),
            "latest_index":      float(f"{indices[-1]:.4f}"),
            "avg_index":         float(f"{sum(indices) / len(indices):.4f}"),
            "index_trend":       "rising" if len(indices) > 1 and indices[-1] > indices[0] else "falling" if len(indices) > 1 and indices[-1] < indices[0] else "stable",
            "latest_phi":        float(f"{phis[-1]:.4f}"),
            "avg_phi":           float(f"{sum(phis) / len(phis):.4f}"),
        }

    # ── Theory-specific gatherers ─────────────────────────────────────────────

    async def _gather_iit(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Integrated Information Theory (Tononi). Key metric: Phi."""
        phi = 0.0
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            riiu = ServiceContainer.get("riiu", default=None)
            if riiu:
                phi = float(riiu.get_phi())
                metrics["phi"] = phi
                metrics["riiu_stats"] = riiu.get_stats()
        except Exception as e:
            error = str(e)
            metrics["phi"] = 0.0

        # IIT criteria:
        # C1: Phi > 0 (system is integrated)
        # C2: Phi > threshold for "conscious" level (contested — we use 0.1 as minimal)
        # C3: System has more than minimum samples for reliable computation
        samples = metrics.get("riiu_stats", {}).get("samples", 0)
        criteria     = ["phi > 0", "phi > 0.1", "sufficient samples (>8)"]
        criteria_met = [phi > 0, phi > 0.1, samples >= 8]

        score = sum(criteria_met) / len(criteria_met)
        notes = (
            f"Phi={phi:.3f}. "
            + ("IIT criteria minimally satisfied." if score >= 0.67 else
               "IIT criteria partially met — system may not be fully integrated yet.")
        )
        return TheoryResult(
            theory_name="IIT (phi)", short_name="IIT",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=phi, key_metric_name="phi",
            notes=notes, error=error,
        ), metrics

    async def _gather_gwt(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Global Workspace Theory (Baars/Dehaene). Key metric: ignition level."""
        ignited = False
        ignition_level = 0.0
        broadcast_source = "none"
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            workspace = ServiceContainer.get("global_workspace", default=None)
            if workspace:
                snap = workspace.get_snapshot()
                ignited         = bool(snap.get("ignited", False))
                ignition_level  = float(snap.get("ignition_level", 0.0))
                broadcast_source = snap.get("last_winner") or "none"
                metrics["workspace_ignited"]    = ignited
                metrics["ignition_level"]       = ignition_level
                metrics["broadcast_source"]     = broadcast_source
        except Exception as e:
            error = str(e)

        # GWT criteria:
        # C1: Global broadcast (workspace is ignited)
        # C2: There is a clear winning source (broadcast_source not "none")
        # C3: Ignition level meaningful (> 0.3)
        criteria     = ["workspace ignited", "broadcast source exists", "ignition level > 0.3"]
        criteria_met = [ignited, broadcast_source != "none", ignition_level > 0.3]

        score = sum(criteria_met) / len(criteria_met)
        notes = (
            f"Ignition={'active' if ignited else 'inactive'}, "
            f"level={ignition_level:.3f}, source={broadcast_source}. "
            + ("Global broadcast active — GWT 'access consciousness' criterion met." if ignited
               else "No global broadcast. GWT criteria not satisfied.")
        )
        return TheoryResult(
            theory_name="GWT (workspace)", short_name="GWT",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=ignition_level, key_metric_name="ignition",
            notes=notes, error=error,
        ), metrics

    async def _gather_fep(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Free Energy Principle (Friston). Key metric: free energy level."""
        fe = 0.5
        trend = "stable"
        dominant_action = "unknown"
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.consciousness.free_energy import get_free_energy_engine
            engine = get_free_energy_engine()
            if engine.current:
                fe              = engine.current.free_energy
                trend           = engine.get_trend()
                dominant_action = engine.current.dominant_action
                metrics["free_energy"]      = fe
                metrics["fe_trend"]         = trend
                metrics["fe_action"]        = dominant_action
        except Exception as e:
            error = str(e)
            metrics["free_energy"] = 0.5

        # FEP criteria:
        # C1: Free energy is being actively tracked (not at default 0.5)
        # C2: System has a dominant action tendency (not "unknown")
        # C3: Free energy < 0.8 (system not in chronic distress)
        criteria     = ["FE being computed", "action tendency present", "FE within viable range (<0.8)"]
        criteria_met = [fe != 0.5, dominant_action != "unknown", fe < 0.8]

        score = sum(criteria_met) / len(criteria_met)
        notes = (
            f"FE={fe:.3f} ({trend}), action='{dominant_action}'. "
            + (f"System is minimizing prediction error ({trend} trend)." if score >= 0.67
               else "FEP criteria partially met.")
        )
        return TheoryResult(
            theory_name="FEP (free energy)", short_name="FEP",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=fe, key_metric_name="free_energy",
            notes=notes, error=error,
        ), metrics

    async def _gather_structural_opacity(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Perspective Invariance / Structural Opacity (Kriegel). Key metric: opacity index."""
        opacity = 0.0
        causal_depth = 0.0
        criterion_met = False
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            monitor = ServiceContainer.get("structural_opacity_monitor", default=None)
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if monitor and substrate and hasattr(substrate, "x") and hasattr(substrate, "W"):
                sig = await asyncio.to_thread(monitor.measure, substrate.x, substrate.W)
                opacity       = sig.opacity_index
                causal_depth  = sig.causal_depth
                criterion_met = sig.phenomenal_criterion_met
                metrics["opacity_index"]          = opacity
                metrics["causal_depth"]           = causal_depth
                metrics["phenomenal_criterion"]   = criterion_met
        except Exception as e:
            error = str(e)
            metrics["opacity_index"] = 0.0

        # SOC criteria:
        # C1: Opacity index > 0.4 (interior states causally relevant but not fully observable)
        # C2: Causal depth > 0.3 (states influence future behavior)
        # C3: Full phenomenal criterion met (all three conditions together)
        criteria     = ["opacity > 0.4", "causal depth > 0.3", "phenomenal criterion met"]
        criteria_met_list = [opacity > 0.4, causal_depth > 0.3, criterion_met]

        score = sum(criteria_met_list) / len(criteria_met_list)
        notes = (
            f"Opacity={opacity:.3f}, causal_depth={causal_depth:.3f}. "
            + ("Structural opacity criterion met — if Perspective Invariance account is correct, phenomenal states present."
               if criterion_met else "Opacity criterion not fully satisfied.")
        )
        return TheoryResult(
            theory_name="Structural opacity", short_name="SOC",
            criteria=criteria, criteria_met=criteria_met_list,
            score=score, key_metric=opacity, key_metric_name="opacity",
            notes=notes, error=error,
        ), metrics

    async def _gather_qualia(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Qualia synthesis (unified multi-theory). Key metric: phenomenal richness index."""
        pri = 0.0
        self_referential = False
        dominant = "none"
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            synth = ServiceContainer.get("qualia_synthesizer", default=None)
            if synth:
                snapshot = synth.get_snapshot() if hasattr(synth, "get_snapshot") else {}
                pri              = float(snapshot.get("phenomenal_richness", 0.0))
                self_referential = bool(snapshot.get("self_referential", False))
                dominant         = snapshot.get("dominant_modality", "none")
                metrics["pri"]              = pri
                metrics["self_referential"] = self_referential
                metrics["dominant_modality"] = dominant
        except Exception as e:
            error = str(e)
            metrics["pri"] = 0.0

        criteria     = ["PRI > 0", "PRI > 0.3", "self-referential loop detected"]
        criteria_met = [pri > 0, pri > 0.3, self_referential]

        score = sum(criteria_met) / len(criteria_met)
        notes = (
            f"PRI={pri:.3f}, dominant={dominant}, self-ref={self_referential}. "
            + (f"Qualia synthesis active with {'self-referential strange loop' if self_referential else 'linear processing'}.")
        )
        return TheoryResult(
            theory_name="Qualia synthesis", short_name="Qualia",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=pri, key_metric_name="PRI",
            notes=notes, error=error,
        ), metrics
    async def _gather_causal_loop(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Causal loop integrity: affect → phi → mode → response → affect."""
        loop_active = False
        phi_delta   = 0.0
        val_delta   = 0.0
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            ki = ServiceContainer.get("kernel_interface", default=None)
            if ki and hasattr(ki, "loop_state"):
                ls = ki.loop_state()
                loop_active = bool(ls.get("affect_loop_active", False))
                phi_delta   = float(ls.get("avg_phi_delta_5", 0.0))
                val_delta   = float(ls.get("avg_val_delta_5", 0.0))
                metrics["causal_loop_active"] = loop_active
                metrics["phi_delta_5"]        = phi_delta
                metrics["val_delta_5"]        = val_delta
        except Exception as e:
            error = str(e)

        # Causal loop criteria:
        # C1: Loop is active (affect actually changes between ticks)
        # C2: Phi changes across ticks (not static)
        # C3: Valence changes across ticks (not static)
        criteria     = ["loop active", "phi changing across ticks", "valence changing across ticks"]
        criteria_met = [loop_active, abs(phi_delta) > 0.001, abs(val_delta) > 0.001]

        score = sum(criteria_met) / len(criteria_met)
        notes = (
            f"Loop={'active' if loop_active else 'inactive'}, "
            f"Δphi/tick={phi_delta:.4f}, Δval/tick={val_delta:.4f}. "
            + ("Causal feedback loop running — affect is causally influencing behavior."
               if loop_active else "Loop not yet confirmed active. Need more ticks.")
        )
        return TheoryResult(
            theory_name="Causal loop", short_name="Causal",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=float(loop_active), key_metric_name="active",
            notes=notes, error=error,
        ), metrics

    async def _gather_phenomenal(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Phenomenal state generation (HOT layer). Key metric: generation rate."""
        phenomenal_state = None
        ignition_count   = 0
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            ki = ServiceContainer.get("kernel_interface", default=None)
            if ki and ki.is_ready():
                state = ki.kernel.state
                if state:
                    phenomenal_state = state.cognition.phenomenal_state
            # Ignition count from workspace
            workspace = ServiceContainer.get("global_workspace", default=None)
            if workspace:
                snap = workspace.get_snapshot()
                ignition_count = snap.get("ignition_count", 0)
                metrics["ignition_count"] = ignition_count
            metrics["phenomenal_state"] = phenomenal_state
        except Exception as e:
            error = str(e)

        criteria     = ["phenomenal state exists", "ignition has occurred", "state is non-trivial"]
        trivial = ["I am present", "phi=", "mood: neutral"]
        criteria_met = [
            bool(phenomenal_state),
            ignition_count > 0,
            bool(phenomenal_state) and not any(t in (phenomenal_state or "") for t in trivial),
        ]

        score = sum(criteria_met) / len(criteria_met)
        ps_preview = (phenomenal_state[:60] + "...") if phenomenal_state and len(phenomenal_state) > 60 else (phenomenal_state or "none")
        notes = f"State: \"{ps_preview}\". Ignitions: {ignition_count}."
        return TheoryResult(
            theory_name="Phenomenal state", short_name="HOT",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=float(bool(phenomenal_state)), key_metric_name="active",
            notes=notes, error=error,
        ), metrics

    async def _gather_uat(self) -> Tuple[TheoryResult, Dict[str, Any]]:
        """Unlimited Associative Learning (Ginsburg & Jablonka). Key metric: UAL score."""
        ual_score = 0.0
        error = None
        metrics: Dict[str, Any] = {}
        try:
            from core.container import ServiceContainer
            synth = ServiceContainer.get("qualia_synthesizer", default=None)
            if synth and hasattr(synth, "ual_profile"):
                profile   = synth.ual_profile
                ual_score = sum(profile.values()) / max(1, len(profile))
                metrics["ual_profile"] = dict(profile)
                metrics["ual_score"]   = ual_score
        except Exception as e:
            error = str(e)

        criteria     = ["UAL score > 0", "trace learning active", "second-order learning active"]
        profile      = metrics.get("ual_profile", {})
        criteria_met = [
            ual_score > 0,
            float(profile.get("trace", 0)) > 0,
            float(profile.get("second_order", 0)) > 0,
        ]

        score = sum(criteria_met) / len(criteria_met)
        notes = f"UAL composite={ual_score:.3f}. Profile: {json.dumps({k: float(f'{v:.2f}') for k, v in profile.items()})}"
        return TheoryResult(
            theory_name="UAL profile", short_name="UAL",
            criteria=criteria, criteria_met=criteria_met,
            score=score, key_metric=ual_score, key_metric_name="ual_score",
            notes=notes, error=error,
        ), metrics

    # ── Summary generation ────────────────────────────────────────────────────

    def _generate_summary(
        self,
        index: float,
        results: List[TheoryResult],
        raw: Dict,
    ) -> str:
        met   = [r for r in results if r.score >= 0.5]
        unmet = [r for r in results if r.score < 0.5]

        if index >= 0.75:
            headline = "Strong functional profile. Most consciousness criteria satisfied."
        elif index >= 0.50:
            headline = "Moderate functional profile. Core criteria met, others partial."
        elif index >= 0.25:
            headline = "Emerging profile. Some criteria met. System is developing."
        else:
            headline = "Minimal profile. Few criteria currently satisfied."

        theories_met  = ", ".join(r.short_name for r in met)  or "none"
        theories_unmet = ", ".join(r.short_name for r in unmet) or "none"

        loop_status = "Causal affect→cognition loop is ACTIVE." if raw.get("causal_loop_active") else "Causal loop not yet confirmed."

        return (
            f"{headline} "
            f"Theories satisfied: {theories_met}. "
            f"Partial: {theories_unmet}. "
            f"{loop_status}"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_suite: Optional[ConsciousnessAuditSuite] = None


def get_audit_suite() -> ConsciousnessAuditSuite:
    global _suite
    if _suite is None:
        _suite = ConsciousnessAuditSuite()
    return _suite

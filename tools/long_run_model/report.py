from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import json
from pathlib import Path
from typing import Dict

from .simulate import ForecastRunSummary


def render_markdown(summary: ForecastRunSummary) -> str:
    lines = [
        "# Aura Long-Run Forecast",
        "",
        f"- Profile: `{summary.profile['name']}`",
        f"- Hardware: `{summary.registry['hardware_model']}`",
        f"- RAM: `{summary.registry['total_ram_gb']} GB`",
        f"- Storage: `{summary.registry['total_storage_gb']} GB`",
        "",
        "## Registry",
        "",
        f"- Liquid-time clamp: `{summary.registry['liquid_time_clamp_s']}s`",
        f"- Circadian update interval: `{summary.registry['circadian_update_interval_s']}s`",
        f"- Episodic evaluation interval: `{summary.registry['episodic_eval_interval_s']}s`",
        f"- LTM consolidation interval: `{summary.registry['ltm_consolidation_interval_s']}s`",
        f"- Conversation prune interval: `{summary.registry['conversation_prune_interval_s']}s`",
        f"- Vector prune interval: `{summary.registry['vector_prune_interval_s']}s`",
        f"- Backup interval: `{summary.registry['backup_interval_s']}s`",
        f"- Backup wired: `{summary.registry['backup_wired']}`",
        f"- Lock watchdog auto-repair: `{summary.registry['lock_watchdog_auto_repair']}`",
        f"- State queue repair enabled: `{summary.registry['state_queue_repair_enabled']}`",
        "",
        "## Checkpoints",
        "",
    ]

    for checkpoint in summary.checkpoints:
        lines.extend([
            f"### {checkpoint.horizon}",
            "",
            f"- Phi: `{checkpoint.organism_summary['phi']}`",
            f"- Affect: `valence={checkpoint.organism_summary['affect']['valence']:+.4f}` "
            f"`arousal={checkpoint.organism_summary['affect']['arousal']:.4f}` "
            f"`curiosity={checkpoint.organism_summary['affect']['curiosity']:.4f}`",
            f"- Motivation: `energy={checkpoint.organism_summary['motivation']['energy']:.2f}` "
            f"`curiosity={checkpoint.organism_summary['motivation']['curiosity']:.2f}` "
            f"`social={checkpoint.organism_summary['motivation']['social']:.2f}`",
            f"- Pending initiatives: `{checkpoint.organism_summary['initiative']['pending_initiatives']}`",
            f"- Research cycles: `{checkpoint.maintenance_summary['research_cycles']}`",
            f"- Dreams / sandbox: `{checkpoint.maintenance_summary['subconscious_dreams']}` / "
            f"`{checkpoint.maintenance_summary['subconscious_sandbox_runs']}`",
            f"- Maintenance: `vacuums={checkpoint.maintenance_summary['db_vacuums']}` "
            f"`backups={checkpoint.maintenance_summary['backups']}` "
            f"`repair_actions={checkpoint.maintenance_summary['scheduled_repair_actions']}`",
            f"- Storage: `turns={checkpoint.storage_summary['turns_total']}` "
            f"`episodic={checkpoint.storage_summary['episodic_memories']}` "
            f"`vector={checkpoint.storage_summary['vector_memories']}` "
            f"`ltm={checkpoint.storage_summary['long_term_memories']}` "
            f"`backup_archives={checkpoint.storage_summary['backup_archives_retained']}`",
            f"- Pressure: `tasks={checkpoint.pressure_summary['task_count']}` "
            f"`queue={checkpoint.pressure_summary['queue_depth']}` "
            f"`tick_ms={checkpoint.pressure_summary['mean_tick_ms']}` "
            f"`rss_gb={checkpoint.pressure_summary['rss_gb']}` "
            f"`lock_age={checkpoint.pressure_summary['lock_hold_age_s']}`",
            "",
            "Cliffs:",
        ])
        for cliff in checkpoint.cliff_summary:
            lines.append(f"- `{cliff.name}`: {'reached' if cliff.reached else 'not yet reached'} — {cliff.impact}")
        if checkpoint.risk_summary:
            lines.append("")
            lines.append("Risks:")
            for risk in checkpoint.risk_summary:
                lines.append(f"- `[{risk.severity}] {risk.subsystem}` {risk.title}: {risk.description}")
        lines.append("")

    lines.extend(["## Remediation Backlog", ""])
    if not summary.remediation_backlog:
        lines.append("- No remediation backlog generated for the selected assumptions.")
    else:
        for item in summary.remediation_backlog:
            lines.append(
                f"- `[{item['severity']}] {item['subsystem']}` {item['title']} "
                f"(seen at {', '.join(item['horizons'])}, count={item['count']}): {item['recommendation']}"
            )
    lines.append("")

    lines.extend(["## Risk Ledger", ""])
    if not summary.risk_ledger:
        lines.append("- No active risks forecast under the selected assumptions.")
    else:
        for risk in summary.risk_ledger:
            lines.append(f"- `[{risk.severity}] {risk.subsystem}` @ `{risk.horizon}`: {risk.title}")
    lines.append("")
    return "\n".join(lines)


def write_report_bundle(summary: ForecastRunSummary, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "forecast_report.md"
    summary_path = output_dir / "forecast_summary.json"
    risk_path = output_dir / "risk_ledger.json"
    remediation_path = output_dir / "remediation_backlog.json"

    atomic_write_text(markdown_path, render_markdown(summary), encoding="utf-8")
    atomic_write_text(summary_path, json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(risk_path, 
        json.dumps([risk for risk in summary.to_dict()["risk_ledger"]], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    atomic_write_text(remediation_path, 
        json.dumps(summary.to_dict()["remediation_backlog"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "markdown": str(markdown_path),
        "summary_json": str(summary_path),
        "risk_json": str(risk_path),
        "remediation_json": str(remediation_path),
    }

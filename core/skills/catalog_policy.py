"""Canonical exposure and effect policy for Aura's skill catalog.

Discovery answers what the source tree declares.  This module answers whether a
declaration belongs on the live capability surface and which authority class it
must traverse.  Keeping that decision executable and exhaustive prevents a new
skill from becoming user-visible merely because it happens to define ``name``.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_EFFECT_SCOPES = frozenset(
    {
        "external_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "privileged_mutation",
        "pure_compute",
        "read_only",
        "read_write_artifacts",
        "sandboxed_compute",
        "state_mutation",
        "status",
    }
)

_AUTHORITY_BY_EFFECT_SCOPE = {
    "status": "observe",
    "read_only": "observe",
    "pure_compute": "observe",
    "sandboxed_compute": "bounded_compute",
    "state_mutation": "state_write",
    "external_io": "external_effect",
    "read_write_artifacts": "artifact_write",
    "foreground_desktop_control": "foreground_control",
    "foreground_browser_dialogue": "foreground_control",
    "privileged_mutation": "privileged",
}


# Skills may declare ``effect_scope`` on the class.  This table is the explicit
# policy for older declarations that have not yet moved that metadata beside the
# implementation.  Unknown names are not guessed: they must declare a valid
# scope before discovery will admit them.
SKILL_EFFECT_SCOPES: dict[str, str] = {
    "ManageAbilities": "state_mutation",
    "add_belief": "state_mutation",
    "auto_refactor": "privileged_mutation",
    "branching_futures": "pure_compute",
    "browser_action": "external_io",
    "build_app": "read_write_artifacts",
    "clock": "status",
    "code_repl": "sandboxed_compute",
    "coding_skill": "pure_compute",
    "cognitive_trainer": "state_mutation",
    "computer_use": "foreground_desktop_control",
    # Presses keys on the foreground desktop over many cycles, so it carries
    # the same authority as a single computer_use action rather than a
    # gentler one for being a loop.
    "pursue_on_screen": "foreground_desktop_control",
    "curiosity": "state_mutation",
    "delegate_shard": "external_io",
    "deploy_ghost_probe": "external_io",
    "desktop_task": "foreground_desktop_control",
    "dream_sleep": "state_mutation",
    "email_adapter": "external_io",
    "embodiment": "external_io",
    "environment_info": "read_only",
    "evolution_status": "read_only",
    "execute_nethack_action": "state_mutation",
    "file_operation": "state_mutation",
    "force_dream_cycle": "state_mutation",
    "free_search": "read_only",
    "grounded_search": "read_only",
    "image_gen": "read_write_artifacts",
    "improve_own_code": "privileged_mutation",
    "install_package": "privileged_mutation",
    "inter_agent_comm": "external_io",
    "internal_sandbox": "sandboxed_compute",
    "knowledge_base": "state_mutation",
    "listen": "read_only",
    "local_reference_search": "read_only",
    "malware_analysis": "read_only",
    "manifest_to_device": "read_write_artifacts",
    "manim_renderer": "read_write_artifacts",
    "mcp_client": "external_io",
    "memory_ops": "state_mutation",
    "memory_sync": "state_mutation",
    "messages": "external_io",
    "native_chat": "pure_compute",
    "network_discovery": "external_io",
    "network_ops": "read_only",
    "network_recon": "read_only",
    "notify_user": "external_io",
    "os_automation": "foreground_desktop_control",
    "os_manipulation": "foreground_desktop_control",
    "personality": "state_mutation",
    "plan_mode": "state_mutation",
    "program_dna_equivalence_battery": "read_write_artifacts",
    "program_dna_reconstruct": "read_write_artifacts",
    "propagation": "pure_compute",
    "query_beliefs": "read_only",
    "query_visual_context": "read_only",
    "reddit_adapter": "external_io",
    "render_bridge": "pure_compute",
    "run_code": "sandboxed_compute",
    "search_web": "read_only",
    "sec_ops": "read_only",
    "self_evolution": "privileged_mutation",
    "self_improvement": "privileged_mutation",
    "self_repair": "privileged_mutation",
    "social_lurker": "external_io",
    "shell": "privileged_mutation",
    "sovereign_browser": "external_io",
    "sovereign_imagination": "read_write_artifacts",
    "sovereign_network": "external_io",
    "sovereign_terminal": "privileged_mutation",
    "sovereign_vision": "read_only",
    "spawn_agent": "external_io",
    "spawn_agents_parallel": "external_io",
    "speak": "external_io",
    "stealth_ops": "read_only",
    "system_proprioception": "read_only",
    "system_ops": "foreground_desktop_control",
    "test_generator": "read_write_artifacts",
    "toggle_senses": "state_mutation",
    "train_self": "privileged_mutation",
    "uplink_local": "state_mutation",
    "voice_mute": "state_mutation",
    "voice_output": "external_io",
    "voice_stop_tts": "state_mutation",
    "voice_unmute": "state_mutation",
    "web_interlocutor": "foreground_browser_dialogue",
    "web_search": "read_only",
    "x_tools": "external_io",
}


INTERNAL_ONLY_SKILLS = frozenset(
    {
        "branching_futures",
        "manim_renderer",
        "mcp_client",
    }
)


# These project-local implementations are still discovered and reported, but
# they are not exposed as parallel authorities for capabilities already owned by
# hardened core skills.  A project declaration not listed here remains eligible
# and must pass the same catalog probe as a core declaration.
CLASS_EXCLUSIONS: dict[str, str] = {
    "skills.browser_action.UnifiedBrowserSkill": "superseded_by:sovereign_browser",
    "skills.network_discovery.NetworkDiscovery": "superseded_by:sovereign_network",
    "skills.network_ops.NetworkOpsSkill": "superseded_by:sovereign_network",
    "skills.network_recon.NetworkReconSkill": "superseded_by:sovereign_network",
    "skills.shell.ShellSkill": "superseded_by:sovereign_terminal",
    "skills.system_ops.SystemOpsSkill": "superseded_by:computer_use",
    "skills.train_self.TrainSelfSkill": "superseded_by:train_self",
}


@dataclass(frozen=True, slots=True)
class SkillPolicy:
    effect_scope: str
    authority_class: str


def authority_class_for(effect_scope: str) -> str | None:
    """Return the mandatory authority class for a recognized effect scope."""

    return _AUTHORITY_BY_EFFECT_SCOPE.get(str(effect_scope or "").strip().lower())


def resolve_skill_policy(name: str, declared_effect_scope: str = "") -> SkillPolicy | None:
    """Resolve a skill's policy without inventing a fallback classification."""

    scope = str(declared_effect_scope or "").strip().lower()
    if not scope:
        scope = SKILL_EFFECT_SCOPES.get(str(name or "").strip(), "")
    authority = authority_class_for(scope)
    if scope not in VALID_EFFECT_SCOPES or authority is None:
        return None
    return SkillPolicy(effect_scope=scope, authority_class=authority)


def class_exclusion_reason(module_path: str, class_name: str) -> str | None:
    return CLASS_EXCLUSIONS.get(f"{module_path}.{class_name}")

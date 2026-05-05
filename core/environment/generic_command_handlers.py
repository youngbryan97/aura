"""Generic command handlers that map ActionIntents to CommandSpecs.

These are environment-agnostic handlers. Environment-specific adapters
(terminal_grid, browser, shell) can register additional or override handlers.
The key contract: ActionIntent -> CommandSpec, never raw keys from policy.
"""
from __future__ import annotations

from .command import ActionIntent, CommandSpec, CommandStep, command_id_for


def _direction_key(direction: str) -> str:
    """Map semantic direction to a generic key. Adapters may override."""
    return {
        "north": "k", "south": "j", "east": "l", "west": "h",
        "northeast": "u", "northwest": "y", "southeast": "n", "southwest": "b",
        "up": "<", "down": ">",
    }.get(direction, direction)


def handle_move(intent: ActionIntent) -> CommandSpec:
    direction = intent.parameters.get("direction", "")
    key = _direction_key(direction)
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=key)],
        expected_effects=[intent.expected_effect or "position_changed"],
    )


def handle_wait(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=".")],
        expected_effects=["turn_passed"],
    )


def handle_search(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value="s")],
        expected_effects=["area_searched"],
    )


def handle_inventory(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value="i")],
        expected_effects=["inventory_displayed"],
    )


def handle_eat(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="e")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=["has_food"],
        steps=steps,
        expected_effects=["nutrition_improved"],
    )


def handle_quaff(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="q")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=["has_potion"],
        steps=steps,
        expected_effects=["potion_effect"],
    )


def handle_read(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="r")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=["has_readable"],
        steps=steps,
        expected_effects=["scroll_effect"],
    )


def handle_wield(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="w")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=steps,
        expected_effects=["weapon_wielded"],
    )


def handle_pickup(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=",")],
        expected_effects=["item_acquired"],
    )


def handle_drop(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="d")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=steps,
        expected_effects=["item_dropped"],
    )


def handle_use_stairs(intent: ActionIntent) -> CommandSpec:
    direction = intent.parameters.get("direction", "down")
    key = "<" if direction == "up" else ">"
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=["on_stairs"],
        steps=[CommandStep(kind="key", value=key)],
        expected_effects=["level_changed"],
    )


def handle_open_door(intent: ActionIntent) -> CommandSpec:
    direction = intent.parameters.get("direction", "")
    steps = [CommandStep(kind="key", value="o")]
    if direction:
        steps.append(CommandStep(kind="key", value=_direction_key(direction)))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=steps,
        expected_effects=["door_opened"],
    )


def handle_use(intent: ActionIntent) -> CommandSpec:
    letter = intent.parameters.get("item_letter", "")
    steps = [CommandStep(kind="key", value="a")]
    if letter:
        steps.append(CommandStep(kind="key", value=letter))
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=steps,
        expected_effects=["item_applied"],
    )


def handle_observe(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="observe", value="look")],
        expected_effects=["observation_updated"],
    )


def handle_explore_frontier(intent: ActionIntent) -> CommandSpec:
    """Explore is a meta-intent. The command compiler resolves it to a concrete move
    based on the belief graph's frontier set. Default: move in a direction."""
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=".")],  # placeholder — policy should resolve
        expected_effects=["frontier_progress"],
    )


def handle_resolve_modal(intent: ActionIntent) -> CommandSpec:
    response = intent.parameters.get("response", " ")
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=response)],
        expected_effects=["modal_cleared"],
    )


def handle_pray(intent: ActionIntent) -> CommandSpec:
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value="#"), CommandStep(kind="text", value="pray\n")],
        expected_effects=["prayer_effect"],
    )


def handle_retreat_to_safety(intent: ActionIntent) -> CommandSpec:
    """Retreat is resolved by the policy — the command compiler picks the best safe direction."""
    direction = intent.parameters.get("direction", "")
    key = _direction_key(direction) if direction else "."
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=key)],
        expected_effects=["retreated"],
    )


def handle_stabilize_resource(intent: ActionIntent) -> CommandSpec:
    """Meta-intent: stabilize a critical resource. Falls back to wait."""
    return CommandSpec(
        command_id=command_id_for("generic", intent),
        environment_id="generic",
        intent=intent,
        preconditions=[],
        steps=[CommandStep(kind="key", value=".")],
        expected_effects=["resource_stabilized"],
    )


# Registry of all generic handlers
GENERIC_HANDLERS: dict[str, callable] = {
    "move": handle_move,
    "wait": handle_wait,
    "search": handle_search,
    "inventory": handle_inventory,
    "eat": handle_eat,
    "quaff": handle_quaff,
    "read": handle_read,
    "wield": handle_wield,
    "pickup": handle_pickup,
    "drop": handle_drop,
    "use_stairs": handle_use_stairs,
    "open_door": handle_open_door,
    "use": handle_use,
    "observe": handle_observe,
    "explore_frontier": handle_explore_frontier,
    "resolve_modal": handle_resolve_modal,
    "pray": handle_pray,
    "retreat_to_safety": handle_retreat_to_safety,
    "stabilize_resource": handle_stabilize_resource,
}


def register_generic_handlers(compiler) -> None:
    """Register all generic handlers on a CommandCompiler instance."""
    for name, handler in GENERIC_HANDLERS.items():
        compiler.register(name, handler)


__all__ = ["GENERIC_HANDLERS", "register_generic_handlers"]

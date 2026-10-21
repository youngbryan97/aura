"""Propose layered artwork from owned source attachment and position effects.

The bytecode describes a possible picture. Calls, aliases, method execution
and later runtime mutations remain unverified. These proposals can only
create search templates; they establish no current scene or object role.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.perception.flash_action_sources import FlashActionSources
from core.perception.guarded_effects_in_flash_methods import FlashEffectGraph, read_flash_effect_graph
from core.perception.symbolic_motion_in_flash_methods import bind_flash_methods
from core.perception.visual_assets_in_a_flash_file import FlashVisualAssets
from core.verify import invariant

_NAME = re.compile(r'[A-Za-z_$][\w$]*\Z')


@dataclass(frozen=True)
class SourceLayerProposal:
    source: str
    method: str
    instruction_source: str
    base_character: int
    over_character: int
    source_delta: tuple[float, float]
    depths: tuple[float, float]
    offsets: tuple[int, int, int, int]
    uncertainties: tuple[str, ...] = ('execution', 'receiver_aliases', 'later_mutations', 'current_frames', 'current_pixels')

    def __post_init__(self) -> None:
        if (not isinstance(self.source, str) or not re.fullmatch(r'swf:[0-9a-f]{64}', self.source)
                or not isinstance(self.method, str) or not 1 <= len(self.method) <= 512
                or not isinstance(self.instruction_source, str) or not self.instruction_source.startswith(self.source+'/')
                or any(type(n) is not int or not 1 <= n <= 65535 for n in (self.base_character, self.over_character))
                or not isinstance(self.source_delta, tuple) or len(self.source_delta) != 2
                or any(type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 4096 for n in self.source_delta)
                or not isinstance(self.depths, tuple) or len(self.depths) != 2
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in self.depths)
                or not 0 <= self.depths[0] < self.depths[1] <= 1048575
                or not isinstance(self.offsets, tuple) or len(self.offsets) != 4
                or any(type(n) is not int or n < 0 for n in self.offsets)
                or self.uncertainties != ('execution', 'receiver_aliases', 'later_mutations', 'current_frames', 'current_pixels')):
            raise ValueError('source layers are bounded owned proposals with execution and current pixels unresolved')


def _literal(graph: FlashEffectGraph, index: int, kind: str):
    node = graph.nodes[index]
    return node.data[0] if node.operation == kind and len(node.data) == 1 else None


def _object(graph: FlashEffectGraph, index: int):
    node = graph.nodes[index]
    if node.operation == 'get_variable':
        name = _literal(graph, node.inputs[1], 'string')
        return ('variable', name) if isinstance(name, str) and _NAME.fullmatch(name) else None
    if node.operation == 'get_register':
        number = _literal(graph, node.inputs[1], 'number')
        return ('register', number) if type(number) is float and number.is_integer() else None
    return None


def _copied_axis(graph: FlashEffectGraph, index: int):
    node, delta = graph.nodes[index], 0.
    if node.operation == 'binary' and node.data[1] in (0x0A, 0x47, 0x0B):
        left, right = node.inputs[1:]
        value = _literal(graph, right, 'number')
        if type(value) is not float or not math.isfinite(value) or abs(value) > 4096:
            return None
        delta = -value if node.data[1] == 0x0B else value
        node = graph.nodes[left]
    if node.operation != 'get_member':
        return None
    owner, axis = _object(graph, node.inputs[1]), _literal(graph, node.inputs[2], 'string')
    return (owner[1], axis, delta) if owner is not None and owner[0] == 'variable' and axis in ('_x', '_y') else None


def layers_in_source_graph(graph: FlashEffectGraph, table: FlashVisualAssets) -> tuple[SourceLayerProposal, ...]:
    """Read literal attachment pairs and final copied positions in the lifted prefix."""
    if (not isinstance(graph, FlashEffectGraph) or not isinstance(table, FlashVisualAssets) or not table.complete
            or not graph.source.startswith(table.source+'/') or len(graph.effects) > 1024
            or len(graph.nodes) > 32768):
        return ()
    attached, duplicates, writes, blocked = {}, set(), {}, set()
    for effect in graph.effects:
        if (effect.offset not in graph.lifted or graph.nodes[effect.guard].operation != 'boolean'
                or graph.nodes[effect.guard].data != (True,)):
            continue
        args = effect.operands
        if effect.operation == 'call_method':
            receiver, method = _object(graph, args[0]), _literal(graph, args[1], 'string')
            if method == 'attachMovie' and len(args) == 5 and receiver is not None:
                symbol, instance, depth = (_literal(graph, args[2], 'string'), _literal(graph, args[3], 'string'),
                                           _literal(graph, args[4], 'number'))
                asset = table.named(symbol) if isinstance(symbol, str) else None
                if (asset is not None and isinstance(instance, str) and _NAME.fullmatch(instance)
                        and type(depth) is float and math.isfinite(depth) and depth.is_integer()
                        and 0 <= depth <= 1048575):
                    if instance in attached:
                        duplicates.add(instance)
                    attached[instance] = receiver, asset.character_id, depth, effect.offset
            elif receiver is not None and receiver[0] == 'variable' and receiver[1] in attached:
                blocked.add(receiver[1])
        elif effect.operation == 'set_member':
            receiver, field = _object(graph, args[0]), _literal(graph, args[1], 'string')
            if receiver is None or receiver[0] != 'variable':
                continue
            if field in ('_rotation', '_xscale', '_yscale', '_alpha', '_transform', 'transform'):
                blocked.add(receiver[1])
            if field in ('_x', '_y'):
                writes[receiver[1], field] = effect.offset, _copied_axis(graph, args[2])
    result = []
    for over, (container, character, depth, attachment) in attached.items():
        if over in duplicates | blocked:
            continue
        x, y = writes.get((over, '_x')), writes.get((over, '_y'))
        if (x is None or y is None or x[1] is None or y[1] is None or x[1][0] != y[1][0]
                or x[1][1] != '_x' or y[1][1] != '_y' or min(x[0], y[0]) <= attachment):
            continue
        base = x[1][0]
        if base not in attached or base in duplicates | blocked or base == over:
            continue
        own_container, own_character, own_depth, own_attachment = attached[base]
        if own_container != container or own_depth >= depth:
            continue
        if any(writes.get((base, axis), (-1,))[0] >= copied[0] for axis, copied in (('_x', x), ('_y', y))):
            continue
        result.append(SourceLayerProposal(table.source, graph.target, graph.source, own_character, character,
            (x[1][2], y[1][2]), (own_depth, depth), (own_attachment, attachment, x[0], y[0])))
    return tuple(result[:32])


def source_layer_proposals(sources: FlashActionSources, tables: tuple) -> tuple[SourceLayerProposal, ...]:
    """Bound source work; incomplete execution graphs may propose their owned prefix."""
    if not isinstance(sources, FlashActionSources) or not sources.complete or len(tables) > 6:
        return ()
    table = next((t for t in tables if isinstance(t, FlashVisualAssets) and t.source == sources.source and t.complete), None)
    bindings = bind_flash_methods(sources.streams)
    if table is None or bindings.issues or len(bindings.methods) > 128:
        return ()
    result = []
    for binding in bindings.methods:
        if not binding.function.body.complete or len(binding.function.body.instructions) > 4096:
            continue
        graph = read_flash_effect_graph(binding)
        result.extend(layers_in_source_graph(graph, table))
        if len(result) >= 32:
            break
    return tuple(dict.fromkeys(result))[:32]


@invariant('perception.source_layer_proposals_are_not_current_scene_facts', scope='perception',
           owner='core/perception/source_layers_with_copied_positions.py', observational=False)
def _source_layer_proposal_invariant() -> tuple:
    source = 'swf:'+'1'*64
    proposal = SourceLayerProposal(source, 'method', source+'/function', 7, 9, (0., 0.), (10., 40.), (1, 2, 3, 4))
    assert 'execution' in proposal.uncertainties and 'current_pixels' in proposal.uncertainties
    assert layers_in_source_graph(None, None) == ()
    return ()

"""Differentiable selected edges from independently conditioned workspaces."""

import mlx.core as mx
import mlx.nn as nn

from core.verify.invariants import invariant


def conditioned_selected_scores(pointer, operations, mentions, candidates, *, rows, columns,
                               adjacency, role_features):
    """Share the exact post-projection function between training and decoding.

    Every alternative retains all graph nodes and message rounds. Gathering
    the final pointwise output early cannot affect another edge or round.
    Inputs are batch x occurrence x relation-width, after depth projection.
    """
    role_count = operations.shape[1]
    role_query = role_features @ pointer.lora_role_query if pointer.role_queries else None
    nodes = mx.concatenate([operations + mentions + (0. if role_query is None else role_query), candidates], axis=1)
    if adjacency is not None:
        degree = mx.sum(adjacency.astype(mx.float32), axis=-1, keepdims=True)
        weights = adjacency.astype(mx.float32) / mx.maximum(degree, 1.)
        for _ in range(pointer.rounds):
            message = weights @ (nodes @ pointer.lora_message)
            update = mx.tanh(mx.concatenate([nodes, message], axis=-1) @ pointer.lora_update)
            nodes = nodes + mx.where(degree > 0, update, 0.)
    index = mx.arange(operations.shape[0])
    o, m = operations[index, rows], mentions[index, rows]
    c, contextual = nodes[index, role_count + columns], nodes[index, rows]
    if role_query is None:
        blocks = [o, m, c, o * m, o * c, m * c, o * m * c, contextual - c]
    else:
        r = role_query[rows]
        blocks = [o, r, m, c, o * r, o * m, o * c, r * m, r * c, m * c,
            o * r * m, o * r * c, o * m * c, r * m * c, o * r * m * c, contextual - c]
    return (nn.silu(mx.concatenate(blocks, axis=-1)) @ pointer.lora_b).squeeze(-1)


def conditional_choice_scores(baseline, learned, *, weight=1.):
    """Preserve the baseline partition while cancelling learned logit offsets."""
    shifted = baseline + weight * (learned - mx.max(learned))
    return shifted - (mx.logsumexp(shifted) - mx.logsumexp(baseline))


@invariant("learning.conditioned_program_choice_mass_is_preserved", scope="learning",
           owner="core/learning/semantic_conditioned_relations.py", observational=False)
def _conditional_choice_mass():
    baseline, learned = mx.array([2., -1., .2]), mx.array([.7, -.8, 1.2])
    updated = conditional_choice_scores(baseline, learned)
    assert abs((mx.logsumexp(updated) - mx.logsumexp(baseline)).item()) < 1e-5
    assert mx.all(mx.abs(updated - conditional_choice_scores(baseline, learned + 10.)) < 1e-5).item()
    return ()

"""Each amendment the transducer can be rebuilt with.

Lifted whole out of `semantic_program_compositional_transducer`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic_program_compositional_transducer import (
        CompositionalSemanticProgramTransducer,
    )


class _CarriesItsAmendments:
    """Lifted whole out of CompositionalSemanticProgramTransducer; see semantic_program_compositional_transducer.py."""

    def with_prefix_feasible_arguments(self) -> CompositionalSemanticProgramTransducer:
        """Create a separately identified search candidate without refitting tissue."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )


        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_search_strategy"] = "prefix_feasible_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_global_constraint_arguments(self) -> CompositionalSemanticProgramTransducer:
        """Create an opt-in solver-assisted chart candidate with unchanged tissue."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_search_strategy"] = "global_constraint_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_conditional_argument_scores(self) -> CompositionalSemanticProgramTransducer:
        """Condition binary argument evidence on selecting one mention per slot."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_score_strategy"] = "conditional_log_odds_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_literal_anchor_retention(self) -> CompositionalSemanticProgramTransducer:
        """Keep exact input identities available after learned mention pruning."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_proposal_retention"] = "ranked_with_literal_anchors_v2"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_overlap_complete_mentions(self) -> CompositionalSemanticProgramTransducer:
        """Retain every scored mention not dominated under overlap constraints."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_proposal_retention"] = "overlap_dominance_v3"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_atomic_literal_arguments(self) -> CompositionalSemanticProgramTransducer:
        """Preserve parser-owned literal boundaries in the learned reference chart."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body["argument_literal_boundaries"] = "atomic_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_feasible_operation_charts(self, *, preserve_arity_states=False) -> CompositionalSemanticProgramTransducer:
        """Spend chart capacity only on cardinalities permitted by the graph contract."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["operation_chart_feasibility"] = "arity_state_bounds_v3" if preserve_arity_states else "register_edge_bounds_v2"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_joint_operation_argument_scores(self) -> CompositionalSemanticProgramTransducer:
        """Compare complete graphs using the sum of their declared factor scores."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body["operation_assignment_policy"] = "joint_factor_score_v2"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_typed_operation_charts(self) -> CompositionalSemanticProgramTransducer:
        """Preserve type-demand states before pruning the operation chart beam."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body["operation_chart_feasibility"] = "typed_state_bounds_v4"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_operation_label_alternatives(self, limit: int) -> CompositionalSemanticProgramTransducer:
        """Make ambiguity retention explicit in the candidate identity."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body["operation_label_limit"] = limit
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_complete_operation_search(self, *, max_expansions=None) -> CompositionalSemanticProgramTransducer:
        """Search all source spans and learned operation labels inside the declared bounds."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body.update(operation_search_policy="complete_bounded_v1",
                    operation_assignment_policy="joint_factor_score_v2",
                    operation_label_limit=len(self.operation_head.labels),
                    operation_search_max_expansions=max_expansions)
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_order_invariant_argument_graph(self) -> CompositionalSemanticProgramTransducer:
        """Let the complete graph decide dependencies regardless of textual order."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_search_strategy"] = "global_constraint_v1"
        body["forward_reference_policy"] = "joint_graph_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_categorical_relation_scores(self) -> CompositionalSemanticProgramTransducer:
        """Preserve trained register log-odds without increasing mention evidence."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["relation_score_strategy"] = "categorical_log_margin_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_expanded_relation_rank(self, rank: int, *, seed: int = 0) -> CompositionalSemanticProgramTransducer:
        """Create a wider development candidate, retaining the old bilinear component."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        head = self.definition_relation_head.expanded_rank(rank, seed=seed)
        body = {key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"}
        body["coefficient_sha256"] = _sha({**self._coefficient_body(), "definition_relation_head": head.to_dict()})
        body["relation_rank_expansions"] = [*body.get("relation_rank_expansions", []), {
            "schema": "aura.semantic_relation_rank_expansion.v1",
            "parent_transducer_receipt_sha256": self.receipt_sha256,
            "previous_rank": self.definition_relation_head.query_projection.shape[1],
            "rank": rank, "seed": seed, "initialization": "orthogonal_query_zero_definition_v1",
            "serving_authority": False, "numerical_replay_required": True,
        }]
        return replace(self, definition_relation_head=head, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_source_ordered_definitions(self) -> CompositionalSemanticProgramTransducer:
        """Bound definition clauses by textual neighbors, not register numbering."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["definition_boundary_policy"] = "source_neighbors_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_joint_definition_graph(self) -> CompositionalSemanticProgramTransducer:
        """Resolve one consistent definition per register using all selected uses."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_search_strategy"] = "global_constraint_v1"
        body["definition_selection_policy"] = "joint_graph_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def with_bidirectional_input_definitions(self) -> CompositionalSemanticProgramTransducer:
        """Admit input names on either side within source-neighbor boundaries."""
        from .semantic_program_compositional_transducer import (
            _sha,
            replace,
        )

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["definition_boundary_policy"] = "source_neighborhood_v2"
        body["argument_search_strategy"] = "global_constraint_v1"
        body["definition_selection_policy"] = "joint_graph_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})


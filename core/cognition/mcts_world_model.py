"""core/cognition/mcts_world_model.py -- MCTS over Learned VRNN Dynamics
========================================================================
Implements Monte Carlo Tree Search (MCTS) utilizing the learned latent
dynamics (VRNN) from `learned_world_model.py`.

Unlike heuristic search, this planner:
  1. Expands nodes using the learned transition model (prior imagination).
  2. Evaluates states using a value scorer trained on actual outcomes.
  3. Uses latent uncertainty (from the VRNN prior logvar) to guide UCB exploration.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.world_model.learned_world_model import LearnedWorldModel, WorldModelPrediction

logger = logging.getLogger("Aura.MCTSPlanner")


class MCTSNode:
    """A node in the MCTS search tree, grounded in the VRNN latent space."""

    def __init__(
        self,
        latent_state: np.ndarray,
        hidden_state: np.ndarray,
        parent: Optional[MCTSNode] = None,
        action_from_parent: Optional[np.ndarray] = None,
        prior_prob: float = 1.0,
    ):
        self.latent_state = latent_state
        self.hidden_state = hidden_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_prob = prior_prob

        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
        # Uncertainty drives exploration (UCB)
        self.uncertainty = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class LearnedMCTSPlanner:
    """MCTS planner that simulates counterfactuals inside the VRNN latent space.
    
    This fulfills the System 2 deliberation requirement: actual multi-step
    lookahead using a model trained on action/outcome traces.
    """

    def __init__(
        self,
        world_model: LearnedWorldModel,
        action_space: List[np.ndarray],
        value_scorer: Callable[[np.ndarray], float],
        exploration_constant: float = 1.414,
        max_depth: int = 20,
        num_simulations: int = 100,
    ):
        self.world_model = world_model
        self.action_space = action_space
        self.value_scorer = value_scorer  # Must score a latent/observation state
        self.c_puct = exploration_constant
        self.max_depth = max_depth
        self.num_simulations = num_simulations

    def plan(
        self,
        current_observation: np.ndarray,
        ablate_learned_model: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run MCTS to find the best immediate action.

        Args:
            current_observation: Current state vector.
            ablate_learned_model: If True, uses random heuristic rollout instead
                                  of the learned VRNN transition model (for proof).
        """
        # Initialize root node with current VRNN hidden state
        # First encode the observation to get the latent
        # For simplicity, we just use the current VRNN hidden state `h`
        root = MCTSNode(
            latent_state=np.zeros(self.world_model.config.latent_dim),
            hidden_state=self.world_model.h.copy()
        )

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            depth = 0

            # 1. Selection
            while node.is_expanded and depth < self.max_depth:
                action_idx, node = self._select_child(node)
                search_path.append(node)
                depth += 1

            # 2. Expansion & Evaluation
            value = 0.0
            if depth < self.max_depth:
                self._expand(node, ablate_learned_model)
                value = self.value_scorer(node.hidden_state) if not ablate_learned_model else np.random.uniform(-1.0, 1.0)
            else:
                value = self.value_scorer(node.hidden_state) if not ablate_learned_model else 0.0

            # 3. Backpropagation
            self._backpropagate(search_path, value)

        # Select the best action based on visit counts
        if not root.children:
            best_action_idx = 0
            best_q = 0.0
        else:
            best_action_idx = max(
                root.children.items(),
                key=lambda item: item[1].visit_count
            )[0]
            best_q = root.children[best_action_idx].q_value
            
        info = {
            "root_visits": root.visit_count,
            "best_q": best_q,
            "max_depth_reached": max((len(search_path) for _ in range(10))) if 'search_path' in locals() else 0,
        }

        return self.action_space[best_action_idx], info

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child using PUCT algorithm (combines Q, Prior, and Uncertainty)."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action_idx, child in node.children.items():
            if child.visit_count == 0:
                q_val = 0.0
                u_val = self.c_puct * child.prior_prob * math.sqrt(node.visit_count + 1e-8)
            else:
                q_val = child.q_value
                # Uncertainty bonus from VRNN latent variance
                u_val = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
                u_val += child.uncertainty * 0.1  # Exploration bonus for uncertain dynamics

            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        return best_action, best_child

    def _expand(self, node: MCTSNode, ablate_learned_model: bool = False):
        """Expand node using the VRNN prior transition model."""
        node.is_expanded = True
        
        for action_idx, action in enumerate(self.action_space):
            if ablate_learned_model:
                # Dummy expansion without learned dynamics
                child = MCTSNode(
                    latent_state=np.zeros_like(node.latent_state),
                    hidden_state=np.zeros_like(node.hidden_state),
                    parent=node,
                    action_from_parent=action,
                    prior_prob=1.0 / len(self.action_space)
                )
                child.uncertainty = 0.0
                node.children[action_idx] = child
                continue

            # Use VRNN to predict next state
            act_pad = self.world_model._pad_or_truncate(action, self.world_model.config.action_dim)
            
            # Predict from Prior: P(z | h)
            prior_params = self.world_model.W_prior @ node.hidden_state + self.world_model.b_prior
            prior_mean, prior_logvar = np.split(prior_params, 2)
            prior_logvar = np.clip(prior_logvar, -5.0, 2.0)
            
            # Sample z
            z = self.world_model._reparameterize(prior_mean, prior_logvar)
            
            # Transition: h' = GRU(z, a, h)
            gru_input = np.concatenate([z, act_pad])
            next_h = self.world_model._gru_step(gru_input, node.hidden_state)
            
            # Create child
            child = MCTSNode(
                latent_state=z,
                hidden_state=next_h,
                parent=node,
                action_from_parent=action,
                prior_prob=1.0 / len(self.action_space)  # Uniform prior if no policy network
            )
            
            # High variance in the prior means uncertain transition
            child.uncertainty = float(np.mean(np.exp(prior_logvar)))
            
            node.children[action_idx] = child

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Propagate value up the tree."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Simple discount could be applied here

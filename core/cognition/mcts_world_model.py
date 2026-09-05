"""core/cognition/mcts_world_model.py -- MCTS over Learned VRNN Dynamics
========================================================================
Implements Monte Carlo Tree Search (MCTS) using the learned latent
dynamics (VRNN) from `learned_world_model.py`.

Unlike heuristic search, this planner:
  1. Expands nodes using the learned transition model (prior imagination).
  2. Evaluates states using a value scorer trained on actual outcomes.
  3. Uses latent uncertainty (from the VRNN prior logvar) to guide UCB exploration.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from core.world_model.learned_world_model import LearnedWorldModel

logger = logging.getLogger("Aura.MCTSPlanner")


class MCTSNode:
    """A node in the MCTS search tree, grounded in the VRNN latent space."""

    def __init__(
        self,
        latent_state: np.ndarray,
        hidden_state: np.ndarray,
        parent: MCTSNode | None = None,
        action_from_parent: np.ndarray | None = None,
        prior_prob: float = 1.0,
    ):
        self.latent_state = latent_state
        self.hidden_state = hidden_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_prob = prior_prob

        self.children: dict[int, MCTSNode] = {}
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
        action_space: list[np.ndarray],
        value_scorer: Callable[[np.ndarray], float],
        exploration_constant: float = 1.414,
        max_depth: int = 20,
        num_simulations: int = 100,
    ):
        self.world_model = world_model
        if not action_space:
            raise ValueError("LearnedMCTSPlanner requires at least one action")
        self.action_space = action_space
        self.value_scorer = value_scorer  # Must score a latent/observation state
        self.c_puct = exploration_constant
        self.max_depth = max_depth
        self.num_simulations = num_simulations

    def plan(
        self,
        current_observation: np.ndarray,
        ablate_learned_model: bool = False
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run MCTS to find the best immediate action.

        Args:
            current_observation: Current state vector.
            ablate_learned_model: If True, uses random heuristic rollout instead
                                  of the learned VRNN transition model (for proof).
        """
        root_latent, root_hidden = self._encode_root(current_observation)
        root = MCTSNode(
            latent_state=root_latent,
            hidden_state=root_hidden,
        )
        max_path_len = 1

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            depth = 0

            # 1. Selection
            while node.is_expanded and depth < self.max_depth:
                action_idx, node = self._select_child(node)
                search_path.append(node)
                depth += 1
            max_path_len = max(max_path_len, len(search_path))

            # 2. Expansion & Evaluation
            value = 0.0
            if depth < self.max_depth:
                self._expand(node, ablate_learned_model)
                value = float(self.value_scorer(node.hidden_state))
            else:
                value = float(self.value_scorer(node.hidden_state))

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
            "max_depth_reached": max_path_len,
            "ablated_learned_model": ablate_learned_model,
            "child_visits": {
                str(idx): child.visit_count for idx, child in root.children.items()
            },
        }

        return self.action_space[best_action_idx], info

    def _encode_root(self, current_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode the current observation without mutating the learned model."""

        obs = self.world_model._pad_or_truncate(
            current_observation,
            self.world_model.config.observation_dim,
        )
        hidden = self.world_model.h.copy()
        enc_input = np.concatenate([obs, hidden])
        post_params = self.world_model.W_enc @ enc_input + self.world_model.b_enc
        post_mean, _post_logvar = np.split(post_params, 2)
        latent = np.asarray(post_mean, dtype=np.float32)
        zero_action = np.zeros(self.world_model.config.action_dim, dtype=np.float32)
        next_hidden = self.world_model._gru_step(np.concatenate([latent, zero_action]), hidden)
        return latent, next_hidden

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
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
                child_latent, child_hidden = self._ablation_transition(node, action)
                child = MCTSNode(
                    latent_state=child_latent,
                    hidden_state=child_hidden,
                    parent=node,
                    action_from_parent=action,
                    prior_prob=1.0 / len(self.action_space)
                )
                child.uncertainty = 1.0
                node.children[action_idx] = child
                continue

            # Use VRNN to predict next state
            act_pad = self.world_model._pad_or_truncate(action, self.world_model.config.action_dim)
            
            # Predict from Prior: P(z | h)
            prior_params = self.world_model.W_prior @ node.hidden_state + self.world_model.b_prior
            prior_mean, prior_logvar = np.split(prior_params, 2)
            prior_logvar = np.clip(prior_logvar, -5.0, 2.0)
            
            # Use the deterministic prior mean for planning stability. The
            # variance still contributes to uncertainty-driven exploration.
            z = prior_mean.astype(np.float32)
            
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

    def _ablation_transition(
        self,
        node: MCTSNode,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic non-learned transition used only for ablation probes."""

        act_pad = self.world_model._pad_or_truncate(action, self.world_model.config.action_dim)
        latent = np.zeros_like(node.latent_state)
        latent_width = min(latent.shape[0], act_pad.shape[0])
        if latent_width:
            latent[:latent_width] = act_pad[:latent_width]

        hidden = node.hidden_state.copy()
        hidden_width = min(hidden.shape[0], act_pad.shape[0])
        if hidden_width:
            hidden[:hidden_width] = np.tanh(hidden[:hidden_width] + 0.05 * act_pad[:hidden_width])
        return latent, hidden

    def _backpropagate(self, search_path: list[MCTSNode], value: float):
        """Propagate value up the tree."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Simple discount could be applied here

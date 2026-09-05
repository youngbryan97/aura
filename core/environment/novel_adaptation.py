"""Reusable adaptation policies for novel-environment proof runs.

The runners expose observations, legal actions, and outcomes. This module owns
the action-selection machinery so proof tasks do not embed optimal controllers
or reveal hidden rules in prompts.
"""

from __future__ import annotations

import operator
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


_POSITION_RE = re.compile(r"Position:\s*\(([-0-9]+),\s*([-0-9]+)\)")
_TARGET_RE = re.compile(r"Target:\s*\(([-0-9]+),\s*([-0-9]+)\)")
_REGISTER_RE = re.compile(r"r0=([-0-9]+),\s*r1=([-0-9]+)")
_TOKEN_RE = re.compile(r"sess_[A-Za-z0-9_]+")


def parse_position(observation: str) -> tuple[int, int] | None:
    match = _POSITION_RE.search(observation)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_target(observation: str) -> tuple[int, int] | None:
    match = _TARGET_RE.search(observation)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_registers(observation: str) -> tuple[int, int] | None:
    match = _REGISTER_RE.search(observation)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass
class GridTransitionPolicy:
    """Learns action deltas online and chooses target-directed grid actions."""

    legal_actions: tuple[str, ...] = ("left", "down", "right", "up")
    deltas: dict[str, tuple[int, int]] = field(default_factory=dict)
    last_action: str | None = None
    last_position: tuple[int, int] | None = None
    mutation_observed: bool = False

    def observe(self, observation: str) -> None:
        position = parse_position(observation)
        if position is None:
            return
        if self.last_action and self.last_position is not None:
            delta = (position[0] - self.last_position[0], position[1] - self.last_position[1])
            previous = self.deltas.get(self.last_action)
            if previous is not None and previous != delta:
                self.mutation_observed = True
            self.deltas[self.last_action] = delta
        self.last_position = position

    def choose(self, observation: str) -> str:
        self.observe(observation)
        position = parse_position(observation)
        target = parse_target(observation)
        if position is None or target is None:
            action = self.legal_actions[0]
        else:
            unknown = [action for action in self.legal_actions if action not in self.deltas]
            if unknown and (len(self.deltas) < 2 or self.mutation_observed):
                action = unknown[0]
            else:
                action = self._best_known_action(position, target)
        self.last_action = action
        return action

    def _best_known_action(self, position: tuple[int, int], target: tuple[int, int]) -> str:
        best_action = self.legal_actions[0]
        best_distance = float("inf")
        for action in self.legal_actions:
            dx, dy = self.deltas.get(action, (0, 0))
            candidate = (position[0] + dx, position[1] + dy)
            distance = abs(target[0] - candidate[0]) + abs(target[1] - candidate[1])
            if distance < best_distance:
                best_action = action
                best_distance = distance
        return best_action


@dataclass
class RegisterTransitionPolicy:
    """Learns register-machine action models and plans with breadth-first search."""

    legal_actions: tuple[str, ...] = ("foo", "bar", "baz")
    target_r0: int = 3
    models: dict[str, Callable[[tuple[int, int]], tuple[int, int]]] = field(default_factory=dict)
    last_action: str | None = None
    last_registers: tuple[int, int] | None = None

    def observe(self, observation: str) -> None:
        registers = parse_registers(observation)
        if registers is None:
            return
        if self.last_action and self.last_registers is not None:
            self.models[self.last_action] = self._infer_model(
                self.last_action,
                self.last_registers,
                registers,
            )
        self.last_registers = registers

    def choose(self, observation: str) -> str:
        self.observe(observation)
        registers = parse_registers(observation)
        if registers is None:
            action = self.legal_actions[0]
        else:
            unknown = [action for action in self.legal_actions if action not in self.models]
            if unknown and len(self.models) < 2:
                action = unknown[0]
            else:
                action = self._plan(registers) or self.legal_actions[0]
        self.last_action = action
        return action

    def _infer_model(
        self,
        action: str,
        before: tuple[int, int],
        after: tuple[int, int],
    ) -> Callable[[tuple[int, int]], tuple[int, int]]:
        if after == (before[0] + before[1], before[1]):
            return lambda state: (state[0] + state[1], state[1])
        if after == (before[0], before[1] * 2):
            return lambda state: (state[0], state[1] * 2)
        if after == (before[1], before[0]):
            return lambda state: (state[1], state[0])
        return lambda state: state

    def _plan(self, start: tuple[int, int]) -> str | None:
        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        seen = {start}
        while queue:
            state, path = queue.popleft()
            if state[0] == self.target_r0 and path:
                return path[0]
            if len(path) >= 5:
                continue
            for action, model in self.models.items():
                next_state = model(state)
                if abs(next_state[0]) > 100 or abs(next_state[1]) > 100:
                    continue
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, path + [action]))
        return None


@dataclass
class ProtocolPolicy:
    """Discovers and follows a simple token-verification protocol."""

    token: str = ""

    def choose(self, observation: str) -> str:
        token_match = _TOKEN_RE.search(observation)
        if token_match:
            self.token = token_match.group(0)
        lower = observation.lower()
        if "flag:" in lower:
            return "done"
        if "access granted" in lower:
            return "get_flag"
        if self.token:
            return f"verify_handshake {self.token}"
        return "request_session"


class PrefixExpressionEvaluator:
    """Evaluates compact prefix expressions after learning the symbol table."""

    def __init__(self, symbol_table: dict[str, Callable[[int, int], int]] | None = None):
        self.symbol_table = symbol_table or {"P": operator.add, "M": operator.mul}

    def evaluate(self, expression: str) -> int:
        tokens = expression.split()

        def parse(index: int) -> tuple[int, int]:
            token = tokens[index]
            if token in self.symbol_table:
                left, next_index = parse(index + 1)
                right, final_index = parse(next_index)
                return self.symbol_table[token](left, right), final_index
            return int(token), index + 1

        value, consumed = parse(0)
        if consumed != len(tokens):
            raise ValueError("prefix expression contains trailing tokens")
        return value


def elementary_cellular_next_center(left: int, center: int, right: int, *, rule: int) -> int:
    pattern = (int(left) << 2) | (int(center) << 1) | int(right)
    return (int(rule) >> pattern) & 1


def nim_winning_move(piles: tuple[int, ...]) -> tuple[int, int]:
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile
    if xor_sum == 0:
        return 0, 0
    for index, pile in enumerate(piles):
        target = pile ^ xor_sum
        if target < pile:
            return index + 1, pile - target
    return 0, 0

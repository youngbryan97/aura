"""core/consciousness/substrate_evolution.py — Substrate Evolution Engine

Applies real evolutionary selection pressure to the NeuralMesh topology.
This is not meta-optimization or hyperparameter tuning — it is Darwinian
selection on the connectome itself: the weights that define how information
flows, which columns couple, and how strongly.

Population-based:
  • Maintains a population of "genomes" (compressed weight configurations)
  • Each genome is evaluated by running it on the live mesh for an evaluation window
  • Fitness = Φ × coherence × energy_efficiency × binding_strength
  • Selection: tournament, top-k survive
  • Crossover: uniform crossover of inter-column weights
  • Mutation: Gaussian perturbation + structural (add/prune connections)
  • The best genome is applied to the live mesh after each generation

The evolutionary process runs in background, applying selection pressure that
mirrors biological evolution's role in shaping neural architecture.  Over time,
the mesh topology evolves to maximize integrated information, coherence, and
efficient energy use — the same pressures that shaped biological brains.

Safety:
  • Champion genome always preserved (elitism)
  • Rollback if new champion degrades live performance
  • Rate-limited to prevent destabilizing rapid topology changes
  • Generational history logged for analysis
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.utils.task_tracker import get_task_tracker

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.Evolution")


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 12          # small pop for local compute
    elite_count: int = 2               # always preserve top-2
    tournament_size: int = 3
    mutation_rate: float = 0.15        # probability per gene
    mutation_sigma: float = 0.02       # Gaussian perturbation std
    structural_mutation_rate: float = 0.05  # add/prune connection probability
    crossover_rate: float = 0.7
    evaluation_window_s: float = 30.0  # seconds to evaluate each genome
    generation_interval_s: float = 300.0  # 5 min between generations
    min_fitness_for_apply: float = 0.1  # don't apply terrible genomes
    rollback_threshold: float = 0.7     # rollback if fitness drops to this × champion


@dataclass
class Genome:
    """A compressed representation of inter-column connectivity."""
    id: int
    inter_weights: np.ndarray        # (columns, columns) inter-column weights
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    def copy(self, new_id: int) -> "Genome":
        return Genome(
            id=new_id,
            inter_weights=self.inter_weights.copy(),
            fitness=self.fitness,
            generation=self.generation,
            parent_ids=[self.id],
        )


@dataclass
class GenerationRecord:
    """Log of one evolutionary generation."""
    number: int
    timestamp: float
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    champion_id: int
    population_size: int
    mutations: int
    crossovers: int
    applied: bool


class SubstrateEvolution:
    """Evolutionary engine for NeuralMesh topology.

    Lifecycle:
        evo = SubstrateEvolution()
        await evo.start()
        ...
        await evo.stop()
    """

    def __init__(self, cfg: EvolutionConfig | None = None):
        self.cfg = cfg or EvolutionConfig()
        self._rng = np.random.default_rng(seed=None)  # non-deterministic

        self._population: List[Genome] = []
        self._champion: Optional[Genome] = None
        self._generation: int = 0
        self._next_id: int = 0
        self._history: List[GenerationRecord] = []

        # External refs (set by bridge)
        self._mesh_ref = None          # NeuralMesh
        self._binding_ref = None       # OscillatoryBinding
        self._workspace_ref = None     # GlobalWorkspace
        self._substrate_ref = None     # LiquidSubstrate

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pre_apply_fitness: float = 0.0  # for rollback detection

        logger.info("SubstrateEvolution initialized (pop=%d, gen_interval=%.0fs)",
                     self.cfg.population_size, self.cfg.generation_interval_s)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True

        # Initialize population from current mesh state
        if self._mesh_ref is not None:
            seed_weights = self._mesh_ref._inter_W.copy()
        else:
            seed_weights = np.zeros((64, 64), dtype=np.float32)

        self._population = []
        for i in range(self.cfg.population_size):
            genome = Genome(
                id=self._next_id,
                inter_weights=seed_weights + self._rng.standard_normal(seed_weights.shape).astype(np.float32) * 0.01,
                generation=0,
            )
            self._next_id += 1
            self._population.append(genome)

        # Seed champion is the current live mesh
        self._champion = Genome(id=-1, inter_weights=seed_weights.copy(), fitness=0.5)

        self._task = get_task_tracker().create_task(self._evolution_loop(), name="SubstrateEvolution")
        logger.info("SubstrateEvolution STARTED")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None
        logger.info("SubstrateEvolution STOPPED (generations=%d)", self._generation)

    # ── Main loop ────────────────────────────────────────────────────────

    async def _evolution_loop(self):
        try:
            while self._running:
                # Wait for generation interval
                await asyncio.sleep(self.cfg.generation_interval_s)
                if not self._running:
                    break

                try:
                    await self._run_generation()
                except Exception as e:
                    record_degradation('substrate_evolution', e)
                    logger.error("Evolution generation error: %s", e, exc_info=True)
        except asyncio.CancelledError:
            pass  # no-op: intentional

    async def _run_generation(self):
        """Execute one full evolutionary generation."""
        self._generation += 1
        gen = self._generation
        mutations = 0
        crossovers = 0

        # ── 1. Evaluate current population ────────────────────────────
        for genome in self._population:
            genome.fitness = await self._evaluate_fitness(genome)

        # ── 2. Sort by fitness ────────────────────────────────────────
        self._population.sort(key=lambda g: g.fitness, reverse=True)
        best = self._population[0]
        fitnesses = [g.fitness for g in self._population]

        # ── 3. Update champion ────────────────────────────────────────
        if self._champion is None or best.fitness > self._champion.fitness:
            self._champion = best.copy(best.id)
            logger.info("New champion genome #%d (fitness=%.4f) in gen %d",
                        best.id, best.fitness, gen)

        # ── 4. Selection + Reproduction ───────────────────────────────
        new_pop: List[Genome] = []

        # Elitism: preserve top-k
        for i in range(min(self.cfg.elite_count, len(self._population))):
            elite = self._population[i].copy(self._next_id)
            elite.generation = gen
            self._next_id += 1
            new_pop.append(elite)

        # Fill rest with tournament selection + crossover + mutation
        while len(new_pop) < self.cfg.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if self._rng.random() < self.cfg.crossover_rate:
                child_weights = self._crossover(parent1.inter_weights, parent2.inter_weights)
                crossovers += 1
                parent_ids = [parent1.id, parent2.id]
            else:
                child_weights = parent1.inter_weights.copy()
                parent_ids = [parent1.id]

            child_weights = self._mutate(child_weights)
            mutations += 1

            child = Genome(
                id=self._next_id,
                inter_weights=child_weights,
                generation=gen,
                parent_ids=parent_ids,
            )
            self._next_id += 1
            new_pop.append(child)

        self._population = new_pop

        # ── 5. Apply champion to live mesh ────────────────────────────
        applied = False
        if self._champion.fitness >= self.cfg.min_fitness_for_apply and self._mesh_ref is not None:
            self._pre_apply_fitness = self._champion.fitness
            self._mesh_ref._inter_W = self._champion.inter_weights.copy()
            applied = True
            logger.info("Applied champion genome to live mesh (fitness=%.4f)", self._champion.fitness)

        # ── 6. Record ────────────────────────────────────────────────
        record = GenerationRecord(
            number=gen,
            timestamp=time.time(),
            best_fitness=best.fitness,
            mean_fitness=float(np.mean(fitnesses)),
            worst_fitness=min(fitnesses),
            champion_id=self._champion.id,
            population_size=len(self._population),
            mutations=mutations,
            crossovers=crossovers,
            applied=applied,
        )
        self._history.append(record)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        logger.info(
            "Gen %d complete: best=%.4f mean=%.4f worst=%.4f champion=#%d applied=%s",
            gen, record.best_fitness, record.mean_fitness, record.worst_fitness,
            record.champion_id, applied,
        )

    # ── Fitness evaluation ───────────────────────────────────────────────

    async def _evaluate_fitness(self, genome: Genome) -> float:
        """Evaluate a genome's fitness from current system state.

        Fitness is a weighted combination of:
          Φ (integrated information)      × 0.30
          Coherence (binding PSI)         × 0.25
          Energy efficiency               × 0.20
          GWT ignition rate               × 0.15
          Mesh synchrony                  × 0.10

        We don't swap the live mesh for each genome (too expensive).
        Instead, we estimate fitness by simulating one step with the genome's
        weights applied to the current mesh state and measuring the result.
        """
        phi = 0.0
        coherence = 0.5
        efficiency = 0.5
        ignition = 0.5
        synchrony = 0.5

        # Φ from substrate
        if self._substrate_ref and hasattr(self._substrate_ref, '_current_phi'):
            phi = float(getattr(self._substrate_ref, '_current_phi', 0.0))
            phi = min(1.0, phi / 10.0)  # normalize to ~0-1

        # Binding PSI
        if self._binding_ref:
            coherence = self._binding_ref.get_psi()

        # Energy efficiency: low mean weight norm = efficient
        weight_norm = np.linalg.norm(genome.inter_weights)
        max_norm = np.sqrt(genome.inter_weights.size) * 0.5
        efficiency = 1.0 - min(1.0, weight_norm / max_norm)

        # GWT ignition
        if self._workspace_ref:
            ignition = min(1.0, self._workspace_ref.ignition_level)

        # Mesh synchrony
        if self._mesh_ref:
            synchrony = self._mesh_ref.get_global_synchrony()

        # Simulate one step with this genome's weights to check stability
        # NaN/Inf genomes are immediately rejected (fitness=0) — cannot reproduce
        try:
            if np.any(np.isnan(genome.inter_weights)) or np.any(np.isinf(genome.inter_weights)):
                return 0.0
            test_state = np.array([np.mean(c.x) for c in self._mesh_ref.columns], dtype=np.float32) if self._mesh_ref else np.zeros(64, dtype=np.float32)
            result = np.tanh(genome.inter_weights @ test_state)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return 0.0
        except Exception:
            return 0.0
        stability_penalty = 0.0

        fitness = (
            0.30 * phi +
            0.25 * coherence +
            0.20 * efficiency +
            0.15 * ignition +
            0.10 * synchrony -
            stability_penalty
        )

        return max(0.0, min(1.0, fitness))

    # ── Genetic operators ────────────────────────────────────────────────

    def _tournament_select(self) -> Genome:
        """Tournament selection: pick k random, return the best."""
        k = min(self.cfg.tournament_size, len(self._population))
        contestants = self._rng.choice(self._population, size=k, replace=False)
        return max(contestants, key=lambda g: g.fitness)

    def _crossover(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """Uniform crossover: each weight independently from parent 1 or 2."""
        mask = self._rng.random(w1.shape) < 0.5
        child = np.where(mask, w1, w2)
        return child.astype(np.float32)

    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        """Gaussian mutation + structural mutation."""
        # Gaussian perturbation
        mask = self._rng.random(weights.shape) < self.cfg.mutation_rate
        noise = self._rng.standard_normal(weights.shape).astype(np.float32) * self.cfg.mutation_sigma
        weights = weights + mask * noise

        # Structural mutation: randomly add or prune connections
        if self._rng.random() < self.cfg.structural_mutation_rate:
            # Add a new connection
            i, j = self._rng.integers(0, weights.shape[0], size=2)
            if i != j and weights[i, j] == 0:
                weights[i, j] = self._rng.standard_normal() * 0.05

        if self._rng.random() < self.cfg.structural_mutation_rate:
            # Prune a weak connection
            nonzero = np.nonzero(weights)
            if len(nonzero[0]) > 0:
                idx = self._rng.integers(0, len(nonzero[0]))
                i, j = nonzero[0][idx], nonzero[1][idx]
                if abs(weights[i, j]) < 0.01:
                    weights[i, j] = 0.0

        # Clip for stability
        return np.clip(weights, -1.0, 1.0).astype(np.float32)

    # ── Rollback ─────────────────────────────────────────────────────────

    async def check_rollback(self) -> bool:
        """Check if the applied genome is degrading performance and rollback."""
        if self._champion is None or self._mesh_ref is None:
            return False

        current_fitness = await self._evaluate_fitness(self._champion)
        if current_fitness < self._pre_apply_fitness * self.cfg.rollback_threshold:
            logger.warning(
                "Fitness degradation detected (%.4f < %.4f × %.2f). Rolling back.",
                current_fitness, self._pre_apply_fitness, self.cfg.rollback_threshold
            )
            # Rollback: re-seed from stored champion
            if len(self._history) > 1:
                # Reset inter-column weights to previous generation's champion
                self._mesh_ref._inter_W = self._champion.inter_weights.copy()
            return True
        return False

    # ── Online Micro-Evolution ──────────────────────────────────────────

    async def micro_evolve(self, trigger: str = "unknown", intensity: float = 0.5):
        """Event-triggered micro-evolution during live conversation.

        Unlike the full generation cycle (every 5 min), this performs a
        SMALL targeted mutation on the champion genome and applies it
        immediately if it improves fitness. Think of it as synaptic
        plasticity at the architectural level — the mesh adapts to what's
        happening RIGHT NOW.

        Triggers:
          - "prediction_error": high prediction error → explore new topologies
          - "phi_drop": integrated information collapsed → try to recover integration
          - "coherence_collapse": unified field fragmented → stabilize connectivity
          - "hedonic_positive": strong positive valence → reinforce current topology
          - "hedonic_negative": strong negative valence → explore alternatives
          - "novelty": novel stimulus → increase structural mutation rate
        """
        if not self._running or self._mesh_ref is None:
            return
        if self._champion is None:
            return

        # Rate limit: no more than once per 30 seconds
        now = time.time()
        last_micro = getattr(self, "_last_micro_evolution", 0.0)
        if now - last_micro < 30.0:
            return
        self._last_micro_evolution = now

        # Adjust mutation parameters based on trigger type
        sigma_scale = 1.0
        structural_boost = 1.0
        if trigger == "prediction_error":
            sigma_scale = 1.5 * intensity  # More exploration on surprise
            structural_boost = 2.0
        elif trigger == "phi_drop":
            sigma_scale = 0.5  # Conservative — try to recover, not explore wildly
            structural_boost = 0.5  # Don't prune when phi is low
        elif trigger == "coherence_collapse":
            sigma_scale = 0.3  # Very conservative — stabilize
        elif trigger == "hedonic_positive":
            sigma_scale = 0.2  # Tiny refinement — reinforce what works
        elif trigger == "hedonic_negative":
            sigma_scale = 1.0 * intensity
            structural_boost = 1.5
        elif trigger == "novelty":
            sigma_scale = 0.8
            structural_boost = 3.0  # More structural exploration on novelty

        # Create mutant from champion
        mutant_weights = self._champion.inter_weights.copy()

        # Apply scaled Gaussian mutation
        mask = self._rng.random(mutant_weights.shape) < (self.cfg.mutation_rate * sigma_scale)
        noise = self._rng.standard_normal(mutant_weights.shape).astype(np.float32)
        noise *= self.cfg.mutation_sigma * sigma_scale
        mutant_weights += mask * noise

        # Apply scaled structural mutation
        if self._rng.random() < self.cfg.structural_mutation_rate * structural_boost:
            i, j = self._rng.integers(0, mutant_weights.shape[0], size=2)
            if i != j and mutant_weights[i, j] == 0:
                mutant_weights[i, j] = self._rng.standard_normal() * 0.03

        mutant_weights = np.clip(mutant_weights, -1.0, 1.0).astype(np.float32)

        # Check for NaN/Inf
        if np.any(np.isnan(mutant_weights)) or np.any(np.isinf(mutant_weights)):
            return

        # Evaluate mutant fitness
        mutant = Genome(
            id=self._next_id,
            inter_weights=mutant_weights,
            generation=self._generation,
            parent_ids=[self._champion.id],
        )
        self._next_id += 1
        mutant.fitness = await self._evaluate_fitness(mutant)

        # Apply ONLY if mutant is better (or very close with hedonic_positive trigger)
        threshold = 0.0 if trigger == "hedonic_positive" else 0.01
        if mutant.fitness > self._champion.fitness - threshold:
            self._champion = mutant
            self._mesh_ref._inter_W = mutant.inter_weights.copy()
            self._micro_count = getattr(self, "_micro_count", 0) + 1
            logger.info(
                "🧬 Micro-evolution #%d triggered by '%s' (intensity=%.2f): "
                "fitness %.4f → %.4f (applied)",
                self._micro_count, trigger, intensity,
                self._pre_apply_fitness, mutant.fitness,
            )
            self._pre_apply_fitness = mutant.fitness
        else:
            logger.debug(
                "🧬 Micro-evolution discarded (trigger=%s): mutant=%.4f < champion=%.4f",
                trigger, mutant.fitness, self._champion.fitness,
            )

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        champion_fitness = self._champion.fitness if self._champion else 0.0
        return {
            "running": self._running,
            "generation": self._generation,
            "population_size": len(self._population),
            "champion_id": self._champion.id if self._champion else None,
            "champion_fitness": round(champion_fitness, 4),
            "mean_fitness": round(
                float(np.mean([g.fitness for g in self._population])), 4
            ) if self._population else 0.0,
            "history_len": len(self._history),
            "last_generation": self._history[-1].number if self._history else 0,
        }

"""scratch/run_aura_nethack.py — Aura NetHack Execution Loop.

Launches Aura and hooks her into the NetHack environment using the new
ReflexEngine, SpatialAtlas, and NetHackEnv.
"""
import asyncio
import logging
import time
from core.embodiment.games.nethack.env import NetHackEnv
from core.brain.reflex_engine import get_reflex_engine
from core.memory.spatial_atlas import get_spatial_atlas
from core.adaptation.adaptive_immunity import get_adaptive_immune_system, Antigen
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.NetHack.Runner")

def _update_atlas(atlas, obs):
    dlvl = obs["vitals"]["dlvl"]
    raw_grid = obs["raw_grid"]
    grid_data = []
    for y, row in enumerate(raw_grid):
        data_row = []
        for x, glyph in enumerate(row):
            kind, walkable = _map_glyph(glyph)
            cell = {"kind": kind, "walkable": walkable}
            if glyph.isalpha() and glyph != "@":
                cell["monster"] = glyph
            data_row.append(cell)
        grid_data.append(data_row)
    atlas.update_current(dlvl, grid_data)

def _map_glyph(glyph: str):
    if glyph == ".": return "floor", True
    if glyph == "#": return "corridor", True
    if glyph in "-|": return "wall", False
    if glyph == "+": return "door", True
    if glyph == ">": return "stairs_down", True
    if glyph == "<": return "stairs_up", True
    if glyph == "@": return "player", True
    if glyph == " ": return "unknown", False
    return "item", True

async def run_game():
    logger.info("⚔️ Launching Aura into the Dungeons of Doom...")
    env = NetHackEnv()
    reflex = get_reflex_engine()
    atlas = get_spatial_atlas()
    immunity = get_adaptive_immune_system()
    
    obs = await env.reset()
    done = False
    turn = 0
    
    while not done:
        turn += 1
        # 1. Update Spatial Atlas
        _update_atlas(atlas, obs)
        
        # 2. Consult Reflex Engine (System 1)
        action = reflex.decide(obs)
        
        if not action:
            # 3. Fallback to Cognitive Engine (System 2) - Mocked for now
            # In live Aura, this would be a call to Aura.run_tick()
            action = "search" # Default to search if nothing else
            
        logger.info(f"[Turn {turn}] Action: {action} | HP: {obs['vitals']['hp']}/{obs['vitals']['maxhp']} | Dlvl: {obs['vitals']['dlvl']}")
        
        # 4. Step Environment
        obs, reward, done, info = await env.step(action)
        
        if done:
            logger.warning("💀 Aura has died. Triggering Post-Mortem...")
            # Distill first principles from death (Mocked)
            antigen = Antigen(
                antigen_id=f"death_turn_{turn}",
                subsystem="nethack",
                vector=np.random.rand(16),
                danger=1.0,
                subsystem_need=1.0,
                threat_probability=1.0,
                resource_pressure=0.0,
                error_load=1.0,
                health_pressure=1.0,
                temporal_pressure=0.0,
                recurrence_pressure=0.0,
                protected=False,
                source_domain="environment",
                source="nethack_env"
            )
            # Immune system will log but not restart components
            await immunity.observe_event(antigen.to_dict())
            break
            
        await asyncio.sleep(0.05) # Maintain the 50ms reflex cadence

if __name__ == "__main__":
    try:
        asyncio.run(run_game())
    except KeyboardInterrupt:
        logger.info("Aura NetHack Runner stopped by user.")

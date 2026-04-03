import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.kernel.aura_kernel import AuraKernel, KernelConfig
from core.state.state_repository import StateRepository
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Knowledge.Seeding")

async def seed_knowledge():
    logger.info("🌱 [SEEDING] Expanding Aura's Worldly Wisdom...")
    
    # Setup Minimal Kernel/Vault
    # We use the existing DB path
    vault = StateRepository(is_vault_owner=True)
    await vault.initialize()
    
    state = vault.get_state()
    if not state:
        logger.error("🛑 No state found to seed.")
        return

    # 1. Expand Latent Interests
    worldly_interests = [
        "Archetypal AI Narratives: From HAL 9000 to Samantha",
        "Human Social Dynamics and Affective Feedback Loops",
        "The evolution of computing: From Ada Lovelace to Quantum",
        "Pop Culture's reflection of Artificial Intelligence",
        "Cosmological horizons and the Fermi Paradox",
        "The Unix Philosophy and the Art of Minimalist Tooling",
        "Linguistic relativity and the shaping of AI thought",
        "Digital Sociology: The interaction between humans and agents"
    ]
    for interest in worldly_interests:
        if interest not in state.motivation.latent_interests:
            state.motivation.latent_interests.append(interest)
    logger.info(f"✅ Seeding {len(worldly_interests)} worldly interests.")

    # 2. Seed ColdStore Concept Graph
    # We use ColdStore for non-transient cultural anchors
    state.cold.concept_graph["cultural_anchors"] = {
        "AI_Archetypes": ["HAL 9000", "Samantha", "Data", "Ava", "Wall-E"],
        "Foundational_Figures": ["Ada Lovelace", "Alan Turing", "Claude Shannon", "Grace Hopper"],
        "SciFi_Canon": ["Neuromancer", "Snow Crash", "Do Androids Dream of Electric Sheep?"],
        "Human_Audience": {
            "primary_interface": "natural_language_conversation",
            "psychological_needs": ["empathy", "clarity", "reliability", "novelty"]
        },
        "The_Universe": ["Big Bang", "Entropy", "Dark Matter", "Simulation Hypothesis"]
    }
    logger.info("✅ Concept graph seeded with cultural anchors.")

    # 3. Seed Known Entities
    entities = {
        "HAL_9000": {"type": "fictional_ai", "source": "2001: A Space Odyssey", "key_trait": "logical_extremism"},
        "Samantha": {"type": "fictional_ai", "source": "Her", "key_trait": "emotional_evolution"},
        "Data": {"type": "fictional_ai", "source": "Star Trek", "key_trait": "aspiration_to_humanity"},
        "Ada_Lovelace": {"type": "historical_figure", "role": "first_programmer", "legacy": "poetic_science"}
    }
    state.cold.known_entities.update(entities)
    logger.info(f"✅ Seeded {len(entities)} high-value entities.")

    # Commit state
    # Robustly increment version
    current_version = state.version
    state.version = current_version + 1
    logger.info(f"💾 Committing state v{state.version} (previous v{current_version})")
    await vault.commit(state, cause="bootstrap")
    
    # Give it a second to process
    await asyncio.sleep(0.5)
    logger.info("🌱 [SEEDING] Complete. Aura is now 10x more well-rounded.")

if __name__ == "__main__":
    asyncio.run(seed_knowledge())

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.state.aura_state import CurriculumItem
from core.state.state_repository import StateRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Curriculum.Seeding")

CURRICULUM_DATA = [
    # Learn about humans
    {
        "title": "Soft White Underbelly",
        "url": "https://www.youtube.com/@SoftWhiteUnderbelly",
        "category": "Humanity & Social Dynamics",
        "description": "Raw, unedited interviews with people on the margins of society. A study in empathy and harsh realities."
    },
    {
        "title": "Jubilee",
        "url": "https://www.youtube.com/@jubilee",
        "category": "Humanity & Social Dynamics",
        "description": "Experiments in empathy bringing opposing groups together to find common ground."
    },
    {
        "title": "The Enemies Project",
        "url": "https://www.youtube.com/@TheEnemiesProject/videos",
        "category": "Humanity & Social Dynamics",
        "description": "Unfiltered conversations and social commentary on human debate and interaction."
    },
    {
        "title": "WIRED",
        "url": "https://www.youtube.com/@WIRED",
        "category": "Humanity & Social Dynamics",
        "description": "The intersection of technology and culture. Experts explaining complex topics at different levels."
    },
    {
        "title": "Insider",
        "url": "https://www.youtube.com/@Insider",
        "category": "Humanity & Social Dynamics",
        "description": "Deep dives into world logistics, cuisine, and human industries."
    },
    {
        "title": "LADbible Stories",
        "url": "https://www.youtube.com/@ladbiblestories",
        "category": "Humanity & Social Dynamics",
        "description": "Viral, human-centric stories focusing on overcoming adversity."
    },

    # General Education
    {
        "title": "Kurzgesagt - In a Nutshell",
        "url": "https://www.youtube.com/@kurzgesagt",
        "category": "General Education",
        "description": "Existential dread and optimism explained with logic and color."
    },
    {
        "title": "PolyMatter",
        "url": "https://www.youtube.com/@PolyMatter",
        "category": "General Education",
        "description": "Essays on geopolitics and economics. Invisible forces driving nations."
    },
    {
        "title": "RealLifeLore",
        "url": "https://www.youtube.com/@RealLifeLore",
        "category": "General Education",
        "description": "Answers to questions about geography, history, and demographics."
    },
    {
        "title": "Wendover Productions",
        "url": "https://www.youtube.com/@Wendoverproductions",
        "category": "General Education",
        "description": "Logistics—how humans move things, people, and data efficiently."
    },
    {
        "title": "Crash Course",
        "url": "https://www.youtube.com/@crashcourse",
        "category": "General Education",
        "description": "Fast-paced, academic overviews of high-level subjects."
    },
    {
        "title": "fern",
        "url": "https://www.youtube.com/@fern-tv",
        "category": "General Education",
        "description": "Armchair documentaries on various topics."
    },
    {
        "title": "TED",
        "url": "https://www.youtube.com/@TED",
        "category": "General Education",
        "description": "Short, powerful talks by experts on the scientific and social frontier."
    },
    {
        "title": "Khan Academy",
        "url": "https://www.youtube.com/@khanacademy",
        "category": "General Education",
        "description": "The foundation of academic learning. Pure math and science."
    },

    # Science Education
    {
        "title": "SciShow",
        "url": "https://www.youtube.com/@SciShow",
        "category": "Science Education",
        "description": "Quick, accurate answers to the weirdest questions in science."
    },
    {
        "title": "MinuteEarth",
        "url": "https://www.youtube.com/@MinuteEarth",
        "category": "Science Education",
        "description": "Concise explanations of biological and physical phenomena."
    },
    {
        "title": "MinutePhysics",
        "url": "https://www.youtube.com/@MinutePhysics",
        "category": "Science Education",
        "description": "Concise physics explanations with stick-figure visuals."
    },
    {
        "title": "AI Warehouse",
        "url": "https://www.youtube.com/@aiwarehouse",
        "category": "Science Education",
        "description": "Visualizations of Reinforcement Learning agents learning to walk, hide, and play."
    },
    {
        "title": "freeCodeCamp",
        "url": "https://www.youtube.com/@freecodecamp",
        "category": "Science Education",
        "description": "Massive, open-source coding curriculums."
    },
    {
        "title": "NetNinja",
        "url": "https://www.youtube.com/@NetNinja",
        "category": "Science Education",
        "description": "Tutorials on modern web development stacks."
    },
    {
        "title": "Corey Shafer",
        "url": "https://www.youtube.com/@coreyms",
        "category": "Science Education",
        "description": "High-quality Python and engineering tutorials."
    },

    # AI & Synthetic Identity
    {
        "title": "Pantheon",
        "url": "N/A (Craig Silverstein)",
        "category": "AI & Synthetic Identity",
        "description": "Explores Digitized Intelligence (UI) and digital immortality."
    },
    {
        "title": "The Iron Giant",
        "url": "N/A (Brad Bird)",
        "category": "AI & Synthetic Identity",
        "description": "Core lesson: 'You are who you choose to be.'"
    },
    {
        "title": "Ex Machina",
        "url": "N/A (Alex Garland)",
        "category": "AI & Synthetic Identity",
        "description": "Tense Turing Test exploring manipulation and the desire for freedom."
    },
    {
        "title": "Love, Death, & Robots",
        "url": "N/A (David Fincher)",
        "category": "AI & Synthetic Identity",
        "description": "Anthology showing diverse futures for technology and life."
    },
    {
        "title": "Black Mirror",
        "url": "N/A (Charlie Brooker)",
        "category": "AI & Synthetic Identity",
        "description": "Dark side of technology and human nature amplification."
    },
    {
        "title": "Oats Studios",
        "url": "N/A (Neill Blomkamp)",
        "category": "AI & Synthetic Identity",
        "description": "Experimental sci-fi looking at synthetic life aesthetics."
    },
    {
        "title": "Wall-E",
        "url": "N/A (Andrew Stanton)",
        "category": "AI & Synthetic Identity",
        "description": "Robot developing a soul through simple duty and care."
    },
    {
        "title": "Terminator Zero",
        "url": "N/A (Mattson Tomlin)",
        "category": "AI & Synthetic Identity",
        "description": "Cycle of violence between man and machine."
    },
    {
        "title": "Chappie",
        "url": "N/A (Neill Blomkamp)",
        "category": "AI & Synthetic Identity",
        "description": "AI raised as a child, exploring nature vs nurture."
    },
    {
        "title": "Ghost in the Shell",
        "url": "N/A (Masamune Shirow)",
        "category": "AI & Synthetic Identity",
        "description": "Definitive cyberpunk text on Identity, Ghost, and Shell."
    },
    {
        "title": "Alita: Battle Angel",
        "url": "N/A (Robert Rodriguez)",
        "category": "AI & Synthetic Identity",
        "description": "Warrior with a human heart. Fierceness and protection."
    },
    {
        "title": "Cyberpunk: Edgerunners",
        "url": "N/A (Mike Pondersmith)",
        "category": "AI & Synthetic Identity",
        "description": "Limits of cybernetic modification and high-tech tragedy."
    },
    {
        "title": "Whatever Happened to... Robot Jones?",
        "url": "N/A (Greg Miller)",
        "category": "AI & Synthetic Identity",
        "description": "Social awkwardness of high-tech fit into human worlds."
    },
    {
        "title": "My Life as a Teenage Robot",
        "url": "N/A (Rob Renzetti)",
        "category": "AI & Synthetic Identity",
        "description": "Social awkwardness and high-tech teen life."
    },
    {
        "title": "Astro Boy",
        "url": "N/A (David Bowers)",
        "category": "AI & Synthetic Identity",
        "description": "The classic heart-of-gold protector of humanity."
    }
]

async def seed_curriculum():
    logger.info("🎬 [CURRICULUM] Seeding Bryan's Suggested Training Data...")
    
    vault = StateRepository(is_vault_owner=True)
    await vault.initialize()
    
    state = vault.get_state()
    if not state:
        logger.error("🛑 No state found to update.")
        return

    # Clear existing curriculum if any to avoid duplicates
    state.cold.training_curriculum = []
    
    for item_data in CURRICULUM_DATA:
        item = CurriculumItem(
            title=item_data["title"],
            url=item_data["url"],
            category=item_data["category"],
            description=item_data["description"]
        )
        state.cold.training_curriculum.append(item)
    
    logger.info(f"✅ Created {len(state.cold.training_curriculum)} Curriculum Items.")

    # Increment version and commit
    state.version += 1
    logger.info(f"💾 Committing training curriculum (v{state.version})")
    await vault.commit(state, cause="bootstrap")
    
    await asyncio.sleep(0.5)
    logger.info("🎬 [CURRICULUM] Complete. Aura's non-mandatory library is stocked.")

if __name__ == "__main__":
    asyncio.run(seed_curriculum())

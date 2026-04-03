"""main_daemon.py
────────────────
Standard entry point for the Aura Cognitive Daemon.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.daemon import main

if __name__ == "__main__":
    # Setup basic logging for the daemon process itself
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/aura_daemon.log", mode='a')
        ]
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.critical("Daemon crashed: %s", e)
        sys.exit(1)

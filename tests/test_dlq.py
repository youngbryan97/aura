import pytest
################################################################################

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.container import ServiceContainer
from core.service_registration import register_all_services



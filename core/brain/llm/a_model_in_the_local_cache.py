"""Find an already-cached model configuration without contacting its repository.

The returned snapshot is a local candidate, not proof of complete weights or
serving authority. Normal artifact inspection and model admission still apply.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from core.verify import invariant

logger = logging.getLogger(__name__)
_REPOSITORY = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')


def model_in_local_cache(repository: str) -> Path | None:
    """Read the cache's current revision only; a missing file stays unavailable."""
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        located = try_to_load_from_cache(repository, 'config.json', revision='main')
        if not isinstance(located, str):
            return None
        config = Path(located)
        snapshot = config.parent
        if (config.name != 'config.json' or snapshot.parent.name != 'snapshots'
                or not re.fullmatch(r'[0-9a-f]{40,64}', snapshot.name) or not config.is_file()):
            return None
        return snapshot.resolve()
    except (ImportError, OSError, ValueError) as error:
        logger.debug('local model-cache lookup unavailable: %s', error)
        return None


@invariant('llm.local_model_cache_rejects_path_requests', scope='llm',
           owner='core/brain/llm/a_model_in_the_local_cache.py', observational=False)
def _local_model_cache_invariant() -> tuple:
    for request in ('', '/tmp/model', '../model', 'owner/../model', 'owner/model/other', True):
        assert model_in_local_cache(request) is None
    return ()

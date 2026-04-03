from .web_search import EnhancedWebSearchSkill


class FreeSearchSkill(EnhancedWebSearchSkill):
    """Compatibility wrapper for legacy 'free_search' skill.
    Redirects to EnhancedWebSearchSkill.
    """

    name = "free_search"
    description = (
        "Search the internet freely for any information, news, or research. "
        "Returns a list of result snippets. Set deep=True to read the first result in detail."
    )

# Alias for compatibility
free_search = FreeSearchSkill

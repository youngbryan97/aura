import difflib

def fuzzy_match_ratio(s1: str, s2: str) -> float:
    """Returns the similarity ratio between two strings using SequenceMatcher."""
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

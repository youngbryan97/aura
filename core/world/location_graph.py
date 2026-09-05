"""core/world/location_graph.py
Location topology graph tracking folder hierarchies and physical host device contexts.
"""
from typing import Dict, List


class LocationGraph:
    """Tracks system directories, workspace scopes, and host paths."""

    def __init__(self):
        self._directories: Dict[str, List[str]] = {}

    def register_directory(self, path: str, child_folders: List[str]) -> None:
        self._directories[path] = child_folders

    def get_children(self, path: str) -> List[str]:
        return self._directories.get(path, [])

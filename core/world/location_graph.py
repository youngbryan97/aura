"""core/world/location_graph.py
Location topology graph tracking folder hierarchies and physical host device contexts.
"""


class LocationGraph:
    """Tracks system directories, workspace scopes, and host paths."""

    def __init__(self):
        self._directories: dict[str, list[str]] = {}

    def register_directory(self, path: str, child_folders: list[str]) -> None:
        self._directories[path] = child_folders

    def get_children(self, path: str) -> list[str]:
        return self._directories.get(path, [])

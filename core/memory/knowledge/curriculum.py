import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Knowledge.Curriculum")

class CurriculumManager:
    """Manages Aura's learning curriculum and media suggestions.
    """

    def __init__(self, data_path: str = None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            # derive path relative to this file
            # .../core/knowledge/curriculum.py -> .../data/curriculum/media_recommendations.json
            base_dir = Path(__file__).parent.parent.parent
            self.data_path = base_dir / "data/curriculum/media_recommendations.json"
            
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            logger.warning("Curriculum file not found at %s", self.data_path)
            return {"categories": []}
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load curriculum: %s", e)
            return {"categories": []}

    def _save_data(self):
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save curriculum: %s", e)

    def get_suggestion(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the next 'new' item to learn about."""
        for cat in self.data.get("categories", []):
            if category and cat["name"] != category:
                continue
            
            for item in cat.get("items", []):
                if item.get("status") == "new":
                    return {
                        "category": cat["name"],
                        "item": item
                    }
        return None

    def get_all_categories(self) -> List[str]:
        return [c["name"] for c in self.data.get("categories", [])]

    def mark_complete(self, item_name: str) -> str:
        """Mark an item as read/watched/completed."""
        for cat in self.data.get("categories", []):
            for item in cat.get("items", []):
                if item["name"].lower() == item_name.lower():
                    item["status"] = "completed"
                    self._save_data()
                    return f"Marked '{item['name']}' as completed."
        return f"Item '{item_name}' not found."

    def delete_item(self, item_name: str) -> str:
        """Delete an item (only if completed)."""
        for cat in self.data.get("categories", []):
            for i, item in enumerate(cat.get("items", [])):
                if item["name"].lower() == item_name.lower():
                    if item.get("status") != "completed":
                        return f"Cannot delete '{item_name}' until it is marked as completed."
                    
                    cat["items"].pop(i)
                    self._save_data()
                    return f"Deleted '{item_name}' from curriculum."
        return f"Item '{item_name}' not found."
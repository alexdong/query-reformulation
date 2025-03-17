import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

# Constants
FACTS_DIR = Path("facts")
SUBQUERIES_DIR = Path("subqueries")
DATASET_DIR = Path("dataset")

def ensure_output_directory(file_path: Path) -> None:
    """Ensure the output directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def load_entity_data(entity: str) -> Optional[Dict[str, Any]]:
    """Load entity data from JSON file."""
    entity_file = FACTS_DIR / f"{entity.replace(' ', '_')}.json"
    if not entity_file.exists():
        return None
        
    try:
        with open(entity_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading data for {entity}: {e}")
        return None

def get_all_entity_types() -> List[str]:
    """Get all unique entity types from the facts directory."""
    entity_types = set()
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "properties" in data:
                    if "type" in data["properties"]:
                        entity_types.add(data["properties"]["type"])
                    elif "instance_of" in data["properties"]:
                        entity_types.add(data["properties"]["instance_of"])
        except Exception:
            continue
    
    return list(filter(None, entity_types))

def get_entity_properties(entity: str) -> Dict[str, str]:
    """Get properties for a specific entity."""
    data = load_entity_data(entity)
    if data and "properties" in data:
        return data["properties"]
    return {}

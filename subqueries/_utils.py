import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    with open(entity_file, "r", encoding="utf-8") as f:
        return json.load(f)

def get_all_entity_types() -> List[str]:
    """Get all unique entity types from the facts directory."""
    entity_types = set()
    for file_path in FACTS_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            entity_types.add(data["properties"]["type"])

    return list(filter(None, entity_types))

def get_entity_properties(entity: str) -> Dict[str, str]:
    """Get properties for a specific entity."""
    data = load_entity_data(entity)
    return data["properties"]

def get_entity_relationships(entity: str) -> List[Dict[str, str]]:
    """Get relationships for a specific entity."""
    data = load_entity_data(entity)
    return data["relationship"]

def get_entity_type(entity: str) -> str:
    """Get the type of an entity."""
    data = load_entity_data(entity)
    return data["properties"]["type"]

def get_all_entities() -> List[str]:
    """Get all entity names from the facts directory."""
    entities = []
    for file_path in FACTS_DIR.glob("*.json"):
        entities.append(file_path.stem.replace("_", " "))
    return entities

if __name__ == "__main__":
    print(get_all_entity_types())
    print(get_entity_properties("Dunedin"))
    print(get_entity_relationships("Dunedin"))
    print(get_entity_type("Dunedin"))
    print(get_all_entities())

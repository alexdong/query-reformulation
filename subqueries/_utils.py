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

def get_entity_relationships(entity: str) -> List[Dict[str, str]]:
    """Get relationships for a specific entity."""
    data = load_entity_data(entity)
    if data and "relationship" in data:
        return data["relationship"]
    return []

def get_entity_type(entity: str) -> str:
    """Get the type of an entity."""
    data = load_entity_data(entity)
    if data and "properties" in data:
        if "type" in data["properties"]:
            return data["properties"]["type"]
        elif "instance_of" in data["properties"]:
            return data["properties"]["instance_of"]
    return ""

def get_all_entities() -> List[str]:
    """Get all entity names from the facts directory."""
    entities = []
    for file_path in FACTS_DIR.glob("*.json"):
        entities.append(file_path.stem.replace("_", " "))
    return entities

def get_entities_with_relationships() -> List[str]:
    """Get all entities that have relationships."""
    entities = []
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "relationship" in data and data["relationship"]:
                    entities.append(file_path.stem.replace("_", " "))
        except Exception:
            continue
    return entities

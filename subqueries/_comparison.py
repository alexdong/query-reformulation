import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

# Constants
FACTS_DIR = Path("facts")
OUTPUT_FILE = Path("dataset/subqueries-comparison.txt")

def get_entities_by_type(entity_type: str, count: int = 3) -> List[str]:
    """Find entities of the specified type."""
    matching_entities = []
    
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Check if entity has the specified type
                entity_type_value = None
                if "properties" in data:
                    if "type" in data["properties"]:
                        entity_type_value = data["properties"]["type"]
                    elif "instance_of" in data["properties"]:
                        entity_type_value = data["properties"]["instance_of"]
                
                if entity_type_value and entity_type.lower() in entity_type_value.lower():
                    matching_entities.append(file_path.stem.replace("_", " "))
                    
                    # If we have enough entities, we can stop
                    if len(matching_entities) >= count * 2:  # Get more than needed to allow for random selection
                        break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # If we found enough entities, randomly select the requested count
    if len(matching_entities) >= count:
        return random.sample(matching_entities, count)
    
    return matching_entities

def get_common_properties(entities: List[str]) -> List[str]:
    """Find common properties among the given entities."""
    entity_properties = {}
    
    for entity in entities:
        entity_file = FACTS_DIR / f"{entity.replace(' ', '_')}.json"
        if not entity_file.exists():
            continue
            
        try:
            with open(entity_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "properties" in data:
                    entity_properties[entity] = set(data["properties"].keys())
        except Exception as e:
            print(f"Error reading properties for {entity}: {e}")
    
    # Find intersection of all property sets
    if not entity_properties:
        return []
        
    common_props = list(set.intersection(*entity_properties.values()))
    
    # Filter out some common metadata properties that aren't interesting for comparison
    filtered_props = [p for p in common_props if p not in ["type", "instance_of", "description"]]
    
    return filtered_props

def generate_comparison_subqueries(count: int = 1333) -> None:
    """Generate comparison subqueries and write to output file."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Get all unique entity types
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
    
    entity_types = list(filter(None, entity_types))
    
    # Generate subqueries
    subqueries_list = []
    attempts = 0
    max_attempts = count * 10  # Limit attempts to avoid infinite loops
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while len(subqueries_list) < count and attempts < max_attempts:
            attempts += 1
            
            # Pick a random entity type
            if not entity_types:
                break
                
            entity_type = random.choice(entity_types)
            
            # Get random entities of this type
            num_entities = random.randint(2, 5)  # Between 2 and 5 entities
            entities = get_entities_by_type(entity_type, num_entities)
            
            if len(entities) < 2:
                continue  # Need at least 2 entities for comparison
            
            # Find common properties
            common_props = get_common_properties(entities)
            
            if not common_props:
                continue
                
            # Pick a random common property
            prop = random.choice(common_props)
            
            # Generate subqueries
            subqueries = [f"{entity} {prop}" for entity in entities]
            subquery_text = "\n".join(subqueries)
            
            # Avoid duplicates
            if subquery_text not in subqueries_list:
                subqueries_list.append(subquery_text)
                f.write(f"{subquery_text}\n")
                
                # Print progress
                if len(subqueries_list) % 100 == 0:
                    print(f"Generated {len(subqueries_list)} comparison subqueries")
    
    print(f"Completed generating {len(subqueries_list)} comparison subqueries")

if __name__ == "__main__":
    generate_comparison_subqueries()

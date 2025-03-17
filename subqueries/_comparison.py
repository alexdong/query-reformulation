import json
import random
from typing import List

from _utils import (
    FACTS_DIR,
    SUBQUERIES_DIR,
    ensure_output_directory,
    get_all_entity_types,
    get_entity_properties,
)

OUTPUT_FILE = SUBQUERIES_DIR / "comparison.txt"

def get_entities_by_type(
    entity_type: str, min_count: int = 2, max_count: int = 5
) -> List[str]:
    """Find entities of the specified type."""
    print(f"Finding entities of type '{entity_type}'...")
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
                    # Get more than needed to allow for random selection
                    if len(matching_entities) >= max_count * 3:
                        break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Found {len(matching_entities)} entities of type '{entity_type}'")

    # If we found enough entities, randomly select between min_count and max_count
    if len(matching_entities) >= min_count:
        count = min(
            max(min_count, random.randint(min_count, max_count)),
            len(matching_entities)
        )
        return random.sample(matching_entities, count)

    return matching_entities

def get_common_properties(entities: List[str]) -> List[str]:
    print(f"Finding common properties among {len(entities)} entities...")
    entity_properties = {}
    
    # Track which properties have numeric values
    properties_with_numbers = set()

    for entity in entities:
        props = get_entity_properties(entity)
        if props:
            # Only consider properties with non-empty values
            entity_properties[entity] = {k for k, v in props.items() if v}
            
            # Check which properties have numeric values
            for prop, value in props.items():
                if value and any(char.isdigit() for char in str(value)):
                    properties_with_numbers.add(prop)

    # Find intersection of all property sets
    if not entity_properties or len(entity_properties) < 2:
        return []

    common_props = list(set.intersection(*entity_properties.values()))

    # Filter to only include properties with numeric values and exclude metadata properties
    filtered_props = [
        p for p in common_props 
        if p not in ["type"] and p in properties_with_numbers
    ]

    print(f"Found {len(filtered_props)} common properties with numeric values")
    return filtered_props

def generate(entity_type: str) -> str:
    """Generate a comparison subquery for a given entity type.
    
    Args:
        entity_type: The entity type to generate a comparison for
        
    Returns:
        A string containing the generated subquery or empty string if generation failed
    """
    # Get random entities of this type
    entities = get_entities_by_type(entity_type, 2, 5)

    if len(entities) < 2:
        return ""  # Need at least 2 entities for comparison

    # Find common properties
    common_props = get_common_properties(entities)

    if not common_props:
        return ""

    # Pick a random common property
    prop = random.choice(common_props)

    # Generate subqueries - join with \n to keep on one line
    subqueries = [f"{entity} {prop}" for entity in entities]
    return "\\n".join(subqueries)

def generate_comparison_subqueries(count: int = 1333) -> None:
    """Generate comparison subqueries and write to output file."""
    print(f"Generating {count} comparison subqueries...")
    ensure_output_directory(OUTPUT_FILE)

    # Get all unique entity types
    entity_types = get_all_entity_types()

    if not entity_types:
        print("No entity types found")
        return

    print(f"Found {len(entity_types)} entity types")

    subqueries_list = []
    attempts = 0
    max_attempts = count * 10  # Limit attempts to avoid infinite loops

    while len(subqueries_list) < count and attempts < max_attempts:
        attempts += 1

        # Print progress more frequently
        if attempts % 100 == 0:
            print(
                f"Attempt {attempts}/{max_attempts}, "
                f"generated {len(subqueries_list)}/{count} subqueries"
            )

        # Pick a random entity type
        entity_type = random.choice(entity_types)
        
        # Generate a subquery
        subquery = generate(entity_type)
        
        # If generation was successful and not a duplicate, add to list
        if subquery and subquery not in subqueries_list:
            subqueries_list.append(subquery)

            # Print progress
            if len(subqueries_list) % 50 == 0:
                print(f"Generated {len(subqueries_list)}/{count} comparison subqueries")

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for subquery in subqueries_list:
            f.write(f"{subquery}\n")

    print(f"Completed generating {len(subqueries_list)} comparison subqueries")

if __name__ == "__main__":
    print(generate("River"))
    
    #generate_comparison_subqueries()

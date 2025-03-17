import json
import os
from pathlib import Path
import argparse

from _utils import DATASET_DIR, ensure_output_directory
from _comparison import generate_comparison_subqueries
from _expansion import generate_expansion_subqueries
from _chaining import generate_chaining_subqueries

def main():
    """Run all subquery generators."""
    parser = argparse.ArgumentParser(description="Generate subqueries for training data")
    parser.add_argument(
        "--count", 
        type=int, 
        default=1333, 
        help="Number of subqueries to generate for each type"
    )
    parser.add_argument(
        "--type", 
        choices=["all", "comparison", "expansion", "chaining"],
        default="all",
        help="Type of subqueries to generate"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    print(f"Generating {args.count} subqueries of each requested type...")
    
    if args.type in ["all", "comparison"]:
        print("\n=== Generating Comparison Subqueries ===")
        generate_comparison_subqueries(args.count)
    
    if args.type in ["all", "expansion"]:
        print("\n=== Generating Expansion Subqueries ===")
        generate_expansion_subqueries(args.count)
    
    if args.type in ["all", "chaining"]:
        print("\n=== Generating Chaining Subqueries ===")
        generate_chaining_subqueries(args.count)
    
    print("\nSubquery generation complete!")
    
    # Print summary
    if os.path.exists(DATASET_DIR / "subqueries-comparison.txt"):
        with open(DATASET_DIR / "subqueries-comparison.txt", "r") as f:
            comparison_count = sum(1 for _ in f)
        print(f"Comparison subqueries: {comparison_count}")
    
    if os.path.exists(DATASET_DIR / "subqueries-expansion.txt"):
        with open(DATASET_DIR / "subqueries-expansion.txt", "r") as f:
            expansion_count = sum(1 for _ in f)
        print(f"Expansion subqueries: {expansion_count}")
    
    if os.path.exists(DATASET_DIR / "subqueries-chaining.txt"):
        with open(DATASET_DIR / "subqueries-chaining.txt", "r") as f:
            chaining_count = sum(1 for _ in f)
        print(f"Chaining subqueries: {chaining_count}")

if __name__ == "__main__":
    main()
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from _utils import (
    FACTS_DIR, 
    DATASET_DIR, 
    ensure_output_directory, 
    load_entity_data,
    get_all_entity_types,
    get_entity_properties
)

OUTPUT_FILE = DATASET_DIR / "subqueries-comparison.txt"

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
        props = get_entity_properties(entity)
        if props:
            entity_properties[entity] = set(props.keys())
    
    # Find intersection of all property sets
    if not entity_properties:
        return []
        
    common_props = list(set.intersection(*entity_properties.values()))
    
    # Filter out some common metadata properties that aren't interesting for comparison
    filtered_props = [p for p in common_props if p not in ["type", "instance_of", "description"]]
    
    return filtered_props

def generate_comparison_subqueries(count: int = 1333) -> None:
    """Generate comparison subqueries and write to output file."""
    ensure_output_directory(OUTPUT_FILE)
    
    # Get all unique entity types
    entity_types = get_all_entity_types()
    
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

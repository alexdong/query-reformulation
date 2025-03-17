#implement the expansion subqueries. also refactor the code to extract common functions that also show up in '_comparison.py' into a separate file `_utils.py`. ai!
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from _utils import (
    FACTS_DIR, 
    DATASET_DIR, 
    ensure_output_directory, 
    load_entity_data,
    get_entity_properties
)

OUTPUT_FILE = DATASET_DIR / "subqueries-expansion.txt"

def find_entities_with_csv_properties(min_values: int = 3) -> List[Tuple[str, str, str]]:
    """
    Find entities with properties that contain CSV lists.
    Returns a list of tuples: (entity_name, property_name, property_value)
    """
    entities_with_csv = []
    
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            entity_name = file_path.stem.replace("_", " ")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if "properties" not in data:
                    continue
                    
                for prop_name, prop_value in data["properties"].items():
                    # Skip certain metadata properties
                    if prop_name in ["type", "instance_of", "description"]:
                        continue
                        
                    # Check if property value is a string and contains commas
                    if isinstance(prop_value, str) and "," in prop_value:
                        values = [v.strip() for v in prop_value.split(",")]
                        if len(values) >= min_values:
                            entities_with_csv.append((entity_name, prop_name, prop_value))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return entities_with_csv

def generate_expansion_subqueries(count: int = 1333) -> None:
    """Generate expansion subqueries and write to output file."""
    ensure_output_directory(OUTPUT_FILE)
    
    # Find entities with CSV properties
    csv_properties = find_entities_with_csv_properties()
    
    if not csv_properties:
        print("No entities with CSV properties found")
        return
    
    # Generate subqueries
    subqueries_list = []
    attempts = 0
    max_attempts = count * 10  # Limit attempts to avoid infinite loops
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while len(subqueries_list) < count and attempts < max_attempts:
            attempts += 1
            
            if not csv_properties:
                break
                
            # Pick a random entity with CSV property
            entity_name, prop_name, prop_value = random.choice(csv_properties)
            
            # Split the CSV value
            values = [v.strip() for v in prop_value.split(",")]
            
            # Generate subqueries
            subqueries = [f"{entity_name} {prop_name} {value}" for value in values]
            subquery_text = "\n".join(subqueries)
            
            # Avoid duplicates
            if subquery_text not in subqueries_list:
                subqueries_list.append(subquery_text)
                f.write(f"{subquery_text}\n")
                
                # Print progress
                if len(subqueries_list) % 100 == 0:
                    print(f"Generated {len(subqueries_list)} expansion subqueries")
    
    print(f"Completed generating {len(subqueries_list)} expansion subqueries")

if __name__ == "__main__":
    generate_expansion_subqueries()

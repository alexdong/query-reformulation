import json
import random
import re
from typing import List, Tuple

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from _utils import (
    FACTS_DIR,
    SUBQUERIES_DIR,
    ensure_output_directory,
)

OUTPUT_FILE = SUBQUERIES_DIR / "expansion.txt"

def find_entities_with_csv_properties(
    min_values: int = 3, max_values: int = 10
) -> List[Tuple[str, str, str]]:
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
                        values = [v.strip() for v in prop_value.split(",") if v.strip()]
                        if min_values <= len(values) <= max_values:
                            entities_with_csv.append(
                                (entity_name, prop_name, prop_value)
                            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return entities_with_csv


def generate(entity_name: str) -> str:
    csv_properties = []
    file_path = FACTS_DIR / f"{entity_name.replace(' ', '_')}.json"
    if not file_path.exists():
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for prop_name, prop_value in data["properties"].items():
            #print(prop_name, prop_value)
            # Skip certain metadata properties
            if prop_name in ["type", "location", "description"]:
                continue

            # Check if property value is a string and contains commas
            if isinstance(prop_value, str) and ", " in prop_value:
                values = [v.strip() for v in prop_value.split(", ") if v.strip()]
                #print(values)  
                if len(values) >= 3:
                    csv_properties.append((prop_name, prop_value))
    
    # If no CSV properties found, return empty string
    if not csv_properties:
        return ""
    
    # Pick a random CSV property
    prop_name, prop_value = random.choice(csv_properties)
    #print(prop_name, prop_value)
    
    # Generate subqueries
    subqueries = []
    for value in values:
        subqueries.append(f"{entity_name} {value}")

    # Join with \n to keep on one line
    return "\\n".join(subqueries)

def generate_expansion_subqueries(count: int = 1333) -> None:
    """Generate expansion subqueries and write to output file."""
    ensure_output_directory(OUTPUT_FILE)

    # Get all entity files
    entity_files = list(FACTS_DIR.glob("*.json"))
    assert entity_files, "No entity files found"
    print(f"Found {len(entity_files)} entity files")

    subqueries_list = []
    max_attempts = count * 10  # Limit attempts to avoid infinite loops
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Generating {count} expansion subqueries", total=count)
        
        attempts = 0
        while len(subqueries_list) < count and attempts < max_attempts:
            attempts += 1
            
            # Pick a random entity
            random_file = random.choice(entity_files)
            entity_name = random_file.stem.replace("_", " ")
            
            # Generate a subquery
            subquery = generate(entity_name)
            
            if not subquery or subquery in subqueries_list:
                continue
            
            subqueries_list.append(subquery)
            progress.update(task, completed=len(subqueries_list))

    # Write the subqueries to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(subqueries_list))
    
    print(f"Completed generating {len(subqueries_list)} expansion subqueries")

if __name__ == "__main__":
    # generate("Dunedin")
    # print(generate("Gifu Prefecture"))
    generate_expansion_subqueries()

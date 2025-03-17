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

def clean_csv_values(csv_string: str) -> List[str]:
    """Clean and split CSV values, handling various formats."""
    # Replace semicolons with commas if they're used as separators
    if ";" in csv_string and "," not in csv_string:
        csv_string = csv_string.replace(";", ",")

    # Split by comma
    values = [v.strip() for v in csv_string.split(",")]

    # Remove empty values
    values = [v for v in values if v]

    # Remove any values that are just numbers or very short
    values = [v for v in values if not re.match(r'^\d+$', v) and len(v) > 1]

    return values

def generate(entity_name: str) -> str:
    """Generate an expansion subquery for a given entity.
    
    Args:
        entity_name: The name of the entity to generate expansion for
        
    Returns:
        A string containing the generated subquery or empty string if generation failed
    """
    # Find CSV properties for this entity
    csv_properties = []
    
    try:
        file_path = FACTS_DIR / f"{entity_name.replace(' ', '_')}.json"
        if not file_path.exists():
            return ""
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if "properties" not in data:
                return ""

            for prop_name, prop_value in data["properties"].items():
                # Skip certain metadata properties
                if prop_name in ["type", "instance_of", "description"]:
                    continue

                # Check if property value is a string and contains commas
                if isinstance(prop_value, str) and "," in prop_value:
                    values = [v.strip() for v in prop_value.split(",") if v.strip()]
                    if len(values) >= 3:
                        csv_properties.append((prop_name, prop_value))
    except Exception as e:
        print(f"Error processing {entity_name}: {e}")
        return ""
    
    # If no CSV properties found, return empty string
    if not csv_properties:
        return ""
    
    # Pick a random CSV property
    prop_name, prop_value = random.choice(csv_properties)
    
    # Split and clean the CSV value
    values = clean_csv_values(prop_value)

    # Skip if we don't have enough valid values
    if len(values) < 3:
        return ""

    # Generate subqueries
    subqueries = []
    for value in values:
        # Format the subquery based on the property and value
        if prop_name.lower() in [
            "uses", "applications", "types", "categories", "examples"
        ]:
            subqueries.append(f"{entity_name} {value}")
        else:
            subqueries.append(f"{entity_name} {prop_name} {value}")

    # Limit to a reasonable number of subqueries
    if len(subqueries) > 8:
        subqueries = random.sample(subqueries, 8)

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
    generate("Dunedin")
    #generate_expansion_subqueries()

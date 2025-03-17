import json
import random
from typing import List, Tuple

from _utils import (
    FACTS_DIR,
    SUBQUERIES_DIR,
    generate_subqueries_with_progress,
    random_entity_selector,
)

OUTPUT_FILE = SUBQUERIES_DIR / "expansion.txt"

def find_entities_with_csv_properties(
    min_values: int = 3, max_values: int = 10,
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
                                (entity_name, prop_name, prop_value),
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
    # Get all entity files
    entity_files = list(FACTS_DIR.glob("*.json"))
    assert entity_files, "No entity files found"
    print(f"Found {len(entity_files)} entity files")
    
    generate_subqueries_with_progress(
        count=count,
        generator_func=generate,
        output_file=OUTPUT_FILE,
        description="expansion subqueries",
        entity_selector=random_entity_selector,
    )

if __name__ == "__main__":
    # generate("Dunedin")
    # print(generate("Gifu Prefecture"))
    generate_expansion_subqueries()

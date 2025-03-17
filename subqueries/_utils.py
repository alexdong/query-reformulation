import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

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
    """Get properties for a specific entity, excluding the 'type' property."""
    data = load_entity_data(entity)
    if not data or "properties" not in data:
        return {}
    
    # Return all properties except 'type'
    return {k: v for k, v in data["properties"].items() if k != "type"}

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

def generate_subqueries_with_progress(
    count: int,
    generator_func: Callable,
    output_file: Path,
    description: str,
    entity_selector: Callable,
) -> None:
    """Generate subqueries with a progress bar and write to output file.
    
    Args:
        count: Number of subqueries to generate
        generator_func: Function that generates a single subquery
        output_file: Path to write the output to
        description: Description for the progress bar
        entity_selector: Function that selects an entity to generate from
    """
    ensure_output_directory(output_file)
    
    subqueries_list = []
    max_attempts = count * 10  # Limit attempts to avoid infinite loops
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Generating {count} {description}", total=count)
        
        attempts = 0
        while len(subqueries_list) < count and attempts < max_attempts:
            attempts += 1
            
            # Select an entity and generate a subquery
            entity = entity_selector()
            subquery = generator_func(entity)
            
            if not subquery or subquery in subqueries_list:
                continue
            
            subqueries_list.append(subquery)
            progress.update(task, completed=len(subqueries_list))

    # Write the subqueries to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(subqueries_list))
    
    print(f"Completed generating {len(subqueries_list)} {description}")

def random_entity_selector() -> str:
    """Select a random entity from the facts directory."""
    entity_files = list(FACTS_DIR.glob("*.json"))
    random_file = random.choice(entity_files)
    return random_file.stem.replace("_", " ")

def random_entity_type_selector(entity_types: List[str]) -> str:
    """Select a random entity type."""
    return random.choice(entity_types)

def format_property_name(prop: str) -> str:
    """Convert property name to human readable format.
    
    Examples:
        area_km2 -> Area km²
        population_density -> Population density
        gdp_per_capita -> GDP per capita
    """
    # Replace underscores with spaces
    formatted = prop.replace('_', ' ')
    
    # Capitalize first letter of each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Handle special cases
    formatted = formatted.replace('Km2', 'km²')
    formatted = formatted.replace('Km 2', 'km²')
    formatted = formatted.replace('Gdp', 'GDP')
    formatted = formatted.replace('Unesco', 'UNESCO')
    formatted = formatted.replace('Oecd', 'OECD')
    
    return formatted

if __name__ == "__main__":
    print(get_all_entity_types())
    print(get_entity_properties("Dunedin"))
    print(get_entity_relationships("Dunedin"))
    print(get_entity_type("Dunedin"))
    print(get_all_entities())

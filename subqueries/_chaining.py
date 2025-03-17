import random
from typing import List, Tuple, Optional

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from _utils import (
    FACTS_DIR,
    SUBQUERIES_DIR,
    ensure_output_directory,
    get_entity_properties,
    get_entity_relationships,
    load_entity_data,
)

OUTPUT_FILE = SUBQUERIES_DIR / "chaining.txt"

def traverse_relationship_chain(
    start_entity: str,
    max_depth: int = 5,
) -> List[Tuple[str, str, str]]:
    """
    Traverse relationships starting from an entity.
    Returns a list of tuples: (source_entity, relationship_type, target_entity)
    """
    chain = []
    visited = set()
    current_entity = start_entity

    for _ in range(max_depth):
        if current_entity in visited:
            break

        visited.add(current_entity)

        # Get relationships for the current entity
        relationships = get_entity_relationships(current_entity)
        if not relationships:
            break

        # Pick a random relationship
        relationship = random.choice(relationships)

        # Extract relationship type and target entity
        rel_type = next(iter(relationship.keys()))
        target_entity = relationship[rel_type]

        # Check if target entity exists
        if not load_entity_data(target_entity):
            break

        # Add to chain
        chain.append((current_entity, rel_type, target_entity))

        # Move to the next entity
        current_entity = target_entity

    return chain

def generate(start_entity: str) -> str:
    """Generate a chaining subquery starting from the given entity.
    
    Args:
        start_entity: The entity to start the chain from
        
    Returns:
        A string containing the generated subquery or empty string if generation failed
    """
    # Traverse relationship chain
    chain = traverse_relationship_chain(start_entity)

    if not chain:
        return ""

    # Add a property query at the end if possible
    final_entity = chain[-1][2]
    properties = get_entity_properties(final_entity)

    # Filter out common metadata properties
    filtered_props = {
        k: v for k, v in properties.items()
        if k not in ["type", "instance_of", "description"]
    }

    if filtered_props:
        # Add a property query
        prop_name = random.choice(list(filtered_props.keys()))
        chain.append((final_entity, prop_name, filtered_props[prop_name]))

    # Generate subqueries
    subqueries = []

    # Process relationship chain
    for i, (source, rel_type, target) in enumerate(chain):
        if i < len(chain) - 1 or not filtered_props:
            # For relationships
            source_type = ""
            source_data = load_entity_data(source)
            if source_data and "properties" in source_data:
                if "type" in source_data["properties"]:
                    source_type = source_data["properties"]["type"]
                elif "instance_of" in source_data["properties"]:
                    source_type = source_data["properties"]["instance_of"]

            target_type = ""
            target_data = load_entity_data(target)
            if target_data and "properties" in target_data:
                if "type" in target_data["properties"]:
                    target_type = target_data["properties"]["type"]
                elif "instance_of" in target_data["properties"]:
                    target_type = target_data["properties"]["instance_of"]

            if source_type and target_type:
                subqueries.append(f"{source} {rel_type} {target_type}")
            else:
                subqueries.append(f"{source} {rel_type} {target}")
        else:
            # For the final property
            subqueries.append(f"{source} {rel_type}")

    # Join with \n to keep on one line
    return "\\n".join(subqueries)

def generate_chaining_subqueries(count: int = 1333) -> None:
    """Generate chaining subqueries and write to output file."""
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
        task = progress.add_task(f"Generating {count} chaining subqueries", total=count)
        
        attempts = 0
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            while len(subqueries_list) < count and attempts < max_attempts:
                attempts += 1
                
                # Pick a random entity to start with
                random_file = random.choice(entity_files)
                start_entity = random_file.stem.replace("_", " ")
                
                # Generate a subquery
                subquery = generate(start_entity)
                
                if not subquery or subquery in subqueries_list:
                    continue
                
                subqueries_list.append(subquery)
                f.write(f"{subquery}\n")
                progress.update(task, completed=len(subqueries_list))
    
    print(f"Completed generating {len(subqueries_list)} chaining subqueries")

if __name__ == "__main__":
    generate_chaining_subqueries()

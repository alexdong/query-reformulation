import random
from typing import List, Tuple

from _utils import (
    FACTS_DIR,
    SUBQUERIES_DIR,
    format_property_name,
    generate_subqueries_with_progress,
    get_entity_properties,
    get_entity_relationships,
    load_entity_data,
    random_entity_selector,
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
        assert relationships, f"No relationships found for {current_entity}"

        # Find relationships where the target entity exists
        valid_relationships = []
        for rel in relationships:
            rel_type = next(iter(rel.keys()))
            target_entity = rel[rel_type]
            
            if target_entity and load_entity_data(target_entity):
                valid_relationships.append(rel)

        # If no valid relationships found, break
        if not valid_relationships:
            break

        # Pick a random valid relationship
        relationship = random.choice(valid_relationships)

        # Extract relationship type and target entity
        rel_type = next(iter(relationship.keys()))
        target_entity = relationship[rel_type]

        # Add to chain
        chain.append((current_entity, rel_type, target_entity))

        # Move to the next entity
        current_entity = target_entity

    return chain

def generate(start_entity: str) -> str:
    # Traverse relationship chain
    chain = traverse_relationship_chain(start_entity)
    print(chain)
    if not chain:
        return ""

    # Generate subqueries
    subqueries = []
    chain_length = random.randint(2, 3)
    while len(subqueries) < chain_length:
        source, rel_type, target = random.choice(chain)
        # Add `target` entity's type to the subquery, ai!
        subqueries.append(f"{source} {format_property_name(rel_type)}")

    # Add a property query at the end if possible
    final_entity = chain[-1][2]
    properties = get_entity_properties(final_entity)

   # Always end with the last target's property query
    target_entity = get_entity_properties(target)
    subqueries.append(f"{final_entity} {format_property_name(random.choice(list(properties.keys())))}")

    # Join with \n to keep on one line
    return "\\n".join(subqueries)

def generate_chaining_subqueries(count: int = 1333) -> None:
    """Generate chaining subqueries and write to output file."""
    # Get all entity files
    entity_files = list(FACTS_DIR.glob("*.json"))
    assert entity_files, "No entity files found"
    print(f"Found {len(entity_files)} entity files")
    
    generate_subqueries_with_progress(
        count=count,
        generator_func=generate,
        output_file=OUTPUT_FILE,
        description="chaining subqueries",
        entity_selector=random_entity_selector,
    )

if __name__ == "__main__":
    #generate("Dunedin")
    generate("Clutha River")
    #generate_chaining_subqueries()

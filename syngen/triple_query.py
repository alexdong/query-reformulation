import random
from typing import Dict, List, Any, Tuple, Optional

from syngen.wikidata import (
    get_random_entity,
    get_entity_details,
    get_entity_label,
    get_property_label,
    get_interesting_properties,
)

def generate_triple_query() -> Tuple[str, Dict[str, Any]]:
    """
    Generate a simple triple query based on a random WikiData entity.
    
    Returns:
        Tuple containing the generated query and metadata about the generation process
    """
    # Step 1: Get a random entity
    entity_id = get_random_entity()
    
    # Step 2: Get entity details
    entity = get_entity_details(entity_id)
    entity_label = get_entity_label(entity)
    print(f"[INFO] Selected entity: {entity_label} ({entity_id})")
    
    # Step 3: Get interesting properties
    properties = get_interesting_properties(entity)
    if not properties:
        print("[WARN] No interesting properties found for this entity")
        return generate_triple_query()  # Try again with a different entity
    
    # Step 4: Select a random property
    property_id = random.choice(properties)
    property_label = get_property_label(property_id)
    print(f"[INFO] Selected property: {property_label} ({property_id})")
    
    # Step 5: Generate the query
    query = f"{entity_label} {property_label.lower()}"
    print(f"[SUCCESS] Generated query: {query}")
    
    # Return the query and metadata
    metadata = {
        "entity_id": entity_id,
        "entity_label": entity_label,
        "property_id": property_id,
        "property_label": property_label,
    }
    
    return query, metadata

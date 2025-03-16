import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
import httpx

def get_api_key() -> str:
    """Get the Google Knowledge Graph API key from environment variables."""
    api_key = os.getenv("GOOGLE_KNOWLEDGE_BASE_API_KEY")
    assert api_key, "GOOGLE_KNOWLEDGE_BASE_API_KEY environment variable not set"
    return api_key

def search_entity(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for entities in Google Knowledge Graph.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of entity dictionaries
    """
    api_key = get_api_key()
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    
    params = {
        "query": query,
        "key": api_key,
        "limit": limit,
        "indent": True
    }
    
    print(f"[INFO] Searching for entity: {query}")
    
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "itemListElement" in data:
            results = []
            for item in data["itemListElement"]:
                if "result" in item:
                    results.append(item["result"])
            
            print(f"[INFO] Found {len(results)} entities for query: {query}")
            return results
        else:
            print(f"[WARN] No entities found for query: {query}")
            return []
            
    except Exception as e:
        print(f"[ERROR] Error searching for entity: {e}")
        return []

def get_entity_details(entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about an entity.
    
    Args:
        entity_id: The entity ID
        
    Returns:
        Entity details dictionary or None if not found
    """
    api_key = get_api_key()
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    
    params = {
        "ids": entity_id,
        "key": api_key,
        "indent": True
    }
    
    print(f"[INFO] Getting details for entity: {entity_id}")
    
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "itemListElement" in data and data["itemListElement"]:
            return data["itemListElement"][0]["result"]
        else:
            print(f"[WARN] No details found for entity: {entity_id}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error getting entity details: {e}")
        return None

def get_random_entity(seed_query: str = "person") -> Optional[Dict[str, Any]]:
    """
    Get a random entity from Knowledge Graph.
    
    Args:
        seed_query: Initial query to find entities
        
    Returns:
        Random entity dictionary or None if not found
    """
    entities = search_entity(seed_query, limit=10)
    
    if entities:
        random_entity = random.choice(entities)
        print(f"[INFO] Selected random entity: {random_entity.get('name', 'Unknown')}")
        return random_entity
    else:
        print(f"[WARN] No entities found for seed query: {seed_query}")
        return None

def extract_entity_properties(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract useful properties from an entity.
    
    Args:
        entity: Entity dictionary
        
    Returns:
        Dictionary of extracted properties
    """
    properties = {
        "name": entity.get("name", "Unknown"),
        "description": entity.get("description", ""),
        "id": entity.get("@id", ""),
        "type": entity.get("@type", []),
    }
    
    # Extract additional properties if available
    if "detailedDescription" in entity:
        properties["detailed_description"] = entity["detailedDescription"].get("articleBody", "")
        properties["url"] = entity["detailedDescription"].get("url", "")
    
    print(f"[INFO] Extracted properties for entity: {properties['name']}")
    return properties

def generate_triple_query(entity_type: str = "person") -> Tuple[str, List[str]]:
    """
    Generate a query and subqueries based on Knowledge Graph entities.
    
    Args:
        entity_type: Type of entity to search for
        
    Returns:
        Tuple of (main query, list of subqueries)
    """
    # Get a random entity
    main_entity = get_random_entity(entity_type)
    if not main_entity:
        print("[ERROR] Failed to get a main entity")
        return "Who is the current president?", ["current president", "president term"]
    
    # Get related entities
    related_query = main_entity.get("name", "") + " related"
    related_entities = search_entity(related_query, limit=5)
    
    if not related_entities:
        print(f"[WARN] No related entities found for {main_entity.get('name', '')}")
        # Fallback to a simple query
        main_properties = extract_entity_properties(main_entity)
        query = f"What is {main_properties['name']} known for?"
        subqueries = [
            f"{main_properties['name']} achievements",
            f"{main_properties['name']} history"
        ]
        return query, subqueries
    
    # Extract properties
    main_properties = extract_entity_properties(main_entity)
    related_properties = [extract_entity_properties(e) for e in related_entities[:2]]
    
    # Generate query and subqueries
    if "person" in str(main_properties["type"]).lower():
        query = f"What is the relationship between {main_properties['name']} and {related_properties[0]['name']}?"
        subqueries = [
            f"{main_properties['name']} connection to {related_properties[0]['name']}",
            f"{related_properties[0]['name']} relationship with {main_properties['name']}",
            f"{main_properties['name']} and {related_properties[0]['name']} collaboration"
        ]
    else:
        query = f"How does {main_properties['name']} relate to {related_properties[0]['name']}?"
        subqueries = [
            f"{main_properties['name']} connection to {related_properties[0]['name']}",
            f"{related_properties[0]['name']} relationship with {main_properties['name']}",
            f"{main_properties['name']} {related_properties[0]['name']} comparison"
        ]
    
    print(f"[INFO] Generated query: {query}")
    print(f"[INFO] Generated subqueries: {subqueries}")
    
    return query, subqueries

def main() -> None:
    """Main function to demonstrate the module."""
    entity_types = ["person", "organization", "place", "book", "movie"]
    selected_type = random.choice(entity_types)
    
    print(f"[INFO] Generating query for entity type: {selected_type}")
    query, subqueries = generate_triple_query(selected_type)
    
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("SUBQUERIES:")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")
    print("="*50)

if __name__ == "__main__":
    main()

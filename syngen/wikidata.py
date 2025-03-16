import httpx
import random
import os
from typing import Dict, List, Any, Optional, Tuple

def get_random_entity() -> str:
    """Get a random entity ID from WikiData."""
    print("[INFO] Fetching random entity from WikiData...")
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 1
    }
    
    response = httpx.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    entity_id = f"Q{data['query']['random'][0]['id']}"
    print(f"[INFO] Random entity ID: {entity_id}")
    return entity_id

def get_entity_details(entity_id: str) -> Dict[str, Any]:
    """Get details about an entity from WikiData."""
    print(f"[INFO] Fetching details for entity {entity_id}...")
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "props": "claims|labels",
        "languages": "en"
    }
    
    response = httpx.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    return data["entities"][entity_id]

def get_entity_label(entity: Dict[str, Any]) -> str:
    """Extract the English label of an entity."""
    try:
        return entity["labels"]["en"]["value"]
    except KeyError:
        return entity["id"]

def get_property_label(property_id: str) -> str:
    """Get the English label for a property."""
    print(f"[INFO] Fetching label for property {property_id}...")
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": property_id,
        "props": "labels",
        "languages": "en"
    }
    
    response = httpx.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    try:
        return data["entities"][property_id]["labels"]["en"]["value"]
    except KeyError:
        return property_id

def get_interesting_properties(entity: Dict[str, Any]) -> List[str]:
    """Get a list of interesting properties for an entity."""
    # Common interesting properties
    interesting_props = [
        "P569",  # date of birth
        "P570",  # date of death
        "P19",   # place of birth
        "P20",   # place of death
        "P106",  # occupation
        "P27",   # country of citizenship
        "P31",   # instance of
        "P279",  # subclass of
        "P361",  # part of
        "P1343", # described by source
        "P800",  # notable work
    ]
    
    # Filter to only properties that exist for this entity
    available_props = [prop for prop in interesting_props if prop in entity.get("claims", {})]
    
    # If we don't have any of our predefined interesting properties, take some random ones
    if not available_props and entity.get("claims"):
        available_props = list(entity["claims"].keys())[:5]  # Take up to 5 random properties
    
    return available_props

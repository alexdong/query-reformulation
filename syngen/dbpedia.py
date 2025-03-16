import httpx
from typing import Dict, Any, Optional, List, Tuple
import random
from urllib.parse import quote

async def get_random_entity() -> Tuple[str, Dict[str, Any]]:
    """
    Fetch a random entity from DBpedia.
    
    Returns:
        Tuple containing:
            - entity URI (str)
            - entity data (Dict)
    """
    # First, get a list of random resources using the DBpedia random resource service
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://dbpedia.org/sparql", params={
            "default-graph-uri": "http://dbpedia.org",
            "query": """
                SELECT DISTINCT ?entity
                WHERE {
                    ?entity a <http://dbpedia.org/ontology/Person> .
                    ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER (LANG(?label) = 'en')
                }
                ORDER BY RAND()
                LIMIT 10
            """,
            "format": "application/json"
        })
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch random entities: {response.status_code}")
        
        data = response.json()
        entities = [result["entity"]["value"] for result in data["results"]["bindings"]]
        
        if not entities:
            raise Exception("No entities found")
        
        # Select one random entity from the results
        entity_uri = random.choice(entities)
        
        # Now fetch the details for this entity
        entity_data = await get_entity_details(client, entity_uri)
        
        return entity_uri, entity_data

async def get_entity_details(client: httpx.AsyncClient, entity_uri: str) -> Dict[str, Any]:
    """
    Fetch details about a specific entity from DBpedia.
    
    Args:
        client: httpx client to use for the request
        entity_uri: URI of the entity to fetch
        
    Returns:
        Dictionary containing entity properties
    """
    # Encode the entity URI for the query
    encoded_uri = quote(entity_uri, safe='')
    
    response = await client.get("https://dbpedia.org/sparql", params={
        "default-graph-uri": "http://dbpedia.org",
        "query": f"""
            SELECT ?property ?value
            WHERE {{
                <{entity_uri}> ?property ?value .
                FILTER (isLiteral(?value) || ?property = <http://www.w3.org/2000/01/rdf-schema#label>)
            }}
        """,
        "format": "application/json"
    })
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch entity details: {response.status_code}")
    
    data = response.json()
    
    # Process the results into a more usable format
    entity_data = {}
    for result in data["results"]["bindings"]:
        prop = result["property"]["value"]
        value = result["value"]["value"]
        
        if prop not in entity_data:
            entity_data[prop] = []
        
        entity_data[prop].append(value)
    
    return entity_data

async def get_entity_relationships(entity_uri: str) -> List[Dict[str, Any]]:
    """
    Get relationships (links to other entities) for a given entity.
    
    Args:
        entity_uri: URI of the entity
        
    Returns:
        List of related entities with their relationship type
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://dbpedia.org/sparql", params={
            "default-graph-uri": "http://dbpedia.org",
            "query": f"""
                SELECT ?relation ?related_entity ?label
                WHERE {{
                    <{entity_uri}> ?relation ?related_entity .
                    OPTIONAL {{ ?related_entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . 
                              FILTER (LANG(?label) = 'en') }}
                    FILTER (isIRI(?related_entity) && 
                           !STRSTARTS(STR(?relation), "http://www.w3.org/2002/07/owl#") &&
                           !STRSTARTS(STR(?relation), "http://www.w3.org/1999/02/22-rdf-syntax-ns#"))
                }}
                LIMIT 100
            """,
            "format": "application/json"
        })
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch entity relationships: {response.status_code}")
        
        data = response.json()
        
        relationships = []
        for result in data["results"]["bindings"]:
            relationship = {
                "relation": result["relation"]["value"],
                "entity": result["related_entity"]["value"],
                "label": result.get("label", {}).get("value", None)
            }
            relationships.append(relationship)
        
        return relationships

async def search_entities(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Search for entities in DBpedia based on a text query.
    
    Args:
        query: Search term
        limit: Maximum number of results to return
        
    Returns:
        List of matching entities with their URIs and labels
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://dbpedia.org/sparql", params={
            "default-graph-uri": "http://dbpedia.org",
            "query": f"""
                SELECT DISTINCT ?entity ?label
                WHERE {{
                    ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                    FILTER (LANG(?label) = 'en' && REGEX(?label, "{query}", "i"))
                }}
                LIMIT {limit}
            """,
            "format": "application/json"
        })
        
        if response.status_code != 200:
            raise Exception(f"Failed to search entities: {response.status_code}")
        
        data = response.json()
        
        results = []
        for result in data["results"]["bindings"]:
            entity = {
                "uri": result["entity"]["value"],
                "label": result["label"]["value"]
            }
            results.append(entity)
        
        return results

# Example usage in an async context
async def example_usage():
    # Get a random entity
    entity_uri, entity_data = await get_random_entity()
    print(f"Random entity: {entity_uri}")
    
    # Get entity labels if available
    labels = entity_data.get("http://www.w3.org/2000/01/rdf-schema#label", [])
    english_labels = [label for label in labels if label.endswith("@en")]
    if english_labels:
        print(f"Label: {english_labels[0]}")
    
    # Get relationships
    relationships = await get_entity_relationships(entity_uri)
    print(f"Found {len(relationships)} relationships")
    
    # Print a few sample relationships
    for rel in relationships[:5]:
        print(f"Relation: {rel['relation']}")
        print(f"  Entity: {rel['entity']}")
        print(f"  Label: {rel['label']}")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

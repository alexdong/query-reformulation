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
    async with httpx.AsyncClient(timeout=60.0) as client:
        query = """
            SELECT DISTINCT ?entity
            WHERE {
                ?entity a <http://dbpedia.org/ontology/Person> .
                ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                FILTER (LANG(?label) = 'en')
            }
            ORDER BY RAND()
            LIMIT 10
        """
        print("Fetching random entities...")
        
        response = await client.get(
            "https://dbpedia.org/sparql",
            params={
                "default-graph-uri": "http://dbpedia.org",
                "query": query,
                "format": "application/sparql-results+json",
                "timeout": "30000"
            }
        )
        if response.status_code >= 300:
            raise Exception(f"Failed to fetch random entities: {response.status_code}")
        
        data = response.json()
        entities = [result["entity"]["value"] for result in data["results"]["bindings"]]
        print(f"Found {len(entities)} entities")
        
        if not entities:
            raise Exception("No entities found")
        
        # Select one random entity from the results
        print(entities)
        entity_uri = random.choice(entities)
        
        # Now fetch the details for this entity
        print(f"Fetching details for entity: {entity_uri}")
        entity_data = await get_entity_details(client, entity_uri)
        print(f"Entity details: {entity_data}")
        
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
    query = f"""
        SELECT ?property ?value
        WHERE {{
            <{entity_uri}> ?property ?value .
            FILTER (isLiteral(?value) || ?property = <http://www.w3.org/2000/01/rdf-schema#label>)
        }}
    """
    
    response = await client.get(
        "https://dbpedia.org/sparql",
        params={
            "default-graph-uri": "http://dbpedia.org",
            "query": query,
            "format": "application/sparql-results+json",
            "timeout": "30000"
        }
    )
    
    if response.status_code >= 300:
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
    Get meaningful relationships (links to other entities) for a given entity.
    
    Args:
        entity_uri: URI of the entity
        
    Returns:
        List of related entities with their relationship type
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Improved query to filter out common but less useful relationships
        # and focus on more meaningful ontological relationships
        query = f"""
            SELECT ?relation ?related_entity ?label
            WHERE {{
                <{entity_uri}> ?relation ?related_entity .
                OPTIONAL {{ ?related_entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . 
                          FILTER (LANG(?label) = 'en') }}
                FILTER (isIRI(?related_entity))
                FILTER (
                    # Exclude common metadata relationships
                    !STRSTARTS(STR(?relation), "http://www.w3.org/2002/07/owl#") &&
                    !STRSTARTS(STR(?relation), "http://www.w3.org/1999/02/22-rdf-syntax-ns#") &&
                    !STRSTARTS(STR(?relation), "http://www.w3.org/2000/01/rdf-schema#") &&
                    !STRSTARTS(STR(?relation), "http://xmlns.com/foaf/0.1/") &&
                    
                    # Exclude wiki-specific links that aren't meaningful for our use case
                    ?relation != <http://dbpedia.org/ontology/wikiPageWikiLink> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageExternalLink> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageID> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageRevisionID> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageLength> &&
                    ?relation != <http://dbpedia.org/ontology/abstract> &&
                    ?relation != <http://dbpedia.org/ontology/thumbnail> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageRedirects> &&
                    ?relation != <http://dbpedia.org/ontology/wikiPageDisambiguates>
                )
            }}
            ORDER BY ?relation
            LIMIT 200
        """
        
        response = await client.get(
            "https://dbpedia.org/sparql",
            params={
                "default-graph-uri": "http://dbpedia.org",
                "query": query,
                "format": "application/sparql-results+json",
                "timeout": "30000"
            }
        )
        
        if response.status_code >= 300:
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
        
        # If we got too few meaningful relationships, try a more focused query
        # to get specific ontological properties
        if len(relationships) < 5:
            print(f"Found only {len(relationships)} meaningful relationships, trying a more focused query...")
            
            # Query for specific DBpedia ontology properties that are typically meaningful
            focused_query = f"""
                SELECT ?relation ?related_entity ?label
                WHERE {{
                    <{entity_uri}> ?relation ?related_entity .
                    OPTIONAL {{ ?related_entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . 
                              FILTER (LANG(?label) = 'en') }}
                    FILTER (isIRI(?related_entity))
                    FILTER (
                        STRSTARTS(STR(?relation), "http://dbpedia.org/ontology/") ||
                        STRSTARTS(STR(?relation), "http://dbpedia.org/property/")
                    )
                }}
                ORDER BY ?relation
                LIMIT 100
            """
            
            focused_response = await client.get(
                "https://dbpedia.org/sparql",
                params={
                    "default-graph-uri": "http://dbpedia.org",
                    "query": focused_query,
                    "format": "application/sparql-results+json",
                    "timeout": "30000"
                }
            )
            
            if focused_response.status_code >= 300:
                print(f"Failed to fetch focused entity relationships: {focused_response.status_code}")
                return relationships
            
            focused_data = focused_response.json()
            
            for result in focused_data["results"]["bindings"]:
                relationship = {
                    "relation": result["relation"]["value"],
                    "entity": result["related_entity"]["value"],
                    "label": result.get("label", {}).get("value", None)
                }
                # Only add if not already in the list
                if relationship not in relationships:
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        sparql_query = f"""
            SELECT DISTINCT ?entity ?label
            WHERE {{
                ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
                FILTER (LANG(?label) = 'en' && REGEX(?label, "{query}", "i"))
            }}
            LIMIT {limit}
        """
        
        response = await client.get(
            "https://dbpedia.org/sparql",
            params={
                "default-graph-uri": "http://dbpedia.org",
                "query": sparql_query,
                "format": "application/sparql-results+json",
                "timeout": "30000"
            }
        )
        
        if response.status_code >= 300:
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
    #entity_uri, entity_data = await get_random_entity()
    entity_uri = "http://dbpedia.org/resource/Nikola_Tesla"
    # entity = (await search_entities("Nicola Tesla", limit=1))[0]
    # print(entity)

    async with httpx.AsyncClient(timeout=60.0) as client:
        entity_data = await get_entity_details(client, entity_uri)

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
    for rel in relationships:
        print(f"Relation: {rel['relation']}")
        print(f"  Entity: {rel['entity']}")
        print(f"  Label: {rel['label']}")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

import httpx
from typing import Dict, Any, Optional, List, Tuple
import random
from urllib.parse import quote

async def get_random_entity() -> str:
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
        return entity_uri


async def get_entity_relationships(entity_uri: str) -> List[Dict[str, Any]]:
    """
    Get meaningful relationships (links to other entities) for a given entity.
    
    Args:
        entity_uri: URI of the entity
        
    Returns:
        List of related entities with their relationship type
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Focus on relationships defined through DBpedia ontology with other DBpedia resources
        query = f"""
            SELECT ?relation ?related_entity ?label
            WHERE {{
                <{entity_uri}> ?relation ?related_entity .
                # Only include relationships from DBpedia ontology
                FILTER(STRSTARTS(STR(?relation), "http://dbpedia.org/ontology/"))
                
                # Only include related entities that are DBpedia resources
                FILTER(STRSTARTS(STR(?related_entity), "http://dbpedia.org/resource/"))
                
                # Get English labels for the related entities
                OPTIONAL {{ 
                    ?related_entity <http://www.w3.org/2000/01/rdf-schema#label> ?label . 
                    FILTER (LANG(?label) = 'en') 
                }}
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
        
        return relationships


async def get_entity_properties(entity_uri: str) -> Dict[str, Any]:
    """
    Get properties of a given entity.
    
    Args:
        entity_uri: URI of the entity
        
    Returns:
        Dictionary of entity properties
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        query = f"""
            SELECT ?property ?value
            WHERE {{
                <{entity_uri}> ?property ?value .
                FILTER (isLiteral(?value) || ?property = <http://www.w3.org/2000/01/rdf-schema#label>)
                
                # Skip wiki-specific properties that aren't useful for our purposes
                FILTER (
                    ?property != <http://dbpedia.org/ontology/wikiPageWikiLink> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageExternalLink> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageID> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageRevisionID> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageLength> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageRedirects> &&
                    ?property != <http://dbpedia.org/ontology/wikiPageDisambiguates>
                )
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
            raise Exception(f"Failed to fetch entity properties: {response.status_code}")
        
        data = response.json()
        
        entity_properties = {}
        for result in data["results"]["bindings"]:
            prop = result["property"]["value"]
            value = result["value"]["value"]
            
            if prop not in entity_properties:
                entity_properties[prop] = []
            
            entity_properties[prop].append(value)
        
        return entity_properties


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

    properties = await get_entity_properties(entity_uri)
    print(f"Properties of entity: {entity_uri}")
    for prop, values in properties.items():
        print(f"{prop}: {values}")
    
    relationships = await get_entity_relationships(entity_uri)
    print(f"Found {len(relationships)} relationships")
    for rel in relationships:
        print(f"Relation: {rel['relation']}")
        print(f"  Entity: {rel['entity']}")
        print(f"  Label: {rel['label']}")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

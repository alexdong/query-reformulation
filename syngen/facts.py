import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import ollama
from neo4j import GraphDatabase
from rich.console import Console

# Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

console = Console()


def get_entity_info(entity: str, related_entity: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Query LLM to get properties and relationships for an entity.
    Uses the prompt from PROMPT-entity_properties_relationship-extraction.md
    
    Args:
        entity: The entity to query
        related_entity: Optional related entity for context
    
    Returns:
        Dictionary containing entity properties and relationships, or None if failed
    """
    try:
        # Load the prompt template
        with open("PROMPT-entity_properties_relationship-extraction.md", "r") as f:
            prompt_template = f.read()
        
        # Format the prompt with the entity and related entity
        prompt = prompt_template.replace("{{ entity }}", entity)
        
        # If related_entity is provided, use it; otherwise, use a generic context
        context = related_entity if related_entity else "general knowledge"
        prompt = prompt.replace("{{ related_entity }}", context)
        
        console.print(f"[bold blue]Querying LLM for entity:[/] {entity}" + 
                     (f" in context of {related_entity}" if related_entity else ""))
        
        # Query the LLM
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        content = response["message"]["content"]
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        
        if json_start == -1 or json_end == 0:
            console.print(f"[bold red]Failed to extract JSON for entity:[/] {entity}")
            return None
        
        json_str = content[json_start:json_end]
        entity_data = json.loads(json_str)
        
        console.print(f"[bold green]Successfully extracted data for:[/] {entity}")
        return entity_data
    
    except Exception as e:
        console.print(f"[bold red]Error getting entity info for {entity}:[/] {str(e)}")
        return None


def store_entity_in_neo4j(driver, entity_data: Dict[str, Any]) -> None:
    """
    Store entity data in Neo4j database
    
    Args:
        driver: Neo4j driver instance
        entity_data: Dictionary containing entity properties and relationships
    """
    entity_name = entity_data["entity"]
    properties = entity_data.get("properties", {})
    relationships = entity_data.get("relationship", [])
    
    with driver.session() as session:
        # Create entity node if it doesn't exist
        session.run(
            """
            MERGE (e:Entity {name: $name})
            SET e += $properties
            """,
            name=entity_name,
            properties=properties
        )
        
        # Create relationships
        for rel in relationships:
            for rel_type, target_entity in rel.items():
                session.run(
                    """
                    MATCH (e:Entity {name: $source})
                    MERGE (t:Entity {name: $target})
                    MERGE (e)-[r:`%s`]->(t)
                    """ % rel_type,
                    source=entity_name,
                    target=target_entity
                )
        
        console.print(f"[bold green]Stored entity in Neo4j:[/] {entity_name}")


def build_knowledge_graph(
    initial_entity: str, 
    max_entities: int = 100, 
    max_depth: int = 3
) -> None:
    """
    Build a knowledge graph by recursively querying entities and their relationships
    
    Args:
        initial_entity: The starting entity
        max_entities: Maximum number of entities to process
        max_depth: Maximum depth of relationships to traverse
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Queue of entities to process: (entity_name, depth)
    entity_queue: List[Tuple[str, int]] = [(initial_entity, 0)]
    processed_entities: Set[str] = set()
    
    try:
        while entity_queue and len(processed_entities) < max_entities:
            current_entity, depth = entity_queue.pop(0)
            
            if current_entity in processed_entities:
                continue
            
            console.print(f"[bold cyan]Processing entity:[/] {current_entity} (depth: {depth})")
            
            # Get entity info from LLM
            if depth == 0:
                entity_data = get_entity_info(current_entity)
            else:
                # For related entities, provide the parent entity as context
                parent_entity = entity_queue[0][0] if entity_queue else initial_entity
                entity_data = get_entity_info(current_entity, parent_entity)
            
            if not entity_data:
                continue
            
            # Store entity in Neo4j
            store_entity_in_neo4j(driver, entity_data)
            processed_entities.add(current_entity)
            
            # Add related entities to queue if not at max depth
            if depth < max_depth:
                for rel in entity_data.get("relationship", []):
                    for _, target_entity in rel.items():
                        if target_entity not in processed_entities:
                            entity_queue.append((target_entity, depth + 1))
            
            console.print(f"[bold green]Processed:[/] {len(processed_entities)} entities, "
                         f"[bold yellow]Queue:[/] {len(entity_queue)} entities")
    
    except Exception as e:
        console.print(f"[bold red]Error building knowledge graph:[/] {str(e)}")
    
    finally:
        driver.close()
        console.print(f"[bold green]Knowledge graph building complete.[/] "
                     f"Processed {len(processed_entities)} entities.")


def get_random_entity() -> str:
    """
    Get a random entity from a predefined list or external source
    
    Returns:
        A random entity name
    """
    # For simplicity, return from a predefined list
    # In a real implementation, this could query DBpedia or another source
    entities = [
        "The Lord of the Rings",
        "Nikola Tesla",
        "Albert Einstein",
        "Marie Curie",
        "World War II",
        "The Beatles",
        "Leonardo da Vinci",
        "Artificial Intelligence",
        "Climate Change",
        "The Internet"
    ]
    
    import random
    return random.choice(entities)


if __name__ == "__main__":
    # Example usage
    random_entity = get_random_entity()
    console.print(f"[bold magenta]Starting knowledge graph with random entity:[/] {random_entity}")
    
    # Build knowledge graph with random entity
    build_knowledge_graph(random_entity, max_entities=10, max_depth=2)

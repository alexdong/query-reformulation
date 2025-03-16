import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
from rich.console import Console

# Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
FACTS_DIR = Path("facts")

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
        console.print(f"[grey]{json.dumps(entity_data, indent=2)}")
        return entity_data
    
    except Exception as e:
        console.print(f"[bold red]Error getting entity info for {entity}:[/] {str(e)}")
        return None


def store_entity_as_json(entity_data: Dict[str, Any]) -> bool:
    """
    Store entity data as a JSON file in the facts directory
    
    Args:
        entity_data: Dictionary containing entity properties and relationships
    
    Returns:
        True if successful, False otherwise
    """
    try:
        entity_name = entity_data["entity"]
        
        # Create a safe filename from the entity name
        safe_filename = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in entity_name)
        safe_filename = safe_filename.replace(' ', '_')
        
        # Ensure facts directory exists
        FACTS_DIR.mkdir(exist_ok=True)
        
        # Write entity data to JSON file
        file_path = FACTS_DIR / f"{safe_filename}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[bold green]Stored entity as JSON:[/] {file_path}")
        return True
    
    except Exception as e:
        console.print(f"[bold red]Error storing entity as JSON:[/] {str(e)}")
        return False


def get_processed_entities() -> Set[str]:
    """
    Get a set of all entities that have already been processed
    by scanning the facts directory for existing JSON files
    
    Returns:
        Set of entity names that have already been processed
    """
    processed = set()
    
    if not FACTS_DIR.exists():
        return processed
    
    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "entity" in data:
                    processed.add(data["entity"])
        except Exception:
            # Skip files that can't be parsed
            pass
    
    return processed


def build_knowledge_graph(
    initial_entity: str, 
    max_entities: int = 100000, 
    max_depth: int = 300
) -> None:
    """
    Build a knowledge graph by recursively querying entities and their relationships
    and storing them as JSON files
    
    Args:
        initial_entity: The starting entity
        max_entities: Maximum number of entities to process
        max_depth: Maximum depth of relationships to traverse
    """
    # Queue of entities to process: (entity_name, depth)
    entity_queue: List[Tuple[str, int]] = [(initial_entity, 0)]
    
    # Get already processed entities from existing files
    processed_entities: Set[str] = get_processed_entities()
    console.print(f"[bold blue]Found {len(processed_entities)} already processed entities")
    
    try:
        while entity_queue and len(processed_entities) < max_entities:
            current_entity, depth = entity_queue.pop(0)
            
            if current_entity in processed_entities:
                console.print(f"[yellow]Skipping already processed entity:[/] {current_entity}")
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
                console.print(f"[red]Failed to get data for entity:[/] {current_entity}")
                continue
            
            # Store entity as JSON file
            if store_entity_as_json(entity_data):
                processed_entities.add(current_entity)
            
            # Add related entities to queue if not at max depth
            if depth < max_depth:
                for rel in entity_data.get("relationship", []):
                    for _, target_entity in rel.items():
                        if target_entity not in processed_entities:
                            entity_queue.append((target_entity, depth + 1))
            
            console.print(f"[bold green]Processed:[/] {len(processed_entities)} entities, "
                         f"[bold yellow]Queue:[/] {len(entity_queue)} entities")
            
            # Add a small delay to avoid overwhelming the LLM service
            time.sleep(1)
    
    except Exception as e:
        console.print(f"[bold red]Error building knowledge graph:[/] {str(e)}")
    
    finally:
        console.print(f"[bold green]Knowledge graph building complete.[/] "
                     f"Processed {len(processed_entities)} entities.")


def get_random_entity_from_facts() -> Optional[str]:
    """
    Get a random entity from the facts directory
    
    Returns:
        A random entity name or None if no entities exist
    """
    import random
    
    if not FACTS_DIR.exists():
        return None
    
    json_files = list(FACTS_DIR.glob("*.json"))
    if not json_files:
        return None
    
    random_file = random.choice(json_files)
    try:
        with open(random_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("entity")
    except Exception:
        return None


if __name__ == "__main__":
    # Example usage
    random_entity = "Dunedin, New Zealand"
    
    # Try to get a random entity from existing facts
    existing_entity = get_random_entity_from_facts()
    if existing_entity:
        console.print(f"[bold magenta]Using random entity from existing facts:[/] {existing_entity}")
        random_entity = existing_entity
    
    console.print(f"[bold magenta]Starting knowledge graph with entity:[/] {random_entity}")
    
    # Build knowledge graph with the entity
    build_knowledge_graph(random_entity, max_entities=10, max_depth=2)

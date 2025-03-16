import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import ollama
from rich.console import Console

# Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
FACTS_DIR = Path("facts")

console = Console()


def get_entity_type(entity: str) -> Optional[str]:
    try:
        prompt = f"""
        What is the primary type or category of "{entity}"?
        
        Please respond with a single word or short phrase that best categorizes this entity.
        Make sure to use a valid dbpedia or wikidata type/ontology.

        For example:
        - "Albert Einstein" → "Person"
        - "New York City" → "City"
        - "Amazon River" → "River"
        - "World War II" → "Historical Event"
        
        Your response should be just the type, nothing else.
        """
        
        console.print(f"[bold blue]Querying LLM for entity type:[/] {entity}")
        
        # Query the LLM
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the type from response
        entity_type = response["message"]["content"].strip()
        
        console.print(f"[bold green]Retrieved type for:[/] {entity} -> {entity_type}")
        return entity_type
    
    except Exception as e:
        console.print(f"[bold red]Error getting entity type for {entity}:[/] {str(e)}")
        return None


def process_json_files() -> None:
    """
    Process all JSON files in the facts directory to ensure they have a type property.
    """
    if not FACTS_DIR.exists():
        console.print(f"[bold red]Facts directory not found:[/] {FACTS_DIR}")
        return
    
    json_files = list(FACTS_DIR.glob("*.json"))
    console.print(f"[bold blue]Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "entity" not in data:
            # Extract the entity from the file name, ai!
            continue

        properties = data.get("properties", {})
        # Skip files that already have a type property
        if "type" in properties:
            console.print(f"[grey]Skipping file with existing type:[/] {file_path.name}")
            continue
        
        # If it has instance_of property, rename it to type
        if "instance_of" in properties:
            properties["type"] = properties.pop("instance_of")
            console.print(f"[cyan]Renamed instance_of to type for:[/] {file_path.name}")
        
        # If it has neither, add a type property
        else:
            entity_type = get_entity_type(data["entity"])
            properties["type"] = entity_type
            console.print(f"[green]Added type for:[/] {file_path.name} -> {entity_type}")

        # Write the updated data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            data["properties"] = properties
            json.dump(data, f, indent=2, ensure_ascii=False)
            


if __name__ == "__main__":
    console.print("[bold magenta]Starting type property processing[/]")
    process_json_files()
    console.print("[bold green]Processing complete[/]")


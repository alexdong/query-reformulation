import os
from pathlib import Path
from typing import List

import jinja2
from openai import OpenAI

# Constants
QUERIES_DIR = Path("queries")
PROMPTS_DIR = QUERIES_DIR
REFORMULATION_TYPES = ["comparison", "expansion", "chaining"]

# Ensure the queries directory exists
QUERIES_DIR.mkdir(exist_ok=True)

def generate_queries(reformulation_type: str, subqueries: str) -> List[str]:
    """
    Generate queries from subqueries using OpenAI's o3-mini model.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
        subqueries: The subqueries string to reformulate
        
    Returns:
        List of generated queries
    """
    # Set up Jinja2 environment with the correct template loader
    template_loader = jinja2.FileSystemLoader(PROMPTS_DIR)
    template_env = jinja2.Environment(loader=template_loader)
    
    # Load the template for the specific reformulation type
    template = template_env.get_template(f"_PROMPT-{reformulation_type}.md")
    
    # Render the template with the subqueries
    prompt = template.render(subqueries=subqueries)
    print(prompt)
    
    print(f"[INFO] Generated prompt for {reformulation_type}")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a NLP aware specialist that reverses engineer search subqueries into their original search query."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4068,
    )

    # Calculate and print token usage and cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    input_cost = input_tokens * 0.00000115  # $1.15 per 1M tokens
    output_cost = output_tokens * 0.00000440  # $4.40 per 1M tokens
    total_cost = input_cost + output_cost
    
    print(f"[INFO] Token usage: {input_tokens} input tokens, {output_tokens} output tokens")
    print(f"[INFO] Estimated cost: ${input_cost:.6f} (input) + ${output_cost:.6f} (output) = ${total_cost:.6f}")
    
    # Extract the generated queries
    generated_text = response.choices[0].message.content
    
    # Split the response into individual queries
    queries = generated_text.strip().split("\n")
    
    # Clean up the queries (remove empty lines, numbering, etc.)
    cleaned_queries = []
    for query in queries:
        query = query.strip()
        if query and not query.startswith("#") and not query.startswith("```"):
            # Remove any numbering (like "1.", "2.", etc.)
            if query[0].isdigit() and query[1:3] in [". ", ") "]:
                query = query[3:].strip()
            cleaned_queries.append(query)
    
    print(f"[INFO] Generated {cleaned_queries}")
    return cleaned_queries

def save_queries(reformulation_type: str, subqueries: str, queries: List[str]) -> None:
    """
    Save the generated queries to a JSONL file (JSON Lines).
    Each line contains a single query and its corresponding subqueries.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
        subqueries: The original subqueries
        queries: List of generated queries
    """
    import json
    
    output_file = QUERIES_DIR / f"{reformulation_type}.jsonl"
    
    print(f"[INFO] Saving queries to {output_file}")
    
    # Append each query as a separate line
    with open(output_file, "a") as f:
        for query in queries:
            pair = {
                "query": query,
                "subqueries": subqueries
            }
            f.write(json.dumps(pair) + "\n")
    
    print(f"[INFO] Saved {len(queries)} queries to {output_file}")

def process_subqueries_file(reformulation_type: str) -> None:
    """
    Process all subqueries from the corresponding file and generate queries.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
    """
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    
    assert reformulation_type in REFORMULATION_TYPES
    
    # Path to the subqueries file
    subqueries_file = Path("subqueries") / f"{reformulation_type}.txt"
    
    if not subqueries_file.exists():
        raise FileNotFoundError(f"Subqueries file not found: {subqueries_file}")
    
    print(f"[INFO] Processing subqueries from {subqueries_file}")
    
    # Read the subqueries file
    with open(subqueries_file, "r") as f:
        subqueries_list = [line.strip() for line in f if line.strip()]
    
    total_subqueries = len(subqueries_list)
    print(f"[INFO] Found {total_subqueries} subqueries to process")
    
    # Process each subquery with a progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Processing {reformulation_type} subqueries", total=total_subqueries)
        
        for i, subqueries in enumerate(subqueries_list):
            # Generate queries for this subquery
            queries = generate_queries(reformulation_type, subqueries)
            
            # Save the generated queries
            save_queries(reformulation_type, subqueries, queries)
            
            # Update progress
            progress.update(task, advance=1, description=f"[cyan]Processing {reformulation_type} subqueries ({i+1}/{total_subqueries})")
    
    print(f"[INFO] Finished processing all subqueries for {reformulation_type}")


if __name__ == "__main__":
    """
    reformulation_type = "chaining"
    subqueries = "Aoraki Mount Cook Part Of Mountain range\nMountain range Provides Water To River\nRiver Basin Area"
    queries = [
            'What is the area of the river basin for the river that receives water from the mountain range which includes Aoraki Mount Cook?',
            'What is the size of the river basin of the river fed by the mountain range that Aoraki Mount Cook is part of?',
            'What is the area of the basin for the river that is supplied by the mountain range containing Aoraki Mount Cook?',
            'What is the area of the river basin linked to the mountain range that provides water to the river encompassing Aoraki Mount Cook?',
            'What is the area of the basin of the river that gets its water from the mountain range which includes Aoraki Mount Cook?'
            ]

    reformulation_type = "comparison"
    subqueries = "Qantas Foundation Date\nJetstar Foundation Date"
    """

    reformulation_type = "expansion"
    subqueries = "Loreena McKennitt Harp\nLoreena McKennitt Keyboards\nLoreena McKennitt Guitar\nLoreena McKennitt Percussion"

    queries = generate_queries(reformulation_type, subqueries)
    save_queries(reformulation_type, subqueries, queries)

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
    # Load the appropriate prompt template
    prompt_path = PROMPTS_DIR / f"_PROMPT-{reformulation_type}.md"
    assert prompt_path.exists(), f"Prompt template not found: {prompt_path}"
    
    # Render the template with the subqueries
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    # Instead of using Jinja2 template, let's use a simple string replacement
    # This avoids issues with Jinja2 syntax in the template files
    prompt = prompt_template.replace("{{subqueries}}", subqueries)
    
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
    input_cost = input_tokens * 0.00015  # $0.15 per 1000 tokens
    output_cost = output_tokens * 0.0006  # $0.60 per 1000 tokens
    total_cost = input_cost + output_cost
    
    print(f"[INFO] Token usage: {input_tokens} input tokens, {output_tokens} output tokens")
    print(f"[INFO] Estimated cost: ${input_cost:.4f} (input) + ${output_cost:.4f} (output) = ${total_cost:.4f}")
    
    # Extract the generated queries
    generated_text = response.choices[0].message.content
    
    # Split the response into individual queries
    queries = generated_text.strip().split("\n")
    
    print(f"[INFO] Generated {len(queries)} queries")
    return queries

def save_queries(reformulation_type: str, subqueries: str, queries: List[str]) -> None:
    """
    Save the generated queries to the appropriate file.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
        subqueries: The original subqueries
        queries: List of generated queries
    """
    output_file = QUERIES_DIR / f"{reformulation_type}.txt"
    
    print(f"[INFO] Saving queries to {output_file}")
    
    with open(output_file, "a") as f:
        for query in queries:
            f.write(f"{query}===>{subqueries}\n")
    
    print(f"[INFO] Saved {len(queries)} queries to {output_file}")

def process_subqueries_file(reformulation_type: str) -> None:
    """
    Process all subqueries from the corresponding file and generate queries.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
    """
    assert reformulation_type in REFORMULATION_TYPES
    
    # Path to the subqueries file
    subqueries_file = Path("subqueries") / f"{reformulation_type}.txt"
    
    if not subqueries_file.exists():
        raise FileNotFoundError(f"Subqueries file not found: {subqueries_file}")
    
    print(f"[INFO] Processing subqueries from {subqueries_file}")
    
    # Read the subqueries file
    with open(subqueries_file, "r") as f:
        subqueries_list = [line.strip() for line in f if line.strip()]
    
    # Process each subquery
    for i, subqueries in enumerate(subqueries_list):
        print(f"[INFO] Processing subquery {i+1}/{len(subqueries_list)}")
        
        # Generate queries for this subquery
        queries = generate_queries(reformulation_type, subqueries)
        
        # Save the generated queries
        save_queries(reformulation_type, subqueries, queries)
        
        print(f"[INFO] Completed processing subquery {i+1}/{len(subqueries_list)}")
    
    print(f"[INFO] Finished processing all subqueries for {reformulation_type}")

if __name__ == "__main__":
   
    generate_queries("chaining", "Aoraki Mount Cook Part Of Mountain range\nMountain range Provides Water To River\nRiver Basin Area")

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
    Generate 25 queries from subqueries using OpenAI's o3-mini model.
    
    Args:
        reformulation_type: One of "comparison", "expansion", or "chaining"
        subqueries: The subqueries string to reformulate
        
    Returns:
        List of generated queries
    """
    print(f"[INFO] Generating queries for {reformulation_type} with subqueries: {subqueries[:50]}...")
    
    # Load the appropriate prompt template
    prompt_path = PROMPTS_DIR / f"_PROMPT-{reformulation_type}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    # Render the template with the subqueries
    template = jinja2.Template(prompt_template)
    prompt = template.render(subqueries=subqueries)
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Call the API
    print("[INFO] Calling OpenAI API with o3-mini model...")
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates search queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Extract the generated queries
    generated_text = response.choices[0].message.content
    
    # Split the response into individual queries
    queries = []
    for line in generated_text.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("```"):
            # Remove any numbering (like "1.", "2.", etc.)
            if line[0].isdigit() and line[1:3] in [". ", ") "]:
                line = line[3:].strip()
            queries.append(line)
    
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
    if reformulation_type not in REFORMULATION_TYPES:
        raise ValueError(f"Invalid reformulation type: {reformulation_type}. Must be one of {REFORMULATION_TYPES}")
    
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
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python queries/main.py <reformulation_type>")
        print(f"Where <reformulation_type> is one of: {', '.join(REFORMULATION_TYPES)}")
        sys.exit(1)
    
    reformulation_type = sys.argv[1].lower()
    
    try:
        process_subqueries_file(reformulation_type)
        print(f"[SUCCESS] Successfully generated queries for {reformulation_type}")
    except Exception as e:
        print(f"[ERROR] Failed to generate queries: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

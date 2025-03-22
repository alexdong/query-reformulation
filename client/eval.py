import time
from pathlib import Path
from typing import List, Tuple
from client.generate import generate_text

def evaluate_queries() -> Tuple[List[str], float]:
    """
    Read queries from vibe.txt, process each with generate_text function,
    and calculate average processing time.
    
    Returns:
        Tuple containing list of results and average processing time
    """
    # Path to the input file
    input_file = Path("client/vibe.txt")
    
    # Check if file exists
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return [], 0.0
    
    # Read queries from file
    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    if not queries:
        print("No queries found in the file")
        return [], 0.0

    print(f"Warming up the model...")
    generate_text(queries[0])
    
    # Process each query and track time
    results = []
    total_time = 0.0
    
    print(f"Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        start_time = time.time()
        
        # Extract just the reformulated text without the timing info
        result = generate_text(query)
        query_time = time.time()- start_time
        total_time += query_time
        
        # Print progress
        print(f"Query {i+1}/{len(queries)}: {query}")
        print(f"Result: {result}")
        print(f"Time: {query_time:.4f} seconds\n")
        
        results.append(result)
    
    # Calculate median time, ai!
    avg_time = total_time / len(queries) if queries else 0.0
    
    return results, avg_time

if __name__ == "__main__":
    print("Evaluating query reformulation model...")
    results, avg_time = evaluate_queries()
    
    print("\n" + "="*50)
    print(f"Evaluation complete!")
    print(f"Processed {len(results)} queries")
    print(f"Average processing time: {avg_time:.4f} seconds")
    print("="*50)

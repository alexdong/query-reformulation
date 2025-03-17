#!/usr/bin/env python3
"""
Braid input and output files to create a full dataset for query reformulation.

This script takes batch-input and batch-output files for each reformulation type
and creates a combined dataset where each query is paired with its subqueries.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

# Define directories
QUERIES_DIR = Path("queries")
DATASET_DIR = Path("dataset")
REFORMULATION_TYPES = ["comparison", "expansion", "chaining"]

# Ensure the dataset directory exists
DATASET_DIR.mkdir(exist_ok=True, parents=True)


def extract_subqueries(input_file: Path) -> Dict[str, str]:
    """Extract subqueries from input batch file."""
    custom_id_to_subqueries = {}
    
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")
            
            # Extract subqueries from the content field
            content = data.get("body", {}).get("messages", [])[-1].get("content", "")
            
            # Find the subqueries section at the end of the prompt
            match = re.search(r'Ok\. Here is the task:\s*\n\s*(.*?)$', content, re.DOTALL)
            if match:
                subqueries = match.group(1).strip()
                custom_id_to_subqueries[custom_id] = subqueries
    
    return custom_id_to_subqueries


def extract_queries(output_file: Path) -> Dict[str, List[str]]:
    """Extract queries from output batch file."""
    custom_id_to_queries = {}
    
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")
            
            # Extract the assistant's response
            response_content = data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Split the response into individual queries
            queries = [q.strip() for q in response_content.split("\n") if q.strip()]
            
            if queries:
                custom_id_to_queries[custom_id] = queries
    
    return custom_id_to_queries


def braid_files() -> None:
    """Braid input and output files for all reformulation types."""
    print("[INFO] Starting to braid input and output files...")
    
    full_dataset = []
    
    for reformulation_type in REFORMULATION_TYPES:
        print(f"[INFO] Processing {reformulation_type} files...")
        
        input_file = QUERIES_DIR / f"batch-input-{reformulation_type}.jsonl"
        output_file = QUERIES_DIR / f"batch-output-{reformulation_type}.jsonl"
        
        if not input_file.exists() or not output_file.exists():
            print(f"[WARNING] Missing files for {reformulation_type}, skipping...")
            continue
        
        # Extract data from files
        subqueries_dict = extract_subqueries(input_file)
        queries_dict = extract_queries(output_file)
        
        # Combine the data
        for custom_id, subqueries in subqueries_dict.items():
            if custom_id in queries_dict:
                queries = queries_dict[custom_id]
                
                for query in queries:
                    full_dataset.append({
                        "query": query,
                        "subqueries": subqueries,
                    })
    
    # Write the full dataset
    output_file = DATASET_DIR / "full.jsonl"
    with open(output_file, "w") as f:
        for item in full_dataset:
            f.write(json.dumps(item) + "\n")
    
    print(f"[INFO] Created dataset with {len(full_dataset)} examples at {output_file}")


if __name__ == "__main__":
    braid_files()


import argparse
import os

from _chaining import generate_chaining_subqueries
from _comparison import generate_comparison_subqueries
from _expansion import generate_expansion_subqueries
from _utils import DATASET_DIR


def main():
    """Run all subquery generators."""
    parser = argparse.ArgumentParser(description="Generate subqueries for training data")
    parser.add_argument(
        "--count",
        type=int,
        default=1333,
        help="Number of subqueries to generate for each type",
    )
    parser.add_argument(
        "--type",
        choices=["all", "comparison", "expansion", "chaining"],
        default="all",
        help="Type of subqueries to generate",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"Generating {args.count} subqueries of each requested type...")

    if args.type in ["all", "comparison"]:
        print("\n=== Generating Comparison Subqueries ===")
        generate_comparison_subqueries(args.count)

    if args.type in ["all", "expansion"]:
        print("\n=== Generating Expansion Subqueries ===")
        generate_expansion_subqueries(args.count)

    if args.type in ["all", "chaining"]:
        print("\n=== Generating Chaining Subqueries ===")
        generate_chaining_subqueries(args.count)

    print("\nSubquery generation complete!")

    # Print summary
    if os.path.exists(DATASET_DIR / "subqueries-comparison.txt"):
        with open(DATASET_DIR / "subqueries-comparison.txt", "r") as f:
            comparison_count = sum(1 for _ in f)
        print(f"Comparison subqueries: {comparison_count}")

    if os.path.exists(DATASET_DIR / "subqueries-expansion.txt"):
        with open(DATASET_DIR / "subqueries-expansion.txt", "r") as f:
            expansion_count = sum(1 for _ in f)
        print(f"Expansion subqueries: {expansion_count}")

    if os.path.exists(DATASET_DIR / "subqueries-chaining.txt"):
        with open(DATASET_DIR / "subqueries-chaining.txt", "r") as f:
            chaining_count = sum(1 for _ in f)
        print(f"Chaining subqueries: {chaining_count}")

if __name__ == "__main__":
    main()
import json
import os
from pathlib import Path
import argparse

from _utils import SUBQUERIES_DIR, ensure_output_directory
from _comparison import generate_comparison_subqueries
from _expansion import generate_expansion_subqueries
from _chaining import generate_chaining_subqueries

def main():
    """Run all subquery generators."""
    parser = argparse.ArgumentParser(description="Generate subqueries for training data")
    parser.add_argument(
        "--count", 
        type=int, 
        default=1333, 
        help="Number of subqueries to generate for each type"
    )
    parser.add_argument(
        "--type", 
        choices=["all", "comparison", "expansion", "chaining"],
        default="all",
        help="Type of subqueries to generate"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(SUBQUERIES_DIR, exist_ok=True)
    
    print(f"Generating {args.count} subqueries of each requested type...")
    
    if args.type in ["all", "comparison"]:
        print("\n=== Generating Comparison Subqueries ===")
        generate_comparison_subqueries(args.count)
    
    if args.type in ["all", "expansion"]:
        print("\n=== Generating Expansion Subqueries ===")
        generate_expansion_subqueries(args.count)
    
    if args.type in ["all", "chaining"]:
        print("\n=== Generating Chaining Subqueries ===")
        generate_chaining_subqueries(args.count)
    
    print("\nSubquery generation complete!")
    
    # Print summary
    if os.path.exists(SUBQUERIES_DIR / "comparison.txt"):
        with open(SUBQUERIES_DIR / "comparison.txt", "r") as f:
            comparison_count = sum(1 for _ in f)
        print(f"Comparison subqueries: {comparison_count}")
    
    if os.path.exists(SUBQUERIES_DIR / "expansion.txt"):
        with open(SUBQUERIES_DIR / "expansion.txt", "r") as f:
            expansion_count = sum(1 for _ in f)
        print(f"Expansion subqueries: {expansion_count}")
    
    if os.path.exists(SUBQUERIES_DIR / "chaining.txt"):
        with open(SUBQUERIES_DIR / "chaining.txt", "r") as f:
            chaining_count = sum(1 for _ in f)
        print(f"Chaining subqueries: {chaining_count}")

if __name__ == "__main__":
    main()

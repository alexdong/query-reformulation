import json
from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console
from rich.progress import track
from transformers import T5Tokenizer

# Constants
SUBQUERIES_DIR = Path("subqueries")
DATASETS_DIR = Path("datasets")
REFORMULATION_TYPES = ["comparison", "expansion", "chaining"]

console = Console()

def count_query_tokens_in_file(tokenizer: T5Tokenizer) -> List[int]:
    token_counts = []
    with open(DATASETS_DIR / "full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            query = data.get("query")
            tokens = tokenizer.encode(query)
            token_counts.append(len(tokens))
    return token_counts


def count_tokens_in_file(file_path: Path, tokenizer: T5Tokenizer) -> List[int]:
    """Count tokens for each line in the file."""
    token_counts = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in track(lines, description=f"Processing {file_path.name}"):
        line = line.strip()
        if not line:
            continue

        tokens = tokenizer.encode(line)
        token_counts.append(len(tokens))

    return token_counts

def main() -> None:
    """Main function to analyze token counts and plot distributions."""
    console.print("[bold blue]Analyzing token counts in reformulation files...[/bold blue]")

    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)

    counts = count_query_tokens_in_file(tokenizer)
    display_stats(counts, "Query")

    for reformulation_type in REFORMULATION_TYPES:
        file_path = SUBQUERIES_DIR / f"{reformulation_type}.txt"
        counts = count_tokens_in_file(file_path, tokenizer)
        display_stats(counts, reformulation_type)



def display_stats(counts: List[int], title: str) -> None:
    p90 = np.percentile(counts, 90)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)

    console.print(f"[green]{title.capitalize()} Statistics:[/green]")
    console.print(f"  Total lines: {len(counts)}")
    console.print(f"  Mean tokens: {np.mean(counts):.2f}")
    console.print(f"  Median tokens: {np.median(counts):.2f}")
    console.print(f"  Max tokens: {np.max(counts)}")
    console.print(f"  Min tokens: {np.min(counts)}")
    console.print(f"  P90 tokens: {p90:.2f}")
    console.print(f"  P95 tokens: {p95:.2f}")
    console.print(f"  P99 tokens: {p99:.2f}")

if __name__ == "__main__":
    main()

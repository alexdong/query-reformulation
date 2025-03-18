import numpy as np
from pathlib import Path
from transformers import T5Tokenizer
from typing import Dict, List
import seaborn as sns
from rich.console import Console
from rich.progress import track

# Constants
SUBQUERIES_DIR = Path("subqueries")
REFORMULATION_TYPES = ["comparison", "expansion", "chaining"]

console = Console()

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
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Count tokens for each reformulation type
    token_counts = {}
    for reformulation_type in REFORMULATION_TYPES:
        file_path = SUBQUERIES_DIR / f"{reformulation_type}.txt"
        if file_path.exists():
            console.print(f"[yellow]Processing {reformulation_type} file...[/yellow]")
            token_counts[reformulation_type] = count_tokens_in_file(file_path, tokenizer)

            # Print statistics, including P90, P95, and P99
            counts = token_counts[reformulation_type]
            p90 = np.percentile(counts, 90)
            p95 = np.percentile(counts, 95)
            p99 = np.percentile(counts, 99)
            
            console.print(f"[green]{reformulation_type.capitalize()} statistics:[/green]")
            console.print(f"  Total lines: {len(counts)}")
            console.print(f"  Mean tokens: {np.mean(counts):.2f}")
            console.print(f"  Median tokens: {np.median(counts):.2f}")
            console.print(f"  Max tokens: {np.max(counts)}")
            console.print(f"  Min tokens: {np.min(counts)}")
            console.print(f"  P90 tokens: {p90:.2f}")
            console.print(f"  P95 tokens: {p95:.2f}")
            console.print(f"  P99 tokens: {p99:.2f}")
        else:
            console.print(f"[red]File not found: {file_path}[/red]")

    # Plot the distributions
    console.print("[bold blue]Plotting token distributions...[/bold blue]")

if __name__ == "__main__":
    main()

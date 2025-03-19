import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from bert_score import score


def semantic_similarity_score(input: List[str], output: List[str]) -> Dict[str, float]:
    # BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli" # 3.6s
    BERTSCORE_MODEL = "roberta-large" # 1.0s
    LANG = "en"
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    return score(input, output, model_type=BERTSCORE_MODEL, lang=LANG, device=device)


if __name__ == "__main__":
    # Get an intuitive understanding of the BERTScore metric
    data_path = Path("data/full.jsonl")
    tests = [] # (input, output, similar)
    
    print(f"[INFO] Loading data from {data_path}")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            tests.append((item["query"], item["subqueries"], 1))
    tests = random.sample(tests, 100)

    # Load random subqueries from reformulation type files and calculate BERTScore
    print("[INFO] Loading random subqueries for comparison...")
    random_comparisons = []
    for ref_type in ["comparison", "expansion", "chaining"]:
        subq_path = Path(f"subqueries/{ref_type}.txt")
        subq_list = []
        with open(subq_path, "r") as f:
            subq_list += [line.strip() for line in f if line.strip()]
        for _ in range(100):
            input_subq, output_subq = random.sample(subq_list, 2)
            random_comparisons.append((input_subq, output_subq, 0))  # 0 indicates dissimilar
    
    tests += random.sample(random_comparisons, 100)
    print(f"[INFO] Calculating BERTScore for {len(tests)} tests...")

    output_path = Path("benchmark/bertscore_results.csv")
    
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Subqueries", "Similarity", "Precision", "Recall", "F1"])
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Calculating BERTScores...", total=len(tests))
            
            for (input, output, similarity) in tests:
                P, R, F1 = semantic_similarity_score([input], [output])
                
                writer.writerow([input, output, similarity, P, R, F1])
                progress.update(task, advance=1)

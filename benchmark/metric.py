import csv
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
from transformers import EvalPrediction, T5Tokenizer

from benchmark.score import score_function


def compute_metrics(eval_pred: EvalPrediction, tokenizer: T5Tokenizer) -> Dict[str, float]:
    predictions, labels = eval_pred
    
    # The predictions are likely coming as logits
    # We need to handle this properly
    if isinstance(predictions, tuple):
        # If predictions is a tuple, take the first element (logits)
        predictions = predictions[0]
    
    # Get the most likely token IDs
    predictions = np.argmax(predictions, axis=-1)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Handle labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # --- ROUGE-L ---
    # Convert string outputs to lists since score_function expects lists
    scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # Make sure we're passing lists to score_function
        pred_list = [pred]
        label_list = [label]
        score = score_function(label_list, pred_list)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": avg_score,
    }

if __name__ == "__main__":
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

    output_path = Path("benchmark/rouge_results.csv")
    
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Subqueries", "Similarity", "ROUGE"])
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Calculating BERTScores...", total=len(tests))
            
            for (input, output, similarity) in tests:
                rouge = score_function([input], [output])
                writer.writerow([input, output, similarity, rouge])
                progress.update(task, advance=1)

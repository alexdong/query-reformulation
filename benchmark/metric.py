from typing import Dict, Tuple

import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from bert_score import score
from transformers import T5Tokenizer


def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    tokenizer: T5Tokenizer,
    model_size: str,
    device: torch.device,
) -> Dict[str, float]:
    predictions, labels = eval_pred

    # Process each sequence individually
    decoded_preds = []
    decoded_labels = []

    for pred, label in zip(predictions, labels):
        # Replace -100 with pad_token_id
        pred_processed = np.where(pred != -100, pred, tokenizer.pad_token_id).tolist()
        label_processed = np.where(label != -100, label, tokenizer.pad_token_id).tolist()

        # Decode to text
        decoded_preds.append(tokenizer.decode(pred_processed, skip_special_tokens=True))
        decoded_labels.append(tokenizer.decode(label_processed, skip_special_tokens=True))

    # Calculate BERTScore
    P, R, F1 = score(decoded_preds, decoded_labels, lang="en",
model_type="microsoft/roberta-large", device=device)
    return {"bertscore_f1": F1.mean().item()}


if __name__ == "__main__":
    # Get an intuitive understanding of the BERTScore metric
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    data_path = Path("data/full.jsonl")
    tests = [] # (input, output, similar)
    
    print(f"[INFO] Loading data from {data_path}")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            tests.append((item["query"], item["subqueries"], 1))
    tests = random.sample(tests, 100)

    # Load random subqueries from reformulation type files and calculate BERTScore
    print(f"[INFO] Loading random subqueries for comparison...")
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
    
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    
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
                P, R, F1 = score([input], [output], 
                                model_type="microsoft/deberta-xlarge-mnli", lang="en", device=device)
                
                writer.writerow([input, output, similarity, P, R, F1])
                progress.update(task, advance=1)

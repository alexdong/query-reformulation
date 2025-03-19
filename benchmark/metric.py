from typing import Dict, Tuple

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
    import csv
    import json
    from pathlib import Path
    
    import torch
    from transformers import T5Tokenizer
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # Load data from full.jsonl
    data_path = Path("data/full.jsonl")
    queries = []
    subqueries = []
    
    print(f"[INFO] Loading data from {data_path}")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item.get("query", ""))
            subqueries.append(item.get("subqueries", ""))
    
    print(f"[INFO] Loaded {len(queries)} query-subquery pairs")
    
    # Calculate BERTScore
    print(f"[INFO] Calculating BERTScores...")
    P, R, F1 = score(queries, subqueries, lang="en", 
                     model_type="microsoft/roberta-large", device=device)
    
    # Save results to CSV
    output_path = Path("benchmark/bertscore_results.csv")
    print(f"[INFO] Saving results to {output_path}")
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Subqueries", "Precision", "Recall", "F1"])
        
        for i in range(len(queries)):
            writer.writerow([
                queries[i], 
                subqueries[i], 
                P[i].item(), 
                R[i].item(), 
                F1[i].item()
            ])
    
    # Print summary statistics
    print(f"[INFO] Results summary:")
    print(f"Average Precision: {P.mean().item():.4f}")
    print(f"Average Recall: {R.mean().item():.4f}")
    print(f"Average F1: {F1.mean().item():.4f}")

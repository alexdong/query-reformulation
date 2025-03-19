from typing import Dict, Tuple

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
        writer.writerow(["Query", "Subqueries", "random", "Precision", "Recall", "F1"])
        
        for i in range(len(queries)):
            writer.writerow([
                queries[i], 
                subqueries[i], 
                0,
                P[i].item(), 
                R[i].item(), 
                F1[i].item()
            ])

    # Load random subqueries from reformulation type files and calculate BERTScore
    reformulation_types = ["comparison", "expansion", "chaining"]
    random_comparisons = []
    
    print(f"[INFO] Loading random subqueries for comparison...")
    for ref_type in reformulation_types:
        subq_path = Path(f"subqueries/{ref_type}.txt")
        if not subq_path.exists():
            print(f"[WARNING] File not found: {subq_path}")
            continue
            
        # Load all subqueries from the file
        with open(subq_path, "r") as f:
            subq_list = [line.strip() for line in f if line.strip()]
        
        # Select random pairs for comparison (limited to ~33 per type to get ~100 total)
        max_pairs = min(33, len(subq_list) // 2)
        for _ in range(max_pairs):
            # Select two different random subqueries
            indices = random.sample(range(len(subq_list)), 2)
            random_comparisons.append((subq_list[indices[0]], subq_list[indices[1]]))
    
    # Limit to 100 comparisons
    random_comparisons = random_comparisons[:100]
    
    if random_comparisons:
        print(f"[INFO] Calculating BERTScores for {len(random_comparisons)} random pairs...")
        random_queries = [pair[0] for pair in random_comparisons]
        random_subqueries = [pair[1] for pair in random_comparisons]
        
        # Calculate BERTScore for random pairs
        random_P, random_R, random_F1 = score(random_queries, random_subqueries, 
                                             lang="en", model_type="microsoft/roberta-large", 
                                             device=device)
        
        # Append results to CSV
        with open(output_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(random_comparisons)):
                writer.writerow([
                    random_queries[i],
                    random_subqueries[i],
                    1,  # Set random column to 1
                    random_P[i].item(),
                    random_R[i].item(),
                    random_F1[i].item()
                ])
        
        # Print summary statistics for random comparisons
        print(f"[INFO] Random comparisons summary:")
        print(f"Average Precision: {random_P.mean().item():.4f}")
        print(f"Average Recall: {random_R.mean().item():.4f}")
        print(f"Average F1: {random_F1.mean().item():.4f}")
    
    # Print summary statistics
    print(f"[INFO] Results summary:")
    print(f"Average Precision: {P.mean().item():.4f}")
    print(f"Average Recall: {R.mean().item():.4f}")
    print(f"Average F1: {F1.mean().item():.4f}")

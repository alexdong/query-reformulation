import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import EvalPrediction
import numpy as np

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def score(input, output):
    return scorer.score(input, output)['rougeL'].fmeasure

def compute_metrics(eval_pred: EvalPrediction, tokenizer=None, model_size=None, device=None):
    """
    Computes ROUGE-L for use with Hugging Face Trainer.evaluate().

    Args:
        eval_pred: An EvalPrediction object containing predictions and labels.
                   predictions: [batch_size, sequence_length] of token IDs
                   label_ids:  [batch_size, sequence_length] of token IDs
                   -100 should be used as padding
        tokenizer: The tokenizer to use for decoding
        model_size: Size of the model (not used, but kept for compatibility)
        device: Device to use (not used, but kept for compatibility)
    Returns:
        A dictionary containing the average ROUGE-L score.
    """
    predictions, labels = eval_pred
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # --- ROUGE-L ---
    rouge_l_scores = [score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return {
        "rouge_l": avg_rouge_l,
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

    output_path = Path("benchmark/rouge_results.csv")
    
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    
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
                rouge = score(input, output)
                writer.writerow([input, output, similarity, rouge])
                progress.update(task, advance=1)

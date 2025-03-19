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
    # load data/full.jsonl and use score function to calculate BERTScore and save the results into a csv file, ai!

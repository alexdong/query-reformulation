import torch
import numpy as np
from bert_score import score

def compute_metrics(eval_pred, tokenizer, model_size, device):
    predictions, labels = eval_pred

    # Process each sequence individually
    decoded_preds = []
    decoded_labels = []

    # Convert to numpy if they're tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()


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

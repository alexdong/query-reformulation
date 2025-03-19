import sys
import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from datasets import load_metric
from bert_score import score

from data import QueryReformulationDataset
from utils.init_models import init_models


def fine_tune(model_size="base", dataset="full", training_epochs=1):
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    train_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="train")
    eval_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="eval")

    training_args = TrainingArguments(
            output_dir=f"./models/sft-{model_size}",
            num_train_epochs=training_epochs,
            per_device_train_batch_size=8,
            save_steps=1_000,
            save_total_limit=2,
            eval_strategy="steps",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="/var/logs",
            logging_steps=1_000,
            overwrite_output_dir=True,
            )

    bleu = load_metric("bleu")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = [[t if t != -100 else tokenizer.pad_token_id for t in p] for p in predictions]
        labels = [[t if t != -100 else tokenizer.pad_token_id for t in l] for l in labels]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        preds = [pred.split() for pred in decoded_preds]
        labels_bleu = [[label.split()] for label in decoded_labels]
        bleu_result = bleu.compute(predictions=preds, references=labels_bleu)["bleu"]
        P, R, F1 = score(decoded_preds, decoded_labels, lang="en", model_type="bert-base-uncased", device="mps")
        return {"bleu": bleu_result, "bertscore_f1": F1.mean().item()}

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model) # dynamic padding

    trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            )
    trainer.train()


if __name__ == "__main__":
    # rewrite the following using click to parse arguments, ai!
    fine_tune(model_size="small", dataset="dev", training_epochs=1)
    #fine_tune(model_size="base", dataset="full", training_epochs=5)

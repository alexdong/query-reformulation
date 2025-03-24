import json
import os
from typing import Tuple

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from data import QueryReformulationDataset, create_datasets
from train.params import get_optimised_hyperparameters
from utils.init_models import init_models


def peft(model_size: str) -> Tuple[T5ForConditionalGeneration, Trainer, QueryReformulationDataset]:
    """Fine-tune a model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
    
    Args:
        model_size: Size of the T5 model to use ('small', 'base', 'large', or '3b')
        
    Returns:
        Tuple containing the fine-tuned model, trainer, and test dataset
    """
    device, tokenizer, model = init_models(model_size)
    model = model.to(device)
    print(f"[DEBUG] Model moved to {device}")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,                     # Rank of the update matrices
        lora_alpha=32,           # Alpha parameter for LoRA scaling
        lora_dropout=0.1,        # Dropout probability for LoRA layers
        target_modules=["q", "v"],  # Attention query and value matrices
        bias="none",             # Don't train bias parameters
    )
    
    # Apply LoRA to the model
    print("[INFO] Applying LoRA to the model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print the percentage of trainable parameters
    
    hyper_parameters = get_optimised_hyperparameters()
    print(f"[DEBUG] Hyperparameters: {hyper_parameters}")
    
    training_dataset, eval_dataset, test_dataset = create_datasets(tokenizer, hyper_parameters['sample_size'])
    print(f"[INFO] Dataset sizes: training={len(training_dataset)}, eval={len(eval_dataset)}, test={len(test_dataset)}")
    assert len(training_dataset) > 0, "Training dataset is empty"
    assert len(eval_dataset) > 0, "Evaluation dataset is empty"
    
    output_dir = f"./models/peft-{model_size}"
    print(f"[INFO] Output directory: {output_dir}")
    
    # LoRA typically allows for larger batch sizes and learning rates
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyper_parameters['num_train_epochs'],
        per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'] * 2,  # Double batch size
        per_device_eval_batch_size=hyper_parameters['per_device_eval_batch_size'] * 2,
        gradient_accumulation_steps=hyper_parameters['gradient_accumulation_steps'],
        save_steps=hyper_parameters['save_steps'],
        save_total_limit=hyper_parameters['save_total_limit'],
        eval_strategy=hyper_parameters['eval_strategy'],
        eval_steps=hyper_parameters['eval_steps'],
        save_strategy=hyper_parameters['save_strategy'],
        logging_dir="/var/logs",
        logging_steps=hyper_parameters['logging_steps'],
        overwrite_output_dir=True,
        fp16=hyper_parameters["fp16"],
        gradient_checkpointing=False,
        logging_first_step=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        learning_rate=hyper_parameters['learning_rate'] * 3,  # Higher learning rate for LoRA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train the model
    model.config.use_cache = False
    print("[INFO] Starting training with LoRA")
    trainer.train()

    # Save the final model and tokenizer
    print(f"[INFO] Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save the trainer state
    trainer_state = {
        "best_model_checkpoint": output_dir,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
    }
    
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f)

    return model, trainer, test_dataset

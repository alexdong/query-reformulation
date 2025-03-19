import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
from typing import Tuple

def get_backend_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_models(model_size: str, use_sft_model: bool = False) -> Tuple[str, T5Tokenizer, T5ForConditionalGeneration]:
    device = get_backend_device()
    
    if use_sft_model:
        # Base output directory
        output_dir = f"./models/sft-{model_size}"
        
        # Check if the final model exists in the output directory
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            model_path = output_dir
            print(f"[INFO] Using final model from: {model_path}")
        else:
            # Look for checkpoints
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            
            if checkpoint_dirs:
                # Sort checkpoints by step number
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                
                # Try to find the best checkpoint using trainer_state.json in each checkpoint
                best_checkpoint = None
                best_metric = -float('inf')
                
                for checkpoint_dir in checkpoint_dirs:
                    state_path = os.path.join(output_dir, checkpoint_dir, "trainer_state.json")
                    if os.path.exists(state_path):
                        with open(state_path, "r") as f:
                            state = json.load(f)
                        
                        # Check if this checkpoint has evaluation metrics
                        if "best_metric" in state:
                            metric = state["best_metric"]
                            if metric > best_metric:
                                best_metric = metric
                                best_checkpoint = checkpoint_dir
                
                if best_checkpoint:
                    model_path = os.path.join(output_dir, best_checkpoint)
                    print(f"[INFO] Using best checkpoint with metric {best_metric}: {model_path}")
                else:
                    # If no best checkpoint found, use the latest
                    latest_checkpoint = checkpoint_dirs[-1]
                    model_path = os.path.join(output_dir, latest_checkpoint)
                    print(f"[INFO] No best checkpoint found, using latest: {model_path}")
            else:
                # No checkpoints found, fall back to pre-trained model
                print(f"[WARNING] No checkpoints found in {output_dir}, using pre-trained model")
                model_path = f"google/flan-t5-{model_size}"
        
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        # Load the pre-trained model
        model_name = f"google/flan-t5-{model_size}"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    return device, tokenizer, model

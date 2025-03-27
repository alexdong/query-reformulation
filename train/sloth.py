import json
from pathlib import Path
import random
from rich.console import Console

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch # Must run after unsloth to ensure proper monkey patching

from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/gemma-3-4b-it"
TARGET_MODEL = f"./models/{BASE_MODEL.split('/')[-1]}-lora"

console = Console()

def train():
    print(f"[INIT] Starting training process with base model: {BASE_MODEL}")
    print(f"[INIT] Target model will be saved to: {TARGET_MODEL}")
    
    print(f"[MODEL] Loading pretrained model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
            model_name = BASE_MODEL,
            max_seq_length = 256,
            load_in_4bit = True,
            load_in_8bit = False)
    print(f"[MODEL] Model loaded successfully")
    
    print(f"[MODEL] Setting up PEFT configuration...")
    model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False)
    print(f"[MODEL] PEFT model configured")
    
    print(f"[TOKENIZER] Applying chat template: gemma-3")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    print(f"[DATA] Loading dataset from ./data/full.jsonl")
    dataset = []
    line_count = 0
    with open(Path("./data/full.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append({"text": tokenizer.apply_chat_template({
                "conversations": [
                    {"content": data["query"], "role": "user"},
                    {"content": data["subqueries"], "role": "assistant"}
                    ]})})
            line_count += 1
            if line_count % 1000 == 0:
                print(f"[DATA] Processed {line_count} examples")
    
    print(f"[DATA] Dataset loaded with {len(dataset)} examples")
    print(f"[DATA] Sample tokenized text (example #100):")
    print(f"[DATA] {dataset[100]['text'][:100]}...")

    print(f"[TRAINER] Configuring SFT trainer...")
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                num_train_epochs=3,
                learning_rate=2e-5,
                max_steps=1000,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="none", # Setup WandB
                ))
    print(f"[TRAINER] SFT trainer configured")

    print(f"[TRAINER] Setting up response-only training...")
    # This ignores the loss on the user's inputs and only trains on the model's responses
    trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",  # apply_chat_template moves 'role': 'assistant' to <start_of_turn>model\n'
            )
    print(f"[TRAINER] Response-only training setup complete")

    print(f"[TRAINING] Starting training process...")
    trainer_stats = trainer.train()
    print(f"[TRAINING] Training completed with stats: {trainer_stats}")
    
    print(f"[SAVE] Saving model to {TARGET_MODEL}")
    model.save_pretrained(TARGET_MODEL)
    tokenizer.save_pretrained(TARGET_MODEL)
    print(f"[SAVE] Model and tokenizer saved successfully")
    # model.save_pretrained_gguf(TARGET_MODEL, quantization_type="Q8_0")

def inference(query):
    print(f"[INFERENCE] Loading model from {TARGET_MODEL}")
    model, tokenizer = FastModel.from_pretrained(
            model_name = TARGET_MODEL,
            max_seq_length = 256,
            load_in_4bit = True)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    print(f"[INFERENCE] Model loaded successfully")

    print(f"[INFERENCE] Processing query: {query}")
    messages = [{"role": "user", "content": {"type": "text", "text": "reformulate: " + query}}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"[INFERENCE] Tokenized input: {text[:50]}...")

    print(f"[INFERENCE] Generating response...")
    outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_length=256,
            temperature=1.0, top_k=64, top_p=0.95
            )
    result = tokenizer.decode(outputs[0])
    print(f"[INFERENCE] Generated response: {result[:50]}...")
    return result

if __name__ == "__main__":
    train()
    print(f"[MAIN] Testing inference with sample query")
    result = inference("What is the capital of France?")
    print(f"[MAIN] Final result: {result}")

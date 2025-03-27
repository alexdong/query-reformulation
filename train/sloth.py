# add detailed print() statement through the code to see data flow and progress. ai!
import json
from pathlib import Path
import random

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch # Must run after unsloth to ensure proper monkey patching

from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/gemma-3-4b-it"
TARGET_MODEL = f"./models/{BASE_MODEL.split('/')[-1]}-lora"

def train():
    model, tokenizer = FastModel.from_pretrained(
            model_name = BASE_MODEL,
            max_seq_length = 256,
            load_in_4bit = True,
            load_in_8bit = False)
    model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    dataset = []
    with open(Path("./data/full.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append({"text": tokenizer.apply_chat_template({
                "conversations": [
                    {"content": data["query"], "role": "user"},
                    {"content": data["subqueries"], "role": "assistant"}
                    ]})})
    dataset[100]["text"]

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

    # TODO: it says that this one ignore the loss on the user's inputs. But why?
    trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",  # apply_chat_template moves 'role': 'assistant' to <start_of_turn>model\n'
            )

    trainer_stats = trainer.train()
    model.save_pretrained(TARGET_MODEL)
    tokenizer.save_pretrained(TARGET_MODEL)
    # model.save_pretrained_gguf(TARGET_MODEL, quantization_type="Q8_0")

def inference(query):
    model, tokenizer = FastModel.from_pretrained(
            model_name = TARGET_MODEL,
            max_seq_length = 256,
            load_in_4bit = True)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    messages = [{"role": "user", "content": {{"type": "text", "text": "reformulate: " + query}}}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    outputs = model.generate(
            **tokenzier([text], return_tensors="pt").to("cuda"),
            max_length=256,
            temperature=1.0, top_k=64, top_p=0.95
            )
    return tokenizer.decode(outputs)

if __name__ == "__main__":
    train()
    print(inference("What is the capital of France?"))

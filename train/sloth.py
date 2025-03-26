import json
from pathlib import Path
import random

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats

import torch # Must run after unsloth to ensure proper monkey patching

model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-4b-it",
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
        dataset.append({"text": tokenizer.apply_chat_template(
        {'content': 'query-reformulate:{data["query"]}', 'role': 'user'},
             {'content': '{data["subqueries"]}', 'role': 'assistant'})})
random.shuffle(dataset)

dataset = standardize_data_formats(to_gpt(dataset))

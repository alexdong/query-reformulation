import click
from transformers import T5ForConditionalGeneration

import time
from transformers import T5ForConditionalGeneration
import click

from benchmark.score import score_function
from data import QueryReformulationDataset
from train.sft import sft




def benchmark(model: T5ForConditionalGeneration, test_dataset: QueryReformulationDataset) -> None:
    # Evaluate the model against the test set
    print("[INFO] Evaluating model on test set...")
    model.config.use_cache = True

    # Get the tokenizer from the trainer
    tokenizer = test_dataset.tokenizer
    
    # Collect actual and predicted subqueries
    labeled_subqueries = []
    predicted_subqueries = []
    
    import time
    from benchmark.score import score_function
    
    start_time = time.time()
    
    # Process each item in the test dataset
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        input_text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        target_text = tokenizer.decode(item['labels'], skip_special_tokens=True)
        
        # Generate prediction using the model
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_return_sequences=1,
            do_sample=False
        )
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        labeled_subqueries.append(target_text)
        predicted_subqueries.append(predicted_text)
    
    # Calculate time taken
    total_time = time.time() - start_time
    avg_time_per_query = total_time / len(test_dataset) if len(test_dataset) > 0 else 0
    
    # Calculate scores using score_function
    score = score_function(labeled_subqueries, predicted_subqueries)
    print(f"[INFO] score: {score}")
    print(f"[INFO] time/query: {avg_time_per_query}")

@click.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
def main(model_size: str) -> None:
    """Train a query reformulation model using the specified parameters."""
    model, trainer, dataset = sft(model_size)
    benchmark(model, dataset)


if __name__ == "__main__":
    main()

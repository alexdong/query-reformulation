from pathlib import Path
from typing import Dict, List

import click

from data import load_dataset_from_jsonl
from models.query import QueryReformulator

DEV_DATASET = Path("datasets/dev.jsonl")
BERT_MODEL_SIZES = ["base", "large"]
MODEL_TYPES = ["t5", "bert"]

def load_dataset(file_path: Path) -> List[Dict[str, str]]:
    """Load a dataset from a JSONL file."""
    return load_dataset_from_jsonl(file_path)

def evaluate(model_size: str, dataset: str, batch_size: int = 8, verbose: bool = False):
    """Evaluate a trained query reformulation model using direct inference.
    
    Args:
        model_size: Size of the T5 model ('small', 'base', or 'large')
        dataset: Dataset to use for evaluation ('dev' or 'full')
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed progress
    """
    # Initialize the model
    reformulator = QueryReformulator(model_size=model_size)
    device = reformulator.device
    
    # Load the test dataset
    dataset_path = Path(f"datasets/{dataset}.jsonl")
    test_data = load_dataset(dataset_path)
    
    if verbose:
        print(f"[INFO] Loaded {len(test_data)} examples from {dataset_path}")
    
    # Prepare for evaluation
    all_predictions = []
    all_references = []
    total_time = 0
    
    # Process examples
    for example in tqdm(test_data, desc="Evaluating"):
        query = example.get("query", "")
        reference = example.get("subqueries", "")
        
        # Measure inference time
        start_time = time.time()
        predictions = reformulator.reformulate(query)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        # Store results
        all_predictions.append(predictions[0])  # Take first prediction
        all_references.append(reference)
        
        if verbose and len(all_predictions) % 10 == 0:
            print(f"Example {len(all_predictions)}:")
            print(f"  Query: {query}")
            print(f"  Prediction: {predictions[0]}")
            print(f"  Reference: {reference}")
            print(f"  Inference time: {inference_time:.4f}s")
    
    # Calculate BERTScore
    from bert_score import score
    
    P, R, F1 = score(all_predictions, all_references, lang="en",
                     model_type="microsoft/roberta-large", device=device)
    
    # Prepare results
    results = {
        "bertscore_f1": F1.mean().item(),
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "avg_inference_time": total_time / len(test_data),
        "total_examples": len(test_data),
    }
    
    print(f"[INFO] Evaluation results: {results}")
    
    # Save results to file
    output_dir = Path("./benchmark/results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"{model_size}_{dataset}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to {output_file}")
    
    return results


@click.group()
def main() -> None:
    """Benchmark and evaluate query reformulation models."""
    pass

@main.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
@click.option('--dataset', type=str, default='dev',
              help='Dataset to use for evaluation (dev or full)')
@click.option('--verbose', is_flag=True, help='Print detailed progress')
def eval(model_size: str, dataset: str, verbose: bool) -> None:
    """Evaluate a trained query reformulation model using the specified parameters."""
    print(f"[INFO] Evaluating with model_size={model_size}, dataset={dataset}")
    evaluate(model_size, dataset, verbose=verbose)

if __name__ == "__main__":
    main()

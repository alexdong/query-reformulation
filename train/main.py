import click

from benchmark.regression import benchmark
from models import MODEL_CLASSES, PEFT_CLASSES
from train.sft import sft
from train.peft import peft


@click.command()
@click.option('--model-size', type=click.Choice(MODEL_CLASSES), default='small', help='Size of the T5 model to use')
def main(model_size: str) -> None:
    """Train a query reformulation model using the specified parameters and method."""
    if model_size in PEFT_CLASSES:
        print(f"[INFO] Using PEFT (LoRA) for fine-tuning {model_size} model")
        model, trainer, dataset = peft(model_size)
    else:
        print(f"[INFO] Using standard SFT for fine-tuning {model_size} model")
        model, trainer, dataset = sft(model_size)
    
    benchmark(model, dataset)


if __name__ == "__main__":
    main()

import click

from benchmark.regression import benchmark
from train.sft import sft
from train.peft import peft


@click.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large', '3b']), default='small',
              help='Size of the T5 model to use')
@click.option('--method', type=click.Choice(['sft', 'peft']), default='sft',
              help='Fine-tuning method to use (standard SFT or PEFT with LoRA)')
def main(model_size: str, method: str) -> None:
    """Train a query reformulation model using the specified parameters and method."""
    if method == 'peft':
        print(f"[INFO] Using PEFT (LoRA) for fine-tuning {model_size} model")
        model, trainer, dataset = peft(model_size)
    else:
        print(f"[INFO] Using standard SFT for fine-tuning {model_size} model")
        model, trainer, dataset = sft(model_size)
    
    benchmark(model, dataset)


if __name__ == "__main__":
    main()

import click

from benchmark.regression import benchmark
from train.sft import sft


@click.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large', '3b']), default='small',
              help='Size of the T5 model to use')
def main(model_size: str) -> None:
    """Train a query reformulation model using the specified parameters."""
    model, trainer, dataset = sft(model_size)
    benchmark(model, dataset)


if __name__ == "__main__":
    main()

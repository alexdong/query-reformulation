Files
===============

- [[`train.py`]]: run SFT the model using the `datasets/full.jsonl`.


Plan
======

1. Fine-tune the model on the `datasets/dev.jsonl` dataset over `Flan-5t-small`
on the Mac Mini M2. 
    - Make sure the `client/chat.py` and `client/api.py` are working with the
      fine-tuned model.
    - Establish a benchmark baseline for both accuracy and speed.

2. Fine-tune the model on the `datasets/full.jsonl` dataset over `Flan-5t-base`
on Google Colab.
    - Check if we can get `quantize.py` working on Google Colab.
    - Download the fine-tuned model and get benchmark results.

3. Latency optimisation through either TorchScript or GGML.
    - [[`quantize.py`]]: dynamic quantizes the model to 8-bits and runs it on a
      TorchScript runtime. I've tried to run this on OSX but errors after errors.
    - [[`ggml.py`]]: compiles the model to GGML and optimise it to run on a CPU.


SFT Base Models
===============

Twitter has proven that BERT-base can be scaled to handle 100ms latency on a
CPU. So we can safely assume that a similar parameter count model can be used
for fine-tuning. In terms of choosing a pre-trained model for fine-tuning, here
are the options:

- ModernBERT 22 layers: 149M parameters
- ModernBERT 28 layers: 395M parameters

- Flan-5T-small: 60M parameters
- Flan-5T-base: 220M parameters
- Flan-5T-large: 770M parameters

Flan-5T feels like a better choice because it's a text-to-text transfer
transformer. It's a better fit for the task because it has both encoder and
decoder, unlike BERT that has only an encoder.

Training Performance
=====================

Train on the `data/dev.jsonl` dataset over `Flan-5t-small` took about 5 seconds and `Flan-5t-base` took about 10 seconds. 

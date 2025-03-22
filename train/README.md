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

The `full.jsonl` dataset took 5:13 minutes to train on `Flan-5t-small` for one epoch.

Benchmark Target
=================

`eval_loss`
------------

The loss value measures how well the model is predicting the target tokens:

 • Poor fit: 10+ (your current value is ~20.5)
 • Decent fit: 2-5
 • Good fit: 1-2
 • Excellent fit: <1

Loss values decrease as training progresses. For T5 models on text generation
tasks, reaching a loss below 2.0 typically indicates the model has learned the
patterns well.


`eval_rouge_l`
---------------

ROUGE-L measures the longest common subsequence between predictions and references:

 • Poor fit: <0.1 (your current value is ~0.04)
 • Decent fit: 0.2-0.3
 • Good fit: 0.3-0.5
 • Excellent fit: >0.5

For query reformulation specifically:

 • Values around 0.3-0.4 indicate the model is capturing the essential information 
 • Values above 0.5 suggest the model is generating high-quality 
   reformulations that closely match the expected outputs

Training Log
=============

Rewrote the hyperparameters in [[`train/params.py`]] to make it easier to track the
hyperparameters used for training in different environments. Run `train/main.py`
will print out the parameters used for training.

5t-small SFT Results
-----------------

| Metric | Value |
|--------|-------|
| model_size | Flan-5t-small |
| dataset_size | 10% |
| epoch | 3.0000 |
| train_runtime | 378 |
| train_samples_per_second | 10.165 |
| eval_loss | 0.3093 |
| eval_rouge_l | 0.6447 |
| eval_runtime | 67.9803 |
| eval_samples_per_second | 2.3680 |
| eval_steps_per_second | 0.3090 |

5t-base SFT Results
-----------------

5t-base-8bit PEFT Results
-----------------

5t-large SFT Results
-----------------

5t-large-8bit PEFT Results
-----------------

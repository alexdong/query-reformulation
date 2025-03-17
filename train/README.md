Files
===============

- [[`train/benchmark.py`]]: runs the benchmark against the
  `datasets/dev.jsonl`. It tries to run the models on CPU on ONNX runtime and
  measure the latency. 

- [[`train/quantize.py`]]: dynamic quantizes the model to 8-bits.

- [[`train/compile.py`]]: compiles the model to GGML and runs it on CPU.

- [[`train/chat.py`]]: contains the code to interact with the model in chat
  mode. It takes in a query and returns the reformulated query as well as the
  time taken to generate the reformulation.

- [[`train/api.py`]]: adds an API layer on top of the model and measures the latency.

- [[`train/evaluate.py`]]: evaluates the model on the `datasets/test.jsonl` and
  returns the accuracy and speed.

- [[`train/train.py`]]: trains the model on the `datasets/training.jsonl` and
  tracks the curve on colab.


Performance Log
===============

Baseline runtime performance on CPU
-------------------------------------

```
==================================================
📊 RESULTS FOR FLAN-T5-BASE (PYTORCH) 📊
==================================================
🕒 Average time per query: 176.42 ms
🕒 Median time per query:  110.08 ms
📏 Standard deviation:     287.26 ms
📈 90th percentile (P90):  221.29 ms
📈 95th percentile (P95):  422.04 ms
📈 99th percentile (P99):  1908.62 ms
==================================================
```

Dynamically Quantized with TorchScript
---------------------------------------



SFT Models
===============

Twitter has proven that BERT-base can be scaled to handle 100ms latency on a
CPU. So we can safely assume that a similar parameter count model can be used
for fine-tuning. In terms of choosing a pre-trained model for fine-tuning, here
are the options:

- BERT-base: 100M parameters
- Flan-5T-small: 60M parameters
- Flan-5T-base: 220M parameters
- Flan-5T-large: 770M parameters

Flan-5T feels like a better choice because it's a text-to-text transfer
transformer. It's a better fit for the task because it has both encoder and
decoder, unlike BERT that has only an encoder.

Further, in terms of the environment to run the fine-tuning, it looks like
running it on Google Colab is the best option. Here is the comparison to
running it locally on M2.

Metric	        | M2 Pro (MPS)	| T4 (CUDA)
----------------------------------------------
Batch Size      |	4	        |   16
Steps/Second	|  ~0.8	        |   ~2.1
Epoch Time	    |  ~4.5 hours	|   ~1.75 hours
Memory Pressure	|  High         |   Moderate


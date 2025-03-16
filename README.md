A ML model that can generate query reformulations 

Project Definition
===================

> Build an API to a machine learning model that given an input query can generate one or more adequate search engine queries to obtain requested information.
> 
> Your API should have maximum 100ms latency on a consumer grade CPU.
> 
> Examples of good API outputs for given inputs:
> 
> 1)
> In what year was the winner of the 44th edition of the Miss World competition born?
> 44th Miss World competition winner birth year
> 
> 2)
> Who lived longer, Nikola Tesla or Milutin Milankovic?
> Nikola Tesla lifespan
> Milutin Milankovic lifespan
> 
> 3)
> Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?
> David Chanoff U.S. Navy admiral collaboration
> U.S. Navy admiral ambassador to United Kingdom
> U.S. President during U.S. Navy admiral's ambassadorship
> 
> 4)
> Create a table for top noise cancelling headphones that are not expensive
> top noise cancelling headphones under $100
> top noise cancelling headphones $100 - $200
> best budget noise cancelling headphones
> noise cancelling headphones reviews
> 
> 5)    
> what are some ways to do fast query reformulation
> fast query reformulation techniques
> query reformulation algorithms
> query expansion methods
> query rewriting approaches
> query refinement strategies


Observation
--------------

Looking at the examples provided, it shows the following cases
- conciseness rewrite (1)
- decomposition (2)
- logical inference (3)
- query expansion (4, 5)

Although the assignment didn't specify the nature or shape of the "search
engine queries", intuitively, each concrete query feels like something we can
feed into a Google Knowledge Base Graph or wikidata to get a concrete answer.

This is not only evident for those questions with a clear answer (1, 2, 3), but
also for maximum coverage of the search space (4). 5 is one where it's clearly
an expansion on the concept of "query reformulation" by providing a list of
possible "relevant" queries.

Two other significant observations are:

1. This requires a 100ms inference time on a CPU. This reminds me of the
   [Twitter](https://blog.x.com/engineering/en_us/topics/insights/2021/speeding-up-transformer-cpu-inference-in-google-cloud)
   team scaling up a BERT model to handle 100ms latency on a CPU through an
   ONNX converted runtime with a dynamic quantization torchscript. Since the
   problem on hand is a rewrite task, I think a Flan-5T-base (text-to-text
   transfer transformer) would be a good starting point.

2. We need a high quality dataset to train and benchmark the model. There are 
    a lot of good Q&A datasets out there. We need to extract the questions and
    use a reasoning model to produce the reformulated output. It'll cost under
    $5 for every 10k reformulations pairs.


Plan
--------


Step 1: Prepare synthetic data

1. Take MS-MARCO and HotpotQA datasets and extract the questions from them.
2. Consolidate notes into a Prompt and iterate over random sample of questions
   to generate outputs. (Start with 50 pairs)
3. Once we are happy with the results from chat.deepseek.com, evaluate the
   locally hosted R1-7b model and see if it can be used to generate answers.
4. Slowly build up to 10 reformulation pairs per type using API.
5. Generate 10k reformulation pairs using API.

Step 2: Train a model

1. Take a standard Flan-5T-base model and quantize it to 8-bits and run on
   different inference runtime.
2. Get a benchmark suite working, establish a baseline for both accuracy and
   speed.
3. Train a model on the 100, 1k and 10k reformulation pairs. Track the curve on
   colab. 
(Rinse and repeat)

Step 3: Performance tuning

1. Quantize the model to 8-bits 
2. run on ONNX runtime
3. Run on a CPU and measure the latency
4. Compile to GGML and run it on CPU
5. add an API layer on top of the model and measure the latency


Prepare Synthetic Data
==========================

15/March
------------

A few considerations to keep in mind when preparing the synthetic data:
1. Chat Prefix Completion is a good way to get output; 
2. Context Caching is critical to lower the cost of the reasoning model;

The approach to use the reasoning model to generate "output" from questions
from MS-MARCO and HotpotQA datasets is not feasible. The cost of the reasoning
model is too high, latency is also incredibly high. Not happy with the result.
Maybe I need a different approach to generate the synthetic data.

16/March
------------

Waking up thinking that I can go the other way. I can randomly sample entities
from wikidata to produce the `subqueries`, then use a LLM to rewrite them into
a "query". The `subqueries` can be generated deterministically from the
wikidata API. The latest LLM seems to be able to handle the "summarise" a lot
better. 

The deterministic generation of `subqueries` will allow me to generate a lot of
data for free. The idea is captured in [[README-synthetic-data-chain.md]].

Given the subqueries, I can use the LLM to generate the final query. The
prompts are:

- Chaining: [[PROMPT-question_generation-chaining.md]]
- Comparison: [[PROMPT-question_generation-comparison.md]]
- Expansion: [[PROMPT-question_generation-expansion.md]]

One more benefit of this approach is that when I ask for the input, I can
generate 25 different queries. This will give us the training data for
"conciseness" and "reparaphrasing" for free.

Choose the LLM - o3-mini
----------------------------

Evaluated the above prompts against main LLMs of different parameter counts. Here are the results:

- DeepSeek V3: poor incorect result: "Which U.S. Navy admiral collaborated with David Chanoff and later served as an ambassador to the United Kingdom, and who was the U.S. President during that admiral's ambassadorship?"
- Grok 3: poor incorrect result: "Which U.S. Navy admiral collaborated with David Chanoff, served as ambassador to the United Kingdom, and who was the U.S. President during that time?"
- Gemma 3 27B: poor incorrect result: "When David Chanoff was a U.S. Navy admiral and ambassador to the United Kingdom, who was the President of the United States?"
- Gemini 2 Flash: incorrect: "What US president held office during David Chanoff's tenure as ambassador to the United Kingdom after his naval service?"
- Gemini 2 Flash-Lite: utterly confused. "SELECT ?president WHERE { ?admiral wdt:P31 wd:Q39678. ..."
- OpenAI-4o: poor incorrect result: "Who collaborated with David Chanoff and was a U.S. Navy admiral?"
- Gemini 2 Pro: excellent result
- OpenAI-3o-mini: excellent result
- Claude 3.7: Excellent result

In terms of cost, rate limit and discounts, here is the comparison:

LLM Model    | Input Tokens | Output Tokens | Rate Limit                 | Discounts
-------------| ------------ | ------------- | -------------------------- | ---------
Sonnet 3.7   | $3 + 0.30    |  $15          |  1.5 tokens per second     |  50% off batch processing
3o-mini      | $1.1 + 0.55  |  $4.4         |  1.5 tokens per second     |  50% off batch processing
Gemini 2 Pro | N/A          |  N/A          |  50 requests per day       | 

Looks like o3-mini is the best option for generating synthetic data.
Batch API: https://platform.openai.com/docs/guides/batch


Fine-tuning
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


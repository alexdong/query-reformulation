A ML model that can generate query reformulations 

## Project Definition

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


## Observation

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

STEP 1: Prepare synthetic data
--------------------------------
1. Take MS-MARCO and HotpotQA datasets and extract the questions from them.
    - MS-MARCO is a parquet file with `query_type` and `query` columns. 
    - HotpotQA is a json file with [{'question'}] field
2. Consolidate notes into a Prompt and iterate over random sample of questions
   to generate outputs. (Start with 50 pairs)
3. Once we are happy with the results from chat.deepseek.com, evaluate the
   locally hosted R1-7b model and see if it can be used to generate answers.
4. Slowly build up to 10 reformulation pairs per type using API.
5. Generate 10k reformulation pairs using API.

STEP 2: Train a model
----------------------
1. Take a standard Flan-5T-base model and quantize it to 8-bits and run on
   different inference runtime.
2. Get a benchmark suite working, establish a baseline for both accuracy and
   speed.
3. Train a model on the 100, 1k and 10k reformulation pairs. Track the curve on
   colab. 
(Rinse and repeat)

Datasets, Models and Tools
-------------------

- [MS-MARCO](https://huggingface.co/datasets/microsoft/ms_marco): `jq '.[].question' hotpot_train_v1.1.json > questions.txt`
- [HotpotQA](https://hotpotqa.github.io/): `pqrs cat test-00000-of-00001.parquet --json | jq -r ".query" >> questions.txt`

- BERT-base: 100M parameters
- Flan-5T-small: 60M parameters
- Flan-5T-base: 220M parameters
- Flan-5T-large: 770M parameters

- [Weights & Bias](https://wandb.ai/site/evaluations/)
- Leverage [DeepEval](https://docs.confident-ai.com/) to save time on the benchmarking suite
- [DeepSeek r1 pricing](https://api-docs.deepseek.com/quick_start/pricing)

- colab compared to local runtime

    Metric	        | M2 Pro (MPS)	| T4 (CUDA)
    -------------------------------------
    Batch Size      |	4	        |   16
    Steps/Second	|  ~0.8	        |   ~2.1
    Epoch Time	    |  ~4.5 hours	|   ~1.75 hours
    Memory Pressure	|  High         |   Moderate


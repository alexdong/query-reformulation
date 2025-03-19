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

1. Compile to GGML and run it on CPU
2. add an API layer on top of the model and measure the latency


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
data for free. The idea is captured in [[subquries/README.md]].

One more benefit of this approach is that when I ask for the input, I can
generate 25 different queries. This will give us the training data for
"conciseness" and "reparaphrasing" for free.



Query Generation
----------------

Neither wikidata, nor Google Knowledge Graph API can be used to generate the knowledge graph. 
There is a SPARQL endpoint at https://dbpedia.org/sparql that takes in SPARQL queries
and returns the results. But its output is surprisingly limited. In fact, pretty much the same
as the wikidata API. For example, here are the only properties for the entity "Nikola Tesla":

```sparql
http://dbpedia.org/ontology/deathDate: ['1943-01-07']
http://dbpedia.org/ontology/birthDate: ['1856-07-10']
http://dbpedia.org/ontology/wikiPageID: ['21473']
http://dbpedia.org/ontology/wikiPageRevisionID: ['1123520515']
http://dbpedia.org/property/birthDate: ['1856-07-10']
http://dbpedia.org/property/date: ['2016-02-02']
http://dbpedia.org/property/deathDate: ['1943-01-07']
http://dbpedia.org/property/video: ['--10-26']
http://dbpedia.org/ontology/wikiPageLength: ['152127']
http://dbpedia.org/ontology/birthYear: ['1856']
http://dbpedia.org/ontology/deathYear: ['1943']
```

Unless we can find a better way to generate the knowledge graph, we'll have to
rethink the approach. 

10:13pm. Hey! we should be able to extract the information from LLM using a
prompt like [[`facts/_PROMPT.md`]]. Even if the facts are hallucinated,
we can use them to generate the subqueries. All we want is the subqueries to be
deterministic. We actually don't care about the truthfulness of the facts for 
this particular project.


Generate Subqueries from Facts Graph
-------------------------------------

17/March: [[`facts/_main.py`]] has produced 1000+ entities with their properties overnight.

Update the algorithm and prompts in [[`subqueries/_README.md`]]. The shape of the data isn't
quite what we need to generate chaining queries. So I need to Need to clean up
the `facts/*.json` to make sure they each contain a `type` property. Then we
can go on to generate the subqueries.

Also took the time to clean up the project structure as follows:

- `facts/`: contains the facts graph, code to generate the graph and individual
  entity.json files.
- `subqueries/`: contains the code to generate the subqueries from the facts
  graph. Each category type has its own file. e.g. `subqueries/chaining.txt`,
  `subqueries/comparison.txt`, `subqueries/expansion.txt`.
- `queries/`: contains the code to generate the queries from the subqueries.
  The txt files' content follows `query===>subqueries` format. One per line.
  Since we have 25 quries per subquery, we have 25 lines per subquery.
  These 3 files are the training/test data for the SFT step.
- `docs/`: contains the READMEs and other documentation files.


Generate Queries from Subqueries
----------------------------------

[[`queries/_README.md`]] has the plan to generate the queries from the subqueries.
[[`datasets/braid.py`]] is the code to generate the dataset from the subqueries and queries batch input/outputs. 

Training dataset stats
-----------------------

```bash
Query Statistics:
  Total lines: 16040
  Mean tokens: 21.27
  Median tokens: 21.00
  Max tokens: 57
  Min tokens: 6
  P90 tokens: 29.00
  P95 tokens: 32.00
  P99 tokens: 36.00

Comparison Statistics:
  Total lines: 1217
  Mean tokens: 18.79
  Median tokens: 18.00
  Max tokens: 37
  Min tokens: 9
  P90 tokens: 27.00
  P95 tokens: 29.00
  P99 tokens: 31.84

Expansion Statistics:
  Total lines: 658
  Mean tokens: 22.67
  Median tokens: 20.00
  Max tokens: 338
  Min tokens: 9
  P90 tokens: 33.00
  P95 tokens: 38.00
  P99 tokens: 70.30

Chaining Statistics:
  Total lines: 1333
  Mean tokens: 30.53
  Median tokens: 30.00
  Max tokens: 76
  Min tokens: 11
  P90 tokens: 40.00
  P95 tokens: 45.00
  P99 tokens: 54.00
```


Alternative Approaches
----------------------

1. Use RL to evaluate the quality of the reformulation. Similar to
   [PraveenSH/RL-Query-Reformulation](https://github.com/PraveenSH/RL-Query-Reformulation).
   I can't find a good enough reward model other than ROUGE-L. 

2. SFT a LLaDA - Diffusion LLM model, like what [Mercury
   Code](https://www.inceptionlabs.ai/news) is doing. It should deliver a
   significant improvement in latency time, which in turns allows us to use a
   larger model. 

3. Generate a GGUF and run it over llama.cpp.

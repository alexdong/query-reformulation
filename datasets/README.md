This module contains the code to generate the dataset and split it into subsets
so we can use to train our Query Reformulation model.

Generate `full.jsonl`
--------------------

`braid.py` that takes the `batch-input-{reformulation_type}.jsonl` and
`batch-output-{reformulation_type}.jsonl` and produce a new file `full.jsonl`
that braids the two together. 

For each line in the `batch-input-{reformulation_type}.jsonl`, we have a single line
in the `batch-output-{reformulation_type}.jsonl`, which contains 5 queries. 
We need to add 5 new lines into the `full.jsonl` where each line contains the
the unique query and the common subqueries.

Here is a line from the `queries/batch-output-chaining.jsonl` file

```json
{"id": "batch_req_67d7fe6bdf3c8190864bb92eb9c35ce5", "custom_id": "chaining_1", "response": {"status_code": 200, "request_id": "3493b5ed7e8feb86fd5b4145f7cf51f2", "body": {"id": "chatcmpl-BC2CVYHULTogaMD2v9CA9FZJj0w5J", "object": "chat.completion", "created": 1742208095, "model": "o3-mini-2025-01-31", "choices": [{"index": 0, "message": {"role": "assistant", "content": "What is the research cost for the research used by a business function required for a transaction linked to market demand?\nHow much does the research cost that is employed by a business function necessary for transactions associated with market demand?\nWhat is the research cost of the research that a business function, required for a transaction connected to market demand, utilizes?\nWhat is the expense of the research used in the business function needed for transactions linked to market demand?\nHow much does the research cost when it is employed by a business function required for transactions related to market demand?", "refusal": null, "annotations": []}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 441, "completion_tokens": 568, "total_tokens": 1009, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 448, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}, "service_tier": "default", "system_fingerprint": "fp_42bfad963b"}}, "error": null}
```

Here is a line from the `queries/batch-input-chaining.jsonl` file

```json
{"custom_id": "chaining_1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "o3-mini", "messages": [{"role": "system", "content": "You are a NLP specialist with search query."}, {"role": "user", "content": "Your job is to reverse engineer one or a list of subqueries into a single query.  \nRemember that all these subqueries are trying to answer a single query from the user.\n\nEach subquery is a human-readable query to the WikiData that conforms to the topography/terminology.\nUse your knowledge to identify the core entities, properties, qualifiers and their relationship.\nIf there are more than one subqueries, guess the intent of the subqueries. \n\nRecommended Process\n-------------------\n\n\n1. Identify the core entities and the shared property in these subqueries.  \n2. The subqueries follow a sequential chain-of-logic triples, follow through the chain to identify the question the user is trying to ask.\n3. Guess the intent of the comparison over the property. Remember that all these subqueries are trying to answer a single query from the user.\n4. Use relative pronouns and subordinating conjunctions to consolidate/collapse the subqueries into a single question.\n\n\nExample\n-------\n\n<input> \nDavid Chanoff U.S. Navy admiral collaboration\nU.S. Navy admiral ambassador to United Kingdom\nU.S. President during U.S. Navy admiral's ambassadorship\n</input>\n<output> \nWhich U.S. President\u2019s administration coincided with the ambassadorship of the U.S. Navy admiral, who collaborated with David Chanoff, in the United Kingdom?\n</output>\n\nRules for the Final Query\n-------------------------\n\nFor the final query you will produce, follow these rules:\n\n1. Keep your output queries consistent with the style of questions from MS-MARCO, Natural Questions and hotpotQA dataset.\n2. Give me 5 options. One per each line. Make them as distinct as possible from each other.\n3. Only return the queries. One per line. DO NOT include numbers, bullets, or any other text.\n4. DO NOT introduce any new entities, qualifiers that didn\u2019t exist in the subqueries.\n\nOk. Here is the task:\n\nMarket Demand Linked To Transaction\\nTransaction Requires Business function\\nBusiness function Uses Research\\nResearch Cost"}], "max_completion_tokens": 4068}}
```

The braided lines in the `dataset/full.jsonl` should look like the following:

```jsonl
{"query": "What is the research cost for the research used by a business function required for a transaction linked to market demand?", "subqueries": "Market Demand Linked To Transaction\nTransaction Requires Business function\nBusiness function Uses Research\nResearch Cost"}
{"query": "How much does the research cost that is employed by a business function necessary for transactions associated with market demand?", "subqueries": "Market Demand Linked To Transaction\nTransaction Requires Business function\nBusiness function Uses Research\nResearch Cost"}
{"query": "What is the research cost of the research that a business function, required for a transaction connected to market demand, utilizes?", "subqueries": "Market Demand Linked To Transaction\nTransaction Requires Business function\nBusiness function Uses Research\nResearch Cost"}
{"query": "What is the expense of the research used in the business function needed for transactions linked to market demand?", "subqueries": "Market Demand Linked To Transaction\nTransaction Requires Business function\nBusiness function Uses Research\nResearch Cost"}
{"query": "How much does the research cost when it is employed by a business function required for transactions related to market demand?", "subqueries": "Market Demand Linked To Transaction\nTransaction Requires Business function\nBusiness function Uses Research\nResearch Cost"}
```

Split the dataset
-----------------

Once the dataset is generated, we use the `split.py` to create the dev, training, validation and test datasets.

Here is the ratio we are going to use:

- datasets/training.jsonl: 80%
- datasets/validation.jsonl: 10%
- datasets/test.jsonl: 10%
- dataset/dev.jsonl: 200 (small subset from the training.jsonl)

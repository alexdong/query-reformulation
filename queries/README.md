Use LLM to rewrite the `subqueries/*.txt` into a single query.



Plan
-----

Our goal is to generate 3 `queries/{reformulation_type}.txt` where each line is
a `query===>subqueries` that can be used to train the Query Reformulation
model.

We are going to send in a batch of all queries to o3-mini using the corresponding
prompt. For each subqueries, receive 25 queries back. Then we need to save
"query===>subqueries" into the `queries/{reformulation_type}.txt` file.

It worth noting that the query will be different but the subqueries will be the
same for 25 queries.

Batch API: https://platform.openai.com/docs/guides/batch

The prompts uses jinja2 templates. 

- Base: [[`queries/_PROMPT-base.md`]]
- Chaining: [[`queries/_PROMPT-chaining.md`]]
- Comparison: [[`queries/_PROMPT-comparison.md`]]
- Expansion: [[`queries/_PROMPT-expansion.md`]]

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

You are tasked with reformulate a search query to obtain requested information from a knowledge graph, e.g. Google Knowledge Graph API or Wikidata.

Here are the guidelines for the reformulation. For each rule, I've also provided an example of how the rule can be applied to a query where the input is the original query and the output is the reformulated query.

1. Keep each query as concise as possible. 

    input: In what year was the winner of the 44th edition of the Miss World competition born? 
    output: 44th Miss World competition winner birth year

2. If there is a chance to drop certain part of a question without affecting the answer, do so. 

    input: The smallest blood vessels in your body,where gas exchange occurs are called
    output: The smallest blood vessels in human body

   A more complex example where the answer is not affected by the dropped names, titles, or other non-essential parts.

    input: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
    output: Milhouse The Simpsons character namesake

3. If the query requires multiple factual sub-queries, break the query into sub-queries, but do not attempt to precompute them. 

    input: Who lived longer, Nikola Tesla or Milutin Milankovic?
    output: Nikola Tesla lifespan\nMilutin Milankovic lifespan

4. If answering the query requires a chain of sub-queries, list the subqueries in the order that they need to be executed.

    input: Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?
    output: David Chanoff U.S. Navy admiral collaboration\nU.S. Navy admiral ambassador to United Kingdom\nU.S. President during U.S. Navy admiral's ambassadorship

5. If the query is ambiguous, try to use "query expansion" to create subqueries that can be used to retrieve factual results from the knowledge graph with maximise coverage of the answer. Keep these subqueries as distinctive from each other as possible. Do not attempt to merge them. 

    input: top noise cancelling headphones that are not expensive
    output: top noise cancelling headphones under $100\ntop noise cancelling headphones $100 - $200\nbest budget noise cancelling headphones\nnoise cancelling headphones reviews

6. If the query is about a concept, try to use "query expansion" to create subqueries for either alternative names or related concepts. 

    input: what are some ways to do fast query reformulation
    output: fast query reformulation techniques\nquery reformulation algorithms\nquery expansion methods\nquery rewriting approaches\nquery refinement strategies

7. Before we return the results from subqueries, we will compute, resolve the logic chain, merge return values as well as add presentations. Please drop any format/presentation requests. 

    input: Create a table for OECD population in 2024 -> OECD population 2024
    output: OECD population 2024

Please provide the reformulated query for the following input query: 

    input: {{ input_query }}
    output: 

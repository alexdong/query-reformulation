I am working on producing a set of synthetic data of queries through DBpedia
API.

The following describes the process from the simpliest query and gradually
building up to a set of subqueries.

1. Generation Algorithm
====================

1.1. Foundation - Triple Query
-------------------------
Here is the rough outline of how I might approach this

1. Randomly choose an entity from DBpedia
2. Query the entity for its properties and qualifiers 
3. Pick a random property and qualifier from the entity
4. Use the properties and qualifiers to generate a query

For example:

1. Randomly choose Nikola Tesla
2. Query Nikola Tesla for its properties and qualifiers and pick the following:
    - lifespan (dbp:birthDate, dbp:deathDate)
    - birth place (dbp:birthPlace)

Queries:
- Nikola Tesla lifespan
- Nikola Tesla birth place


1.2. Chain of questions
------------------

Here we try to navigate through the graph to create a chain of queries. Building up from the previous workflow.

4. Randomly choose a relationship `dbr`: from the previous entity (e.g. dbr:Smiljan)
5. Use the entity to repeat step 2 and 3 as described in section 1.1.
6. Form a chain of 2-5 queries.

For example

3. Nikla Tesla birth place
4. Smiljan -> Notable people born in Smiljan
5. Ferdinand Kovačević -> Invention
6. Telegraphy

Query chain:

- Nikola Tesla birth place
  Notable people born in birth place
  Inventor of telegraphy


1.3. Comparison 
--------------------------

Based on the "Simplest Query" section, we can also introduce a comparison of
the results from shared properties and qualifiers.

For example, the following subqeries can be generated for the question: "Who
lives longer? Nikola Tesla or Ferdinand Kovačević"

- Nikola Tesla lifespan
- Ferdinand Kovačević lifespan


1.4. Query Composition
-----------------

Further, the comparison can be extended to a list of items.

Use a randomly selected property to produce a set of queries that cover all
available properties in a specific subdomain.

For example, the following subqueries can be generated for the question: "of
all the inventions by Tesla, which one came the earliest?"

- Year of Tesla invented Induction motor
- Year of Tesla invented Tesla Coil
- Year of Tesla invented AC system design
- Year of Tesla invented Wireless Communication
- Year of Tesla invented Remote Control drone
- Year of Tesla invented Neon and Flourescent lighting
- Year of Tesla invented Tesla Turbine
- Year of Tesla invented Tesla Valve
- Year of Tesla invented Tesla Osciillator

2. Key Rules for Queries
=========================

Make sure the following rules are carefully followed:

1. question is as concise as possible.
2. align with WikiData ontology
3. avoid ambiguity and comparatives
4. prioritize specificity
5. join subqueries into a single line, separated by \n

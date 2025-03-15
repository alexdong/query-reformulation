I am working on producing a set of synthetic data by query WikiData through its API endpoint.

Here is the rough outline of how I might approach this

1. Randomly choose an entity from WikiData
2. Query the entity for its properties and qualifiers
3. Use the properties and qualifiers to generate a question

For example:

1. Randomly choose Nikola Tesla
2. Query Nikola Tesla for its properties and qualifiers and pick the following:
    - birth date
    - death date
3. Nikola Tesla lifespan

Make sure the following rules are carefully followed:

1. question is as concise as possible.
2. align with WikiData ontology
3. avoid ambiguity and comparatives
4. prioritize specificity



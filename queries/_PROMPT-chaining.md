{% extends "_PROMPT-base.md" %}

{% block guidelines %}
1. Identify the core entities and the shared property in these subqueries.  
2. The subqueries follow a sequential chain-of-logic triples, follow through the chain to identify the question the user is trying to ask.
3. Guess the intent of the comparison over the property. Remember that all these subqueries are trying to answer a single query from the user.
4. Use relative pronouns and subordinating conjunctions to consolidate/collapse the subqueries into a single question.
{% endblock %}

{% block human_query %}
Which U.S. President’s administration coincided with the ambassadorship of the U.S. Navy admiral, who collaborated with David Chanoff, in the United Kingdom?
{% endblock %}

{% block reformulated_queries %}
David Chanoff U.S. Navy admiral collaboration
U.S. Navy admiral ambassador to United Kingdom
U.S. President during U.S. Navy admiral's ambassadorship
{% endblock %}


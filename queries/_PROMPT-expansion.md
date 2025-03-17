{% extends "_PROMPT-base.md" %}

{% block guidelines %}
1. Identify the entities, properties, and qualifiers in each subquery.
2. Determine the core entity shared by all subqueries.
3. Infer the original user question based on the intent behind the subqueries.
4. Generalize non-core entities into a broader concept while ensuring they are
   not explicitly included in the final query.
{% endblock %}

{% block reformulated_queries %}
Nicola Tesla Turbine
Nicola Tesla AC system
Nicola Tesla Turbine
Nicola Tesla Neon
{% endblock %}

{% block human_query %}
what did Nicola Tesla invent?
{% endblock %}

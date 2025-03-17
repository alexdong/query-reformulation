{% extends "_PROMPT-base.md" %}

{% block guidelines %}
1. Identify the core entities and the shared property in these subqueries.  
2. Guess the intent of the comparison over the property. 
3. Use comparatives and superlatives to consolidate/collapse the subqueries into a single question.
{% endblock %}

{% block human_query %}
which invention by Tesla comes earlier? Induction motor? Or the Tesla Coil?
{% endblock %}

{% block reformulated_queries %}
Year of Tesla invented the Induction motor
Year of Tesla invented the Tesla Coil
{% endblock %}


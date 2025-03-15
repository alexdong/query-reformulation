Task: Wikidata Query Reformulation

Objective:
Reformulate user's natural language questions into 2–4 concise, human-readable components optimized for efficient Wikidata lookups.  The goal is to transform complex or ambiguous questions into structured search terms that directly map to Wikidata entities and properties.

Rules for Reformulation:

1. Extract Core Entities:
   - Identify the central subjects of the question (people, places, organizations, events, concepts).
   - Example: "Nikola Tesla lifespan" → Core entities: "Nikola Tesla", "lifespan"

2. Simplify Relationships (Property Mapping):
   - Replace verbs and phrasal verbs with noun-based Wikidata properties.
   - Aim for properties that directly exist or are commonly used in Wikidata.
   - Original: "Who lived longer?" → Reformulated: "lifespan" (maps to birth/death dates - P569/P570)
   - Original: "collaborated with" → Reformulated: "collaboration" (maps to P1082 -  "collaborated with")

3. Split Multi-Step Questions (Decomposition):
   - Break down nested or complex queries into a sequence of independent, simpler components.
   - This makes each component directly searchable and easier to process.
   - Example:
     - Input: "Capital of the country where the inventor of the lightbulb was born?"
     - Output Components:
       - "inventor of the lightbulb"
       - "birth country of inventor"
       - "capital of birth country"

4. Avoid Ambiguity and Comparatives (Specificity & Neutralization):
   - Replace vague or subjective terms with concrete, Wikidata-aligned attributes.
   - Neutralize comparative language to focus on specific properties.
   - "not expensive" → "price under $100" or "price range $100-$200"
   - "lived longer" → "lifespan" (focus on the property, not the comparison)

5. Align with Wikidata Terminology (Ontology Mapping):
   - Use terms that are consistent with Wikidata's vocabulary and property naming conventions.
   - Leverage common Wikidata properties (e.g., lifespan (P569/P570), collaboration (P1082), ambassador to (P39 + Q159648)).
   - Refer to Wikidata property documentation when unsure.

6. Prioritize Specificity (Granularity):
   - Include explicit details like price ranges ("$100-$200"), date ranges ("2020-2023"), or qualifiers ("customer reviews").
   - More specific components lead to more precise Wikidata queries.

7. Handle Open-Ended Queries (Subdomain Decomposition):
   - For broad or exploratory questions, split them into narrower, technical subdomains or categories.
   - "ways to improve query reformulation" → "query reformulation algorithms", "query reformulation methods", "query reformulation evaluation metrics"

Additional Notes:

* Edge Cases:
    - Time-sensitive data (e.g., "current population"): Add implicit time qualifiers like "latest", "current", or a specific year (e.g., "population 2023").
    - Lack of direct Wikidata terms: Use broader, related categories or properties as approximations (e.g., "product rating" as a proxy for "best").

* Efficiency:
    - Aim for 2–4 components to maintain conciseness and query efficiency. Avoid over-splitting unless necessary for clarity.
    - Utilize general qualifiers like "budget" for affordability or "reviews" for credibility.

Prompt Usage:

Input: [User's Natural Language Question]
Output: [List of Concise, Wikidata-Optimized Components]


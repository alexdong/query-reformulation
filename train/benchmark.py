"""
Read the lines from `datasets/dev.jsonl` where a line might look like this:

{"query": "How does Diana, Princess of Wales link to both Prince William and Prince Harry?", "subqueries": "Diana  Princess of Wales Prince William\nDiana  Princess of Wales Prince Harry”}

I want the benchmark tool to:

- Go through each line. 
- Extract the query and
- Send into the `flan-5T-{model_size}` an instruction “reformulate:{{query}}” 

Measure the time it takes to go through all lines.
"""

# implement this function for me, ai!

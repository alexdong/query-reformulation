runs the benchmark against the `datasets/dev.jsonl`. 

BERTScore vs. ROUGE
---------

I've piloted the BERTScore in the `__main__` section of the `metric.py` file. I
used 100 (input, output) and 100 (random_input, input) pairs to test the score.
The result is not any good. They were very slow to execute (>1s) and the
`roberta-large` was only slightly better than random.
The evaluation score is in `BERTScore evaluation.xlsx` file.


ROUGE turns out to be a better choice. It's almost instantaneous and the results
shows a stronger correlation with "similarity" than BERTScore. 

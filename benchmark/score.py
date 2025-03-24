from typing import List

import evaluate
from Levenshtein import distance as levenshtein_distance

# simplify the code so we only use levenshtein score. update dependent code, ai!
import nltk
from bert_score import score as bert_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



def preprocess_query_string(query_string: str) -> set:
    tokens = word_tokenize(query_string.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return set(words)

def jaccard_similarity(set1: set, set2: set) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def levenshtein_similarity(s1: str, s2: str) -> float:
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2)) or 1
    normalized_distance = distance / max_len
    return 1.0 - normalized_distance

def bert_similarity(reference_string: str, prediction_string: str) -> float:
    _, _, f1 = bert_score([prediction_string], [reference_string], model_type="distilbert-base-uncased", lang="en")
    return f1.item()

# Load the ROUGE metric from Hugging Face evaluate
rouge = evaluate.load("rouge")

def rouge_l_similarity(reference_string: str, prediction_string: str) -> float:
    """Calculates ROUGE-L F1-score between two strings using Hugging Face evaluate."""
    scores = rouge.compute(predictions=[prediction_string], references=[reference_string], rouge_types=["rougeL"])
    return scores["rougeL"] # Returns ROUGE-L F1-score by default


def score_function(
        labeled_subqueries: List[str],
        predicted_subqueries: List[str],
        weight_rouge_l: float = 0.2,
        weight_levenshtein: float = 0.4,
        weight_jaccard: float = 0.4,
        ) -> float:
    num_labeled = len(labeled_subqueries)
    if num_labeled != len(predicted_subqueries):
        return 0.0

    """
    bert_similarities = []
    for i in range(num_labeled):
        labeled_sub = labeled_subqueries[i].strip()
        predicted_sub = predicted_subqueries[i].strip()

        bert_sim = bert_similarity(labeled_sub, predicted_sub)
        bert_similarities.append(bert_sim)
    return sum(bert_similarities) / num_labeled
    """
    
    rouge_l_similarities = []
    levenshtein_similarities = []
    jaccard_similarities = []
    for i in range(num_labeled):
        labeled_sub = labeled_subqueries[i].strip()
        predicted_sub = predicted_subqueries[i].strip()

        rouge_l_sim = rouge_l_similarity(labeled_sub, predicted_sub)
        rouge_l_similarities.append(rouge_l_sim)

        lev_sim = levenshtein_similarity(labeled_sub, predicted_sub)
        levenshtein_similarities.append(lev_sim)

        jac_sim = jaccard_similarity(preprocess_query_string(labeled_sub), preprocess_query_string(predicted_sub))
        jaccard_similarities.append(jac_sim)

    return (sum(rouge_l_similarities) * weight_rouge_l + \
            sum(levenshtein_similarities) * weight_levenshtein + \
            sum(jaccard_similarities) * weight_jaccard)/num_labeled


if __name__ == "__main__":
    labeled = [
        "David Chanoff U.S. Navy admiral collaboration",
        "U.S. Navy admiral ambassador to United Kingdom",
        "U.S. President during U.S. Navy admiral's ambassadorship",
    ]
    predicted_mismatch = [
        "David Chanoff U.S. Navy admiral collaboration",
        "U.S. Navy admiral ambassador to United Kingdom",
    ]
    predicted_typo = [
        "David Chanoff",
        "United Kingdom Small",
        "U.S. President Hahaha",
    ]
    predicted_rephrased = [
        "David Chanoff collaborates with US admiral in Navy",
        "US Navy admiral is ambassador to UK",
        "President of US during admiral's UK ambassadorship",
    ]


    print(score_function(labeled, predicted_mismatch))
    print(score_function(labeled, predicted_typo))
    print(score_function(labeled, predicted_rephrased))

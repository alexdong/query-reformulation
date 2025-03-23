from typing import List

import evaluate
from Levenshtein import distance as levenshtein_distance
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
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

# Load the ROUGE metric from Hugging Face evaluate
rouge = evaluate.load("rouge")

def rouge_l_similarity(reference_string: str, prediction_string: str) -> float:
    """Calculates ROUGE-L F1-score between two strings using Hugging Face evaluate."""
    scores = rouge.compute(predictions=[prediction_string], references=[reference_string], rouge_types=["rougeL"])
    return scores["rougeL"] # Returns ROUGE-L F1-score by default


def evaluate_query_reformulation_combined_rouge(
        labeled_subqueries: List[str],
        predicted_subqueries: List[str],
        weight_rouge_l: float = 0.5, 
        weight_levenshtein: float = 0.3, 
        weight_jaccard: float = 0.2
        ) -> dict:
    """
    Evaluates query reformulation using a combination of ROUGE-L, Levenshtein, and Jaccard similarity.

    Args:
        labeled_subqueries: List of labeled subquery strings.
        predicted_subqueries: List of predicted subquery strings.
        weight_rouge_l: Weight for ROUGE-L similarity.
        weight_levenshtein: Weight for Levenshtein similarity.
        weight_jaccard: Weight for Jaccard similarity.

    Returns:
        A dictionary containing evaluation scores.
    """
    num_labeled = len(labeled_subqueries)
    num_predicted = len(predicted_subqueries)

    if num_labeled != num_predicted:
        num_subqueries_score = 0
        combined_content_score = 0
        overall_score = 0
        similarity_details = ["Number of subqueries differs."]
        avg_rouge_l_sim = 0.0
        avg_levenshtein_sim = 0.0
        avg_jaccard_sim = 0.0
    else:
        num_subqueries_score = 1
        rouge_l_similarities = []
        levenshtein_similarities = []
        jaccard_similarities = []
        similarity_details = []

        for i in range(num_labeled):
            labeled_sub = labeled_subqueries[i].strip()
            predicted_sub = predicted_subqueries[i].strip()

            rouge_l_sim = rouge_l_similarity(labeled_sub, predicted_sub)
            rouge_l_similarities.append(rouge_l_sim)

            lev_sim = levenshtein_similarity(labeled_sub, predicted_sub)
            levenshtein_similarities.append(lev_sim)

            jac_sim = jaccard_similarity(preprocess_query_string(labeled_sub), preprocess_query_string(predicted_sub))
            jaccard_similarities.append(jac_sim)

            similarity_details.append(f"Subquery {i+1}: ROUGE-L={rouge_l_sim:.4f}, Levenshtein={lev_sim:.4f}, Jaccard={jac_sim:.4f} "
                                      f"(Labeled: '{labeled_sub}', Predicted: '{predicted_sub}')")

        avg_rouge_l_sim = sum(rouge_l_similarities) / num_labeled if num_labeled > 0 else 0.0
        avg_levenshtein_sim = sum(levenshtein_similarities) / num_labeled if num_labeled > 0 else 0.0
        avg_jaccard_sim = sum(jaccard_similarities) / num_labeled if num_labeled > 0 else 0.0

        combined_content_score = (weight_rouge_l * avg_rouge_l_sim) + (weight_levenshtein * avg_levenshtein_sim) + (weight_jaccard * avg_jaccard_sim)
        overall_score = (num_subqueries_score * 50) + (combined_content_score * 50)


    evaluation_output_strings = [
        f"Number of Labeled Subqueries: {num_labeled}",
        f"Number of Predicted Subqueries: {num_predicted}",
        f"Number of Subqueries Match: {num_subqueries_score} (1 if counts are same, 0 if different)",
        f"Avg. ROUGE-L Similarity: {avg_rouge_l_sim:.4f}",
        f"Avg. Levenshtein Similarity: {avg_levenshtein_sim:.4f}",
        f"Avg. Jaccard Similarity: {avg_jaccard_sim:.4f}",
        f"Combined Content Score (Weighted): {combined_content_score:.4f}",
        f"Overall Score: {overall_score:.2f}",
        "Subquery Similarity Details (ROUGE-L, Levenshtein, Jaccard):"
    ] + similarity_details
    print(evaluation_output_strings)

    return overall_score


# --- Example Usage (same examples as before) ---
labeled_subs_ex2 = [
    "David Chanoff U.S. Navy admiral collaboration",
    "U.S. Navy admiral ambassador to United Kingdom",
    "U.S. President during U.S. Navy admiral's ambassadorship"
]
predicted_subs_ex2_match = [
    "David Chanoff U.S. Navy admiral collaboration",
    "U.S. Navy admiral ambassador to United Kingdom",
    "U.S. President during U.S. Navy admiral's ambassadorship"
]
predicted_subs_ex2_typo = [
    "David Chanoff U.S. Nvy admiral collaboration", # Typo in "Navy"
    "U.S. Navy admiral ambassador to United Kingdom",
    "U.S. President during U.S. Navy admiral's ambassadorship"
]
predicted_subs_ex2_rephrased = [
    "David Chanoff collaborates with US admiral in Navy", # Rephrased first subquery
    "US Navy admiral is ambassador to UK", # Rephrased second
    "President of US during admiral's UK ambassadorship" # Rephrased third
]


# Example 2 - Exact Match (Combined with ROUGE-L)
result_ex2_combined_rouge_match = evaluate_query_reformulation_combined_rouge(labeled_subs_ex2, predicted_subs_ex2_match)
print("--- Example 2 - Exact Match (Combined with ROUGE-L) ---")
for line in result_ex2_combined_rouge_match["evaluation_output_strings"]:
    print(line)

# Example 2 - Typo (Combined with ROUGE-L)
result_ex2_combined_rouge_typo = evaluate_query_reformulation_combined_rouge(labeled_subs_ex2, predicted_subs_ex2_typo)
print("\n--- Example 2 - Typo (Combined with ROUGE-L) ---")
for line in result_ex2_combined_rouge_typo["evaluation_output_strings"]:
    print(line)

# Example 2 - Rephrased (Combined with ROUGE-L) - ROUGE-L should perform better than Levenshtein for rephrasing
result_ex2_combined_rouge_rephrased = evaluate_query_reformulation_combined_rouge(labeled_subs_ex2, predicted_subs_ex2_rephrased)
print("\n--- Example 2 - Rephrased (Combined with ROUGE-L) ---")
for line in result_ex2_combined_rouge_rephrased["evaluation_output_strings"]:
    print(line)

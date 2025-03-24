from typing import List

from Levenshtein import distance as levenshtein_distance

def levenshtein_similarity(s1: str, s2: str) -> float:
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2)) or 1
    normalized_distance = distance / max_len
    return 1.0 - normalized_distance


def score_function(
        labeled_subqueries: List[str],
        predicted_subqueries: List[str]
        ) -> float:
    num_labeled = len(labeled_subqueries)
    if num_labeled != len(predicted_subqueries):
        return 0.0
    
    levenshtein_similarities = []
    for i in range(num_labeled):
        labeled_sub = labeled_subqueries[i].strip()
        predicted_sub = predicted_subqueries[i].strip()
        lev_sim = levenshtein_similarity(labeled_sub, predicted_sub)
        levenshtein_similarities.append(lev_sim)

    return sum(levenshtein_similarities) / num_labeled


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

"""
Exact Evaluation Metric for Business Entity Resolution Challenge.
Computes Macro F0.5 Score per Source 1 entity including singleton handling,
as well as candidate recall and diagnostic metrics.
"""

from typing import Dict, Iterable, List, Mapping, Set, Tuple


def compute_entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.
    
    Singleton logic:
    - If true_ids is empty (true singleton):
      - If pred_ids is empty -> score 1.0 (precision 1.0, recall 1.0)
      - If pred_ids is non-empty -> score 0.0 (precision 0.0, recall 0.0)
    
    Non-singleton logic:
    - If pred_ids is empty -> 0.0 (precision 0.0, recall 0.0)
    - Otherwise:
      - TP = len(true_ids & pred_ids)
      - P = TP / len(pred_ids)
      - R = TP / len(true_ids)
      - F0.5 = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    beta_sq = beta ** 2
    weight = 1.0 + beta_sq  # 1.25 for beta=0.5

    # True singleton case
    if len(true_ids) == 0:
        if len(pred_ids) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Non-singleton true, but model predicted empty
    if len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    # Standard case
    tp = len(true_ids & pred_ids)
    p = tp / float(len(pred_ids))
    r = tp / float(len(true_ids))

    denom = (beta_sq * p) + r
    if denom == 0.0 or (p + r) == 0.0:
        f05 = 0.0
    else:
        f05 = (weight * p * r) / denom

    return f05, p, r


def evaluate_predictions(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth across all Source 1 entities.
    Returns dictionary with:
    - macro_f05: The official competition metric
    - macro_precision
    - macro_recall
    - singleton_accuracy
    - num_entities
    - num_true_singletons
    - num_pred_singletons
    """
    f05_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    true_singletons = 0
    correct_singletons = 0
    pred_singletons = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = set(predictions.get(s1_id, []))

        if len(true_set) == 0:
            true_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1

        if len(pred_set) == 0:
            pred_singletons += 1

        f05, p, r = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f05)
        precision_scores.append(p)
        recall_scores.append(r)

    total_s1 = len(ground_truth)
    macro_f05 = sum(f05_scores) / total_s1 if total_s1 > 0 else 0.0
    macro_p = sum(precision_scores) / total_s1 if total_s1 > 0 else 0.0
    macro_r = sum(recall_scores) / total_s1 if total_s1 > 0 else 0.0

    singleton_acc = (
        (correct_singletons / true_singletons) if true_singletons > 0 else 1.0
    )

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "num_entities": total_s1,
        "num_true_singletons": true_singletons,
        "num_pred_singletons": pred_singletons,
    }


def compute_candidate_recall(
    ground_truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> Dict[str, float]:
    """
    Measures blocking quality: candidate recall and reduction ratio.
    Candidate recall: fraction of all true matching pairs present in candidate pairs.
    """
    total_true_pairs = 0
    retrieved_true_pairs = 0
    total_candidate_pairs = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        total_true_pairs += len(true_set)
        cand_set = set(candidates.get(s1_id, []))
        total_candidate_pairs += len(cand_set)
        retrieved_true_pairs += len(true_set & cand_set)

    cand_recall = (
        (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    )

    return {
        "candidate_recall": cand_recall,
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "missed_true_pairs": total_true_pairs - retrieved_true_pairs,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_s1": (
            total_candidate_pairs / len(ground_truth) if ground_truth else 0.0
        ),
    }

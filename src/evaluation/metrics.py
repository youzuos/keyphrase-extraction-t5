from typing import List, Tuple


def calculate_f1_precision_recall(
    true_keyphrases: List[str],
    predicted_keyphrases: List[str]
) -> Tuple[float, float, float]:
    if not true_keyphrases and not predicted_keyphrases:
        return 1.0, 1.0, 1.0
    
    if not true_keyphrases:
        return 0.0, 0.0, 0.0
    
    if not predicted_keyphrases:
        return 0.0, 0.0, 0.0
    
    true_set = set(kp.lower().strip() for kp in true_keyphrases)
    pred_set = set(kp.lower().strip() for kp in predicted_keyphrases)
    
    intersection = true_set & pred_set
    
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(true_set) if true_set else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def calculate_average_metrics(
    all_true_keyphrases: List[List[str]],
    all_predicted_keyphrases: List[List[str]]
) -> dict:
    if len(all_true_keyphrases) != len(all_predicted_keyphrases):
        raise ValueError("Mismatch in number of true and predicted keyphrases")
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for true_kps, pred_kps in zip(all_true_keyphrases, all_predicted_keyphrases):
        precision, recall, f1 = calculate_f1_precision_recall(true_kps, pred_kps)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    avg_true_count = sum(len(kps) for kps in all_true_keyphrases) / len(all_true_keyphrases) if all_true_keyphrases else 0
    avg_pred_count = sum(len(kps) for kps in all_predicted_keyphrases) / len(all_predicted_keyphrases) if all_predicted_keyphrases else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'num_samples': len(all_true_keyphrases),
        'avg_true_keyphrases': avg_true_count,
        'avg_predicted_keyphrases': avg_pred_count,
    }

from typing import Any, Dict
from math_verify import parse, StringExtractionConfig, LatexExtractionConfig, verify


def extract_pred_and_parse(solution, data_name):
    if data_name in ["gpqa"]:
        pred = parse(
            solution,
            extraction_config=[StringExtractionConfig(lowercase=False)],
        )
    elif "boxed" in solution:
        pred = parse(
            solution, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )
    else:
        pred = []
    return pred


def parse_ground_truth(example: Dict[str, Any], data_name):
    if data_name in ["gpqa"]:
        abcd = "ABCD"
        answer = "$" + abcd[example["answer"]] + "$"
    else:
        answer = "$" + example["answer"] + "$"
    return parse(answer)


def obtain_nHm_scores_and_preds(gt, sample_preds):
    """
    Process the predictions with a 3D structure: n x H x m
    
    Args:
        gt: Ground truth answer
        preds: List of predictions in 3D structure [n][H][m]
    
    Returns:
        Processed ground truth, predictions, and scores in hierarchical structure
    """
    # Convert ground truth to string format
    new_gt = str(gt[0])
    nHm_scores = []
    nHm_preds = []

    for Hm_preds in sample_preds:
        Hm_scores = []
        Hm_new_preds = []

        for m_preds in Hm_preds:  # For each chunk
            m_scores = [verify(gt, m_pred) for m_pred in m_preds]
            Hm_scores.append(m_scores)
            
            # Format predictions for this chunk
            m_new_preds = []
            for pred, score in zip(m_preds, m_scores):
                if score:
                    m_new_preds.append(new_gt)
                else:
                    if pred:
                        m_new_preds.append(str(pred[0]))
                    else:
                        m_new_preds.append("")
            
            Hm_new_preds.append(m_new_preds)
        
        nHm_scores.append(Hm_scores)
        nHm_preds.append(Hm_new_preds)
    
    return new_gt, nHm_preds, nHm_scores
import pandas as pd
import numpy as np

def calculate_recall(candidates_df, gt_df):
    """
    Measures the candidate recall: how many of the true matched_entity_ids are present in candidates_df.
    candidates_df: expected to have 'entity_id_s1' and 'entity_id_cand'
    gt_df: ground truth dataframe containing 'source1_entity_id' and 'matched_entity_ids'
    """
    # Create a DataFrame of true positive pairs
    true_pairs_list = []
    total_true_matches = 0
    for _, row in gt_df.iterrows():
        if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == "":
            continue
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(',')
        for m in matches:
            true_pairs_list.append((s1_id, m))
            total_true_matches += 1
            
    if total_true_matches == 0:
        return 0, 0
        
    true_df = pd.DataFrame(true_pairs_list, columns=['entity_id_s1', 'entity_id_cand'])
    
    # Fast inner join to find matches
    matched_df = pd.merge(true_df, candidates_df, on=['entity_id_s1', 'entity_id_cand'], how='inner')
    found = len(matched_df.drop_duplicates())
    
    recall = found / total_true_matches
    return recall, total_true_matches

def evaluate_predictions(predictions, gt_df):
    """
    predictions: df with 'source1_entity_id' and 'matched_entity_ids' (predicted)
    Calculates precision, recall, F1, and F0.5
    """
    true_pairs = set()
    for _, row in gt_df.iterrows():
        if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == "":
            continue
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(',')
        for m in matches:
            true_pairs.add((s1_id, m))
            
    pred_pairs = set()
    for _, row in predictions.iterrows():
        if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == "":
            continue
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(',')
        for m in matches:
            pred_pairs.add((s1_id, m))
            
    tp = len(true_pairs.intersection(pred_pairs))
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f0_5 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f0_5,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def print_candidate_stats(candidates_df, name=""):
    num_cands = len(candidates_df)
    cands_per_s1 = candidates_df.groupby('entity_id_s1').size()
    print(f"--- Candidate Stats: {name} ---")
    print(f"Total candidate pairs: {num_cands}")
    print(f"Average candidates per S1: {cands_per_s1.mean():.2f}")
    print(f"Median candidates per S1: {cands_per_s1.median()}")
    print(f"Max candidates per S1: {cands_per_s1.max()}")
    print("--------------------------------")

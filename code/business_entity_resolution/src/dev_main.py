import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from config import OUTPUT_DIR
from data_loading import load_data
from retrieval import normalize_dataframe, generate_exact_match_candidates
from evaluation import calculate_recall, evaluate_predictions, print_candidate_stats
from features import build_features

def main():
    # 1. Load Data with strict Train/Val Split (10% validation)
    s1_train, s1_val, s2, s3, gt_train, gt_val = load_data(split_ratio=0.1)
    
    # 2. Normalization
    print("Normalizing text fields...")
    s1_train = normalize_dataframe(s1_train)
    s1_val = normalize_dataframe(s1_val)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    # 3. Candidate Generation & Recall Tracking (Validation Set Only for evaluation)
    print("Generating validation candidates (Exact Name)...")
    cand_name_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_name")
    recall_name, total_val_matches = calculate_recall(cand_name_val, gt_val)
    print(f"Exact Name Recall: {recall_name*100:.2f}%")
    
    print("Generating validation candidates (Exact Address)...")
    cand_addr_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_address")
    recall_addr, _ = calculate_recall(cand_addr_val, gt_val)
    print(f"Exact Address Recall: {recall_addr*100:.2f}%")
    
    cand_val_union = pd.concat([cand_name_val, cand_addr_val], ignore_index=True).drop_duplicates()
    recall_union, _ = calculate_recall(cand_val_union, gt_val)
    
    print_candidate_stats(cand_val_union, name="Validation Union")
    
    # Freeze Candidates for Validation
    # In a real run, you'd save it to output/development/validation/candidate_pairs.tsv here
    
    # 4. Feature Engineering
    print("Building features for Validation Set...")
    X_val, df_val = build_features(cand_val_union, s1_val, s2, s3)
    
    print("Building features for Training Set...")
    cand_name_train = generate_exact_match_candidates(s1_train, s2, s3, "norm_name")
    cand_addr_train = generate_exact_match_candidates(s1_train, s2, s3, "norm_address")
    cand_train_union = pd.concat([cand_name_train, cand_addr_train], ignore_index=True).drop_duplicates()
    X_train, df_train = build_features(cand_train_union, s1_train, s2, s3)
    
    # 5. Label Assignment
    def get_labels(df_features, gt):
        true_pairs = set()
        for _, row in gt.iterrows():
            if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == "":
                continue
            s1_id = row['source1_entity_id']
            matches = row['matched_entity_ids'].split(',')
            for m in matches:
                true_pairs.add((s1_id, m))
        return df_features.apply(lambda r: 1 if (r['entity_id_s1'], r['entity_id_cand']) in true_pairs else 0, axis=1)

    y_train = get_labels(df_train, gt_train)
    
    # 6. Model Training
    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 7. Validation Inference & Threshold Sweep
    print("Running Inference on Validation...")
    preds_proba = model.predict_proba(X_val)[:, 1]
    df_val['score'] = preds_proba
    
    print("\n--- Threshold Sweep ---")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_f0_5 = 0
    best_thresh = 0.5
    best_metrics = {}
    
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'F0.5':<10}")
    for t in thresholds:
        matches = df_val[df_val['score'] >= t]
        match_grouped = matches.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ','.join(x)).reset_index()
        match_grouped.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'matched_entity_ids'}, inplace=True)
        all_s1 = s1_val[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
        preds_df = pd.merge(all_s1, match_grouped, on='source1_entity_id', how='left')
        preds_df['matched_entity_ids'] = preds_df['matched_entity_ids'].fillna("")
        
        metrics = evaluate_predictions(preds_df, gt_val)
        print(f"{t:<10} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f} | {metrics['f0_5']:<10.4f}")
        
        if metrics['f0_5'] > best_f0_5:
            best_f0_5 = metrics['f0_5']
            best_thresh = t
            best_metrics = metrics
            
    print(f"\nSelected Threshold: {best_thresh} with F0.5={best_f0_5:.4f}")
    
    # 8. Entity-level distribution
    final_matches = df_val[df_val['score'] >= best_thresh]
    cands_per_s1 = final_matches.groupby('entity_id_s1').size()
    
    zero_match = len(s1_val) - len(cands_per_s1)
    one_match = (cands_per_s1 == 1).sum()
    multi_match = (cands_per_s1 > 1).sum()
    
    print("\n========================================")
    print("ENTITY RESOLUTION DEVELOPMENT REPORT")
    print("========================================")
    print("\nDATA")
    print(f"Training entities: {len(s1_train)}")
    print(f"Validation entities: {len(s1_val)}")
    
    print("\nCANDIDATE GENERATION")
    print(f"Exact name recall: {recall_name*100:.2f}%")
    print(f"Exact address recall: {recall_addr*100:.2f}%")
    print(f"Union recall: {recall_union*100:.2f}%")
    print(f"Candidate count Final candidate_pairs.tsv: {len(cand_val_union)}")
    
    print("\nMODEL")
    print("Model: LightGBM")
    print("Number of features: 3")
    
    print("\nVALIDATION")
    print(f"Threshold: {best_thresh}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1: {best_metrics['f1']:.4f}")
    print(f"F0.5: {best_metrics['f0_5']:.4f}")
    print(f"TP: {best_metrics['tp']}")
    print(f"FP: {best_metrics['fp']}")
    print(f"FN: {best_metrics['fn']}")
    
    print("\nENTITY-LEVEL")
    print(f"Zero matches: {zero_match}")
    print(f"One match: {one_match}")
    print(f"Multiple matches: {multi_match}")
    print("========================================")

if __name__ == "__main__":
    main()

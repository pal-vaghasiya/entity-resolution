import pandas as pd
import numpy as np
import os
import gc
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import lightgbm as lgb
from fuzzywuzzy import fuzz
from scipy.sparse import vstack, csr_matrix
from ablation_blocking import get_rare_tokens_blocking, get_name_context_blocking, get_numeric_address_blocking

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_dataframe(df):
    df["norm_name"] = df["business_name"].apply(normalize_text)
    df["norm_address"] = df["business_address"].apply(normalize_text)
    return df

def generate_exact_match_candidates(s1_df, s2_df, s3_df, match_column="norm_name"):
    """
    Generate candidates by exactly matching a specific column.
    """
    candidates = []
    
    # Match S1 with S2
    s1_s2 = pd.merge(s1_df[['entity_id', match_column]], 
                     s2_df[['entity_id', match_column]], 
                     on=match_column, 
                     how='inner', 
                     suffixes=('_s1', '_cand'))
    if not s1_s2.empty:
        s1_s2 = s1_s2[s1_s2[match_column] != ""]
        candidates.append(s1_s2[['entity_id_s1', 'entity_id_cand']])
        
    # Match S1 with S3
    s1_s3 = pd.merge(s1_df[['entity_id', match_column]], 
                     s3_df[['entity_id', match_column]], 
                     on=match_column, 
                     how='inner', 
                     suffixes=('_s1', '_cand'))
    if not s1_s3.empty:
        s1_s3 = s1_s3[s1_s3[match_column] != ""]
        candidates.append(s1_s3[['entity_id_s1', 'entity_id_cand']])
        
    if candidates:
        cand_df = pd.concat(candidates, ignore_index=True)
        cand_df = cand_df.drop_duplicates()
        return cand_df
    else:
        return pd.DataFrame(columns=['entity_id_s1', 'entity_id_cand'])

def build_features(candidate_pairs, s1_df, s2_df, s3_df):
    # Combine S2 and S3 for easy lookup
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Merge text features
    df = pd.merge(candidate_pairs, s1_df, left_on='entity_id_s1', right_on='entity_id', how='left')
    df = pd.merge(df, s23_df, left_on='entity_id_cand', right_on='entity_id', how='left', suffixes=('_1', '_2'))
    
    # Calculate simple features
    def calc_fuzz_ratio(row):
        return fuzz.ratio(str(row['norm_name_1']), str(row['norm_name_2']))
    
    def calc_addr_ratio(row):
        return fuzz.ratio(str(row['norm_address_1']), str(row['norm_address_2']))

    tqdm.pandas(desc="Calculating name fuzzy ratio")
    df['name_sim'] = df.progress_apply(calc_fuzz_ratio, axis=1)
    
    tqdm.pandas(desc="Calculating address fuzzy ratio")
    df['addr_sim'] = df.progress_apply(calc_addr_ratio, axis=1)
    
    df['country_match'] = (df['country_1'] == df['country_2']).astype(int)
    
    features = ['name_sim', 'addr_sim', 'country_match']
    return df[features], df

def process_pipeline(mode="train"):
    """
    mode can be "train" or "test"
    """
    print(f"[{mode}] Loading data...")
    s1 = pd.read_csv(os.path.join(DATA_DIR, mode, f"{mode}_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(DATA_DIR, mode, f"{mode}_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(DATA_DIR, mode, f"{mode}_source3.tsv"), sep="\t")
    
    print(f"[{mode}] Normalizing data...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    print(f"[{mode}] Generating candidates (Exact Name)...")
    cand_name = generate_exact_match_candidates(s1, s2, s3, "norm_name")
    print(f"[{mode}] Generated {len(cand_name)} candidates from Name match.")
    
    print(f"[{mode}] Generating candidates (Exact Address)...")
    cand_addr = generate_exact_match_candidates(s1, s2, s3, "norm_address")
    print(f"[{mode}] Generated {len(cand_addr)} candidates from Address match.")
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    print(f"[{mode}] Generating candidates (Rare Tokens)...")
    cand_rare = get_rare_tokens_blocking(s1, s23, top_n=2)
    print(f"[{mode}] Generated {len(cand_rare)} candidates from Rare Tokens.")
    
    print(f"[{mode}] Generating candidates (Name + Context)...")
    cand_ctx = get_name_context_blocking(s1, s23)
    print(f"[{mode}] Generated {len(cand_ctx)} candidates from Name + Context.")
    
    print(f"[{mode}] Generating candidates (Numeric Address)...")
    cand_num = get_numeric_address_blocking(s1, s23)
    print(f"[{mode}] Generated {len(cand_num)} candidates from Numeric Address.")
    
    # Union candidates
    candidates = pd.concat([cand_name, cand_addr, cand_rare, cand_ctx, cand_num], ignore_index=True).drop_duplicates()
    print(f"[{mode}] Total unique candidates: {len(candidates)}")
    
    # Stage 1 Candidate Output
    if mode == "test":
        print(f"[{mode}] Writing candidate_pairs.tsv ...")
        # Format: source1_entity_id, candidate_entity_ids (comma separated)
        cand_grouped = candidates.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ','.join(x)).reset_index()
        cand_grouped.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_ids'}, inplace=True)
        # Ensure all S1 entities are present
        all_s1 = s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
        cand_grouped = pd.merge(all_s1, cand_grouped, on='source1_entity_id', how='left')
        cand_grouped['candidate_entity_ids'] = cand_grouped['candidate_entity_ids'].fillna("")
        cand_grouped.to_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
        
    print(f"[{mode}] Building features...")
    X, df_features = build_features(candidates, s1, s2, s3)
    
    if mode == "train":
        print(f"[{mode}] Loading ground truth...")
        gt = pd.read_csv(os.path.join(DATA_DIR, mode, f"{mode}_ground_truth.tsv"), sep="\t")
        
        # Create a set of true positive pairs for fast lookup
        true_pairs = set()
        for _, row in gt.iterrows():
            if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == "":
                continue
            s1_id = row['source1_entity_id']
            matches = row['matched_entity_ids'].split(',')
            for m in matches:
                true_pairs.add((s1_id, m))
                
        # Assign labels
        print(f"[{mode}] Assigning labels...")
        df_features['is_match'] = df_features.apply(lambda r: 1 if (r['entity_id_s1'], r['entity_id_cand']) in true_pairs else 0, axis=1)
        y = df_features['is_match']
        return X, y, df_features
    else:
        return X, candidates, df_features, s1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train
    X_train, y_train, df_train = process_pipeline("train")
    
    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Test
    X_test, candidates_test, df_test, s1_test = process_pipeline("test")
    
    print("Predicting...")
    preds = model.predict_proba(X_test)[:, 1]
    df_test['score'] = preds
    
    # Match decision
    threshold = 0.5
    matches = df_test[df_test['score'] >= threshold]
    
    print("Writing matching_results.tsv ...")
    match_grouped = matches.groupby('entity_id_s1')['entity_id_cand'].apply(lambda x: ','.join(x)).reset_index()
    match_grouped.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'matched_entity_ids'}, inplace=True)
    
    # Ensure all S1 entities are present
    all_s1 = s1_test[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    final_output = pd.merge(all_s1, match_grouped, on='source1_entity_id', how='left')
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna("")
    
    final_output.to_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()

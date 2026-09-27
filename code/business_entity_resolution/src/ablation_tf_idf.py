import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack

from data_loading import load_data
from retrieval import normalize_dataframe, generate_exact_match_candidates
from evaluation import calculate_recall

def get_top_k_sparse(query_sparse, target_sparse, query_ids, target_ids, k=10):
    batch_size = 500
    results = []
    
    for i in range(0, query_sparse.shape[0], batch_size):
        end = min(i + batch_size, query_sparse.shape[0])
        batch = query_sparse[i:end]
        
        # Sparse dot product -> (batch_size, N_target)
        sim_matrix = batch.dot(target_sparse.T)
        
        for j in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(j)
            if row.nnz == 0:
                continue
            
            data = row.data
            indices = row.indices
            
            if len(data) > k:
                top_k_idx_local = np.argpartition(data, -k)[-k:]
                top_k_idx = indices[top_k_idx_local]
            else:
                top_k_idx = indices
                
            query_id = query_ids[i + j]
            for t_idx in top_k_idx:
                results.append((query_id, target_ids[t_idx]))
                
    return pd.DataFrame(results, columns=['entity_id_s1', 'entity_id_cand'])

def run_tfidf_ablation():
    print("Loading data...")
    # Load with a split. We only evaluate on validation set (10%)
    _, s1_val, s2, s3, _, gt_val = load_data(split_ratio=0.1)
    
    print("Normalizing data...")
    s1_val = normalize_dataframe(s1_val)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    # Baseline: Exact Name + Exact Address
    print("\n--- Running Baseline ---")
    start_time = time.time()
    cand_name_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_name")
    cand_addr_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_address")
    baseline_cands = pd.concat([cand_name_val, cand_addr_val], ignore_index=True).drop_duplicates()
    baseline_time = time.time() - start_time
    
    baseline_recall, _ = calculate_recall(baseline_cands, gt_val)
    baseline_cands_per_s1 = baseline_cands.groupby('entity_id_s1').size()
    
    # TF-IDF Retrieval
    print("\n--- Running Character TF-IDF ---")
    start_time = time.time()
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    target_ids = s23['entity_id'].values
    query_ids = s1_val['entity_id'].values
    
    print("Fitting TF-IDF Name...")
    # Train on all available target data to build vocabulary (no ground truth used, so safe)
    vectorizer_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), min_df=5, max_df=0.5)
    target_sparse_name = vectorizer_name.fit_transform(s23['norm_name'].fillna(""))
    query_sparse_name = vectorizer_name.transform(s1_val['norm_name'].fillna(""))
    
    print("Retrieving Top-K TF-IDF Name (K=10)...")
    cand_tfidf_name = get_top_k_sparse(query_sparse_name, target_sparse_name, query_ids, target_ids, k=10)
    tfidf_name_recall, _ = calculate_recall(cand_tfidf_name, gt_val)
    
    print("Fitting TF-IDF Address...")
    vectorizer_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), min_df=5, max_df=0.5)
    target_sparse_addr = vectorizer_addr.fit_transform(s23['norm_address'].fillna(""))
    query_sparse_addr = vectorizer_addr.transform(s1_val['norm_address'].fillna(""))
    
    print("Retrieving Top-K TF-IDF Address (K=10)...")
    cand_tfidf_addr = get_top_k_sparse(query_sparse_addr, target_sparse_addr, query_ids, target_ids, k=10)
    tfidf_addr_recall, _ = calculate_recall(cand_tfidf_addr, gt_val)
    
    print("Unioning TF-IDF candidates...")
    new_cands = pd.concat([baseline_cands, cand_tfidf_name, cand_tfidf_addr], ignore_index=True).drop_duplicates()
    tfidf_time = time.time() - start_time
    
    new_recall, _ = calculate_recall(new_cands, gt_val)
    new_cands_per_s1 = new_cands.groupby('entity_id_s1').size()
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print("\n========================================")
    print("RETRIEVAL ABLATION — CHARACTER TF-IDF")
    print("========================================")
    print("\nBaseline:")
    print("Exact Name + Exact Address\n")
    print(f"Candidate Recall: {baseline_recall*100:.2f}%")
    print(f"Candidate Pairs: {len(baseline_cands)}")
    print(f"Avg / S1: {baseline_cands_per_s1.mean():.2f}")
    print(f"Median / S1: {baseline_cands_per_s1.median():.2f}")
    print(f"P95 / S1: {baseline_cands_per_s1.quantile(0.95):.2f}")
    print(f"Runtime: {baseline_time:.2f}s")
    
    print("\nCharacter TF-IDF:")
    print(f"Character TF-IDF Name Recall: {tfidf_name_recall*100:.2f}%")
    print(f"Character TF-IDF Address Recall: {tfidf_addr_recall*100:.2f}%")
    print(f"Final UNION recall: {new_recall*100:.2f}%")
    print(f"Candidate Pairs: {len(new_cands)}")
    print(f"Avg / S1: {new_cands_per_s1.mean():.2f}")
    print(f"Median / S1: {new_cands_per_s1.median():.2f}")
    print(f"P95 / S1: {new_cands_per_s1.quantile(0.95):.2f}")
    print(f"Runtime: {tfidf_time:.2f}s")
    print(f"Memory: {memory_mb:.2f} MB")
    
    incremental = (new_recall - baseline_recall) * 100
    print(f"\nIncremental Recall: +{incremental:.2f} pp")
    print("========================================")

if __name__ == "__main__":
    run_tfidf_ablation()

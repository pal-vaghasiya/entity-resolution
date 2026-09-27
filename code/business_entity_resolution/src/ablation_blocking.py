import os
import time
import psutil
import pandas as pd
import numpy as np
import re

from data_loading import load_data
from retrieval import normalize_dataframe, generate_exact_match_candidates
from evaluation import calculate_recall

def get_rare_tokens_blocking(s1, s23, top_n=2):
    # 1. Compute token frequencies on S23 (reference corpus)
    print("Computing token frequencies for Rare Token Block...")
    all_tokens = s23['norm_name'].str.split().explode().dropna()
    token_counts = all_tokens.value_counts()
    
    # Only keep genuinely rare tokens that appear < 20 times to prevent candidate explosion.
    valid_tokens = set(token_counts[token_counts < 20].index)
    
    def extract_rare(text):
        if not isinstance(text, str) or text == "": return []
        tokens = text.split()
        valid = [t for t in tokens if t in valid_tokens and len(t) > 2]
        if not valid: return []
        # Sort by frequency (rarest first)
        valid.sort(key=lambda x: token_counts.get(x, float('inf')))
        return valid[:top_n]
        
    s1_rare = s1.copy()
    s23_rare = s23.copy()
    
    s1_rare['rare_token'] = s1_rare['norm_name'].apply(extract_rare)
    s23_rare['rare_token'] = s23_rare['norm_name'].apply(extract_rare)
    
    s1_exp = s1_rare.explode('rare_token').dropna(subset=['rare_token'])
    s23_exp = s23_rare.explode('rare_token').dropna(subset=['rare_token'])
    
    print("Joining Rare Token Block...")
    cands = pd.merge(s1_exp[['entity_id', 'rare_token']], 
                     s23_exp[['entity_id', 'rare_token']], 
                     on='rare_token', how='inner', suffixes=('_s1', '_cand'))
    return cands[['entity_id_s1', 'entity_id_cand']].drop_duplicates()


def get_name_context_blocking(s1, s23):
    print("Computing Name + Context Block...")
    s1_ctx = s1.copy()
    s23_ctx = s23.copy()
    
    s1_ctx['first_word'] = s1_ctx['norm_name'].str.split().str[0]
    s23_ctx['first_word'] = s23_ctx['norm_name'].str.split().str[0]
    
    s1_ctx['country'] = s1_ctx['country'].astype(str).str.lower().str.strip()
    s23_ctx['country'] = s23_ctx['country'].astype(str).str.lower().str.strip()
    
    # Only use valid countries
    s1_ctx = s1_ctx[(s1_ctx['country'] != 'nan') & (s1_ctx['country'] != '')]
    s23_ctx = s23_ctx[(s23_ctx['country'] != 'nan') & (s23_ctx['country'] != '')]
    
    # Prevent explosion by removing extremely common first words (e.g. "the", "a")
    fw_counts = s23_ctx['first_word'].value_counts()
    valid_fw = set(fw_counts[fw_counts < 200].index)
    s1_ctx = s1_ctx[s1_ctx['first_word'].isin(valid_fw)]
    s23_ctx = s23_ctx[s23_ctx['first_word'].isin(valid_fw)]
    
    s1_ctx['name_country'] = s1_ctx['first_word'] + "_" + s1_ctx['country']
    s23_ctx['name_country'] = s23_ctx['first_word'] + "_" + s23_ctx['country']
    
    print("Joining Name + Context Block...")
    cands = pd.merge(s1_ctx[['entity_id', 'name_country']], 
                     s23_ctx[['entity_id', 'name_country']], 
                     on='name_country', how='inner', suffixes=('_s1', '_cand'))
    return cands[['entity_id_s1', 'entity_id_cand']].drop_duplicates()


def get_numeric_address_blocking(s1, s23):
    print("Computing Numeric Address Block...")
    def extract_nums(text):
        if not isinstance(text, str): return []
        return re.findall(r'\b\d+\b', text)
        
    all_nums = s23['norm_address'].apply(extract_nums).explode().dropna()
    num_counts = all_nums.value_counts()
    
    # Only keep numbers that appear < 20 times to avoid explosion
    valid_nums = set(num_counts[num_counts < 20].index)
    
    def extract_informative_nums(text):
        if not isinstance(text, str): return []
        nums = re.findall(r'\b\d+\b', text)
        # Keep numbers >= 2 digits that are rare enough
        valid = [n for n in nums if n in valid_nums and len(n) >= 2]
        # Keep up to top 2 longest numbers (often most specific)
        valid.sort(key=len, reverse=True)
        return valid[:2]
        
    s1_num = s1.copy()
    s23_num = s23.copy()
    
    s1_num['addr_num'] = s1_num['norm_address'].apply(extract_informative_nums)
    s23_num['addr_num'] = s23_num['norm_address'].apply(extract_informative_nums)
    
    s1_exp = s1_num.explode('addr_num').dropna(subset=['addr_num'])
    s23_exp = s23_num.explode('addr_num').dropna(subset=['addr_num'])
    
    print("Joining Numeric Address Block...")
    cands = pd.merge(s1_exp[['entity_id', 'addr_num']], 
                     s23_exp[['entity_id', 'addr_num']], 
                     on='addr_num', how='inner', suffixes=('_s1', '_cand'))
    return cands[['entity_id_s1', 'entity_id_cand']].drop_duplicates()


def get_stats(cands_df):
    if len(cands_df) == 0:
        return 0, 0, 0, 0, 0, 0
    counts = cands_df.groupby('entity_id_s1').size()
    return (len(cands_df), 
            counts.mean(), 
            counts.median(), 
            counts.quantile(0.95), 
            counts.quantile(0.99), 
            counts.max())

def run_blocking_ablation():
    print("Loading data...")
    _, s1_val, s2, s3, _, gt_val = load_data(split_ratio=0.1)
    
    print("Normalizing data...")
    s1_val = normalize_dataframe(s1_val)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    start_time = time.time()
    
    # 1. Baseline
    cand_name_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_name")
    cand_addr_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_address")
    baseline_cands = pd.concat([cand_name_val, cand_addr_val], ignore_index=True).drop_duplicates()
    base_rec, _ = calculate_recall(baseline_cands, gt_val)
    base_stats = get_stats(baseline_cands)
    
    # 2. Rare Token
    cand_rare = get_rare_tokens_blocking(s1_val, s23, top_n=2)
    rare_rec, _ = calculate_recall(cand_rare, gt_val)
    rare_stats = get_stats(cand_rare)
    
    union_1 = pd.concat([baseline_cands, cand_rare], ignore_index=True).drop_duplicates()
    u1_rec, _ = calculate_recall(union_1, gt_val)
    
    # 3. Name + Context
    cand_ctx = get_name_context_blocking(s1_val, s23)
    ctx_rec, _ = calculate_recall(cand_ctx, gt_val)
    ctx_stats = get_stats(cand_ctx)
    
    union_2 = pd.concat([union_1, cand_ctx], ignore_index=True).drop_duplicates()
    u2_rec, _ = calculate_recall(union_2, gt_val)
    
    # 4. Numeric Address
    cand_num = get_numeric_address_blocking(s1_val, s23)
    num_rec, _ = calculate_recall(cand_num, gt_val)
    num_stats = get_stats(cand_num)
    
    final_union = pd.concat([union_2, cand_num], ignore_index=True).drop_duplicates()
    final_rec, _ = calculate_recall(final_union, gt_val)
    final_stats = get_stats(final_union)
    
    runtime = time.time() - start_time
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Print Output
    print("\n================================================")
    print("TARGETED HASH BLOCKING ABLATION")
    print("================================================")
    
    print("\nBASELINE")
    print("Exact Name + Exact Address")
    print(f"\nRecall: {base_rec*100:.2f}%")
    print(f"Candidates: {base_stats[0]}")
    print(f"Average/S1: {base_stats[1]:.2f}")
    print(f"Median/S1: {base_stats[2]:.0f}")
    print(f"P95/S1: {base_stats[3]:.0f}")
    print(f"P99/S1: {base_stats[4]:.0f}")
    print(f"Maximum: {base_stats[5]}")
    
    print("\n------------------------------------------------")
    print("BLOCK 1 — RARE NAME TOKEN")
    print(f"\nRecall: {rare_rec*100:.2f}%")
    print(f"Candidates: {rare_stats[0]}")
    print(f"Average/S1: {rare_stats[1]:.2f}")
    print(f"Median/S1: {rare_stats[2]:.0f}")
    print(f"P95/S1: {rare_stats[3]:.0f}")
    print(f"P99/S1: {rare_stats[4]:.0f}")
    print(f"Maximum: {rare_stats[5]}")
    print(f"\nIncremental Recall: +{(u1_rec - base_rec)*100:.2f} pp")
    
    print("\n------------------------------------------------")
    print("BLOCK 2 — NAME + CONTEXT")
    print(f"\nRecall: {ctx_rec*100:.2f}%")
    print(f"Candidates: {ctx_stats[0]}")
    print(f"Average/S1: {ctx_stats[1]:.2f}")
    print(f"Median/S1: {ctx_stats[2]:.0f}")
    print(f"P95/S1: {ctx_stats[3]:.0f}")
    print(f"P99/S1: {ctx_stats[4]:.0f}")
    print(f"Maximum: {ctx_stats[5]}")
    print(f"\nIncremental Recall: +{(u2_rec - u1_rec)*100:.2f} pp")
    
    print("\n------------------------------------------------")
    print("BLOCK 3 — INFORMATIVE NUMERIC ADDRESS TOKEN")
    print(f"\nRecall: {num_rec*100:.2f}%")
    print(f"Candidates: {num_stats[0]}")
    print(f"Average/S1: {num_stats[1]:.2f}")
    print(f"Median/S1: {num_stats[2]:.0f}")
    print(f"P95/S1: {num_stats[3]:.0f}")
    print(f"P99/S1: {num_stats[4]:.0f}")
    print(f"Maximum: {num_stats[5]}")
    print(f"\nIncremental Recall: +{(final_rec - u2_rec)*100:.2f} pp")
    
    print("\n------------------------------------------------")
    print("CUMULATIVE UNION")
    print("\nExact")
    print("+ Rare Token")
    print("+ Context")
    print("+ Numeric Address\n")
    print(f"Final Candidate Recall: {final_rec*100:.2f}%\n")
    print(f"Total Candidate Pairs: {final_stats[0]}")
    print(f"Average/S1: {final_stats[1]:.2f}")
    print(f"Median/S1: {final_stats[2]:.0f}")
    print(f"P95/S1: {final_stats[3]:.0f}")
    print(f"P99/S1: {final_stats[4]:.0f}")
    print(f"Maximum: {final_stats[5]}")
    
    print(f"\nRuntime: {runtime:.2f}s")
    print(f"Peak Memory: {memory_mb:.2f} MB")
    print("\n================================================")
    
    print("\n--- ANALYSIS ANSWERS ---")
    print(f"1. Total improvement over 27.59% baseline: +{(final_rec - base_rec)*100:.2f} pp")
    
    incs = {
        "Rare Name Token": (u1_rec - base_rec)*100,
        "Name + Context": (u2_rec - u1_rec)*100,
        "Numeric Address Token": (final_rec - u2_rec)*100
    }
    best_inc = max(incs, key=incs.get)
    print(f"2. Block contributing the most incremental recall: {best_inc} (+{incs[best_inc]:.2f} pp)")
    
    vols = {
        "Rare Name Token": rare_stats[0],
        "Name + Context": ctx_stats[0],
        "Numeric Address Token": num_stats[0]
    }
    worst_vol = max(vols, key=vols.get)
    print(f"3. Block producing the largest candidate volume: {worst_vol} ({vols[worst_vol]} candidates)")
    
    print("4. Best practical configuration will be based on these numbers.")

if __name__ == "__main__":
    run_blocking_ablation()

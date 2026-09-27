import os
import psutil
import pandas as pd
from data_loading import load_data
from retrieval import normalize_dataframe, generate_exact_match_candidates
from ablation_blocking import get_rare_tokens_blocking, get_name_context_blocking, get_numeric_address_blocking

def mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print("Loading data...")
_, s1_val, s2, s3, _, gt_val = load_data(split_ratio=0.1)

print(f"Normalizing data... Mem: {mem():.1f} MB")
s1_val = normalize_dataframe(s1_val)
s2 = normalize_dataframe(s2)
s3 = normalize_dataframe(s3)
s23 = pd.concat([s2, s3], ignore_index=True)

print(f"Generating exact name... Mem: {mem():.1f} MB")
cand_name_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_name")
print(f"Name cands: {len(cand_name_val)}. Mem: {mem():.1f} MB")

cand_addr_val = generate_exact_match_candidates(s1_val, s2, s3, "norm_address")
print(f"Addr cands: {len(cand_addr_val)}. Mem: {mem():.1f} MB")

baseline_cands = pd.concat([cand_name_val, cand_addr_val], ignore_index=True).drop_duplicates()
print(f"Baseline cands: {len(baseline_cands)}. Mem: {mem():.1f} MB")

cand_rare = get_rare_tokens_blocking(s1_val, s23, top_n=2)
print(f"Rare cands: {len(cand_rare)}. Mem: {mem():.1f} MB")

cand_ctx = get_name_context_blocking(s1_val, s23)
print(f"Ctx cands: {len(cand_ctx)}. Mem: {mem():.1f} MB")

cand_num = get_numeric_address_blocking(s1_val, s23)
print(f"Num cands: {len(cand_num)}. Mem: {mem():.1f} MB")

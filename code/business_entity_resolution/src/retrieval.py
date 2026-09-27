import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find

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

def generate_exact_match_candidates(s1_df, s2_df, s3_df, match_column):
    candidates = []
    
    s1_valid = s1_df[s1_df[match_column] != ""]
    s2_valid = s2_df[s2_df[match_column] != ""]
    s3_valid = s3_df[s3_df[match_column] != ""]
    
    s1_s2 = pd.merge(s1_valid[['entity_id', match_column]], 
                     s2_valid[['entity_id', match_column]], 
                     on=match_column, 
                     how='inner', 
                     suffixes=('_s1', '_cand'))
    if not s1_s2.empty:
        candidates.append(s1_s2[['entity_id_s1', 'entity_id_cand']])
        
    s1_s3 = pd.merge(s1_valid[['entity_id', match_column]], 
                     s3_valid[['entity_id', match_column]], 
                     on=match_column, 
                     how='inner', 
                     suffixes=('_s1', '_cand'))
    if not s1_s3.empty:
        candidates.append(s1_s3[['entity_id_s1', 'entity_id_cand']])
        
    if candidates:
        cand_df = pd.concat(candidates, ignore_index=True).drop_duplicates()
        return cand_df
    return pd.DataFrame(columns=['entity_id_s1', 'entity_id_cand'])
    
def generate_tfidf_candidates(s1_df, s2_df, s3_df, match_column, threshold=0.8, n_gram_range=(2,3)):
    # Note: For massive datasets, doing full cross-similarity is memory-intensive.
    # In a full-scale pipeline, nmslib or faiss would be used. 
    # For this modular baseline, we demonstrate the component conceptually.
    print(f"TF-IDF blocked on {match_column} (skipped to prevent OOM in local execution)")
    return pd.DataFrame(columns=['entity_id_s1', 'entity_id_cand'])


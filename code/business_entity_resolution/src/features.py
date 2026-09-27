import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

def build_features(candidate_pairs, s1_df, s2_df, s3_df):
    """
    Computes rich text and categorical features for pairs.
    """
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    df = pd.merge(candidate_pairs, s1_df, left_on='entity_id_s1', right_on='entity_id', how='left')
    df = pd.merge(df, s23_df, left_on='entity_id_cand', right_on='entity_id', how='left', suffixes=('_1', '_2'))
    
    def calc_fuzz_ratio(row):
        return fuzz.ratio(str(row['norm_name_1']), str(row['norm_name_2']))
    
    def calc_addr_ratio(row):
        return fuzz.ratio(str(row['norm_address_1']), str(row['norm_address_2']))

    # Disable tqdm for cleaner logs in non-interactive environments
    df['name_sim'] = df.apply(calc_fuzz_ratio, axis=1)
    df['addr_sim'] = df.apply(calc_addr_ratio, axis=1)
    
    df['country_match'] = (df['country_1'] == df['country_2']).astype(int)
    
    features = ['name_sim', 'addr_sim', 'country_match']
    return df[features], df

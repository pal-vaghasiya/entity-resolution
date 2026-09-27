import pandas as pd
import os
import numpy as np
from config import DATA_DIR

def load_data(split_ratio=0.1, random_state=42):
    """
    Loads full dataset, splits S1 entities into train/val.
    Returns: s1_train, s1_val, s2, s3, gt_train, gt_val
    """
    print("Loading datasets...")
    s1 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), sep="\t")
    
    # Shuffle and split Source 1
    s1_shuffled = s1.sample(frac=1.0, random_state=random_state)
    val_size = int(len(s1_shuffled) * split_ratio)
    
    s1_val = s1_shuffled.iloc[:val_size]
    s1_train = s1_shuffled.iloc[val_size:]
    
    # Split Ground truth accordingly
    gt_val = gt[gt['source1_entity_id'].isin(s1_val['entity_id'])]
    gt_train = gt[gt['source1_entity_id'].isin(s1_train['entity_id'])]
    
    print(f"Data Split: Train S1 size={len(s1_train)}, Val S1 size={len(s1_val)}")
    
    return s1_train, s1_val, s2, s3, gt_train, gt_val

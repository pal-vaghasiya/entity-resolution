import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")

def analyze():
    print("Loading data...")
    train_s1 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), sep="\t")
    train_gt = pd.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), sep="\t")
    
    test_s1 = pd.read_csv(os.path.join(DATA_DIR, "test", "test_source1.tsv"), sep="\t")
    
    print("Train Source 1 size:", len(train_s1))
    print("Test Source 1 size:", len(test_s1))
    
    print("\nTrain S1 unique countries:")
    print(train_s1['country'].value_counts(dropna=False))
    
    print("\nTest S1 unique countries:")
    print(test_s1['country'].value_counts(dropna=False))
    
    # Check Ground Truth
    gt_matched_counts = train_gt['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != '' else 0)
    print("\nTrain Ground Truth Matched Counts Distribution:")
    print(gt_matched_counts.value_counts().sort_index())
    
    print("\nData Audit complete.")

if __name__ == "__main__":
    analyze()

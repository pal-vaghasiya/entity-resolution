import pandas as pd
import os

def audit_file(filepath):
    print(f"--- Auditing {filepath} ---")
    try:
        # Just read a few rows to get basic info quickly
        df_head = pd.read_csv(filepath, sep="\t", nrows=10000, keep_default_na=False)
        print("Columns:", df_head.columns.tolist())
        print("Number of rows in sample:", len(df_head))
        print("Head:")
        print(df_head.head(2))
        print("Missing values in sample:")
        df_head_na = pd.read_csv(filepath, sep="\t", nrows=10000)
        print(df_head_na.isnull().sum())
        print("\n")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == '__main__':
    base_dir = r"c:\Users\palva\OneDrive\Desktop\entity-resolution\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
    files = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]
    for f in files:
        audit_file(os.path.join(base_dir, f))

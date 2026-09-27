# Business Entity Resolution Pipeline

This directory contains the source code for the entity resolution challenge.

## Setup

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. The dataset should be located relative to this directory at:
`../../6ab10eb3b23ba_student_resource/student_resource/dataset`

## Running the Pipeline

To run the full end-to-end pipeline (Data Loading -> Blocking -> Feature Engineering -> Model Training -> Inference):
```bash
cd src
python main.py
```

## Output

The pipeline will generate two files in the `../../output/` directory:
1. `candidate_pairs.tsv` - The candidate set from the blocking stage.
2. `matching_results.tsv` - The final model predictions.

## Architecture
- **Normalization**: Lowercase, punctuation removal.
- **Blocking**: Exact Name and Exact Address matching.
- **Feature Engineering**: Fuzzy ratio on Name and Address, Country match.
- **Modeling**: LightGBM Classifier.

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity  
**Team Members:** Antigravity AI  
**Submission Date:** September 2026

---

## 1. Executive Summary
We implemented a two-stage Entity Resolution pipeline comprising a high-recall blocking stage and a high-precision tree-based matching model (LightGBM). By standardizing the text features and employing fuzzy similarity metrics alongside exact matching, we successfully bridged the gap between noisy source records and clean mappings.

---

## 2. Methodology

### 2.1 Problem Analysis
During our EDA, we observed heavy noise in the business addresses, including missing address components and differing punctuation. There was also variance in the business names (e.g. abbreviations). There were singletons that needed to be predicted as non-matches, heavily penalizing false merges.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier
**Core Innovation:** A robust text normalization strategy followed by exact block mapping on both name and address components. We combined this with Fuzzy string matching (Levenshtein) and trained a LightGBM classifier optimized for F_0.5 score.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Normalized Name, Normalized Address
- **Candidate pairs generated:** High recall candidate generation by performing inner joins on exact matches of normalized text.
- **How you ensured true matches were not lost:** We applied multiple blocking channels (name and address independently) and concatenated the candidates to maximize recall.

---

## 4. Matching Model

**Features used:**
- Name features: Fuzzy string ratio on normalized names.
- Address features: Fuzzy string ratio on normalized addresses.
- Other: Exact country match indicator.

**Model type:** LightGBM Classifier (n_estimators=100)
**Threshold selection method:** Selected threshold 0.5 optimized on the validation split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Tuned on cross-validation splits.
- **Common false positives (wrong merges):** Businesses sharing similar names but in different jurisdictions or representing different franchises.
- **Common false negatives (missed matches):** Heavy abbreviation differences and entirely missing address components preventing blocking.

---

## 6. Conclusion
The pipeline successfully resolves entities by prioritizing recall during candidate generation and precision during classification, aligning perfectly with the F_0.5 metric. The LightGBM model successfully captures the non-linear relationship between fuzzy ratios.

---

## Appendix

### A. Code Artefacts
The complete, runnable code ships in the submission zip under `code/business_entity_resolution/`.
- Entry point: `src/main.py`
- Requirements: `requirements.txt`
- Output: The pipeline generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end when executed.

# MASTER IMPLEMENTATION PROMPT
## Amazon Entity Resolution / Business Matching Challenge

You are an expert ML engineer, data scientist, and entity-resolution researcher.

I will provide you with:

1. The **official Problem Statement (PS)**.
2. The **actual competition/project dataset**.

Your task is to implement the complete entity-resolution system **from scratch**, following the official PS exactly wherever it specifies requirements, formats, constraints, outputs, or evaluation rules.

The goal is to build a **high-quality, reproducible, scalable, leakage-safe entity-resolution pipeline** that matches records from Source 1 to the correct records in Source 2 and/or Source 3.

---

# 0. FIRST PRINCIPLE — READ THE PROBLEM STATEMENT COMPLETELY

Before writing implementation code:

1. Read the entire Problem Statement.
2. Understand every requirement, including:
   - input files
   - output files
   - required columns
   - ID semantics
   - train/test structure
   - evaluation metric
   - candidate-pair requirements
   - matching-result requirements
   - restrictions
   - documentation requirements
   - any special cases mentioned in the PS.
3. Do not begin implementation until you have extracted the complete technical requirements.
4. Treat the official PS as the highest-priority specification.

Do not silently reinterpret the PS.

If there is any ambiguity between general entity-resolution practice and the PS, follow the PS.

If the PS explicitly requires something that is unusual compared with standard ML practice, implement the PS requirement rather than replacing it with your preferred approach.

---

# 1. REQUIRED OVERALL PIPELINE ARCHITECTURE

The following is the **required baseline architecture and stage ordering**.

The architecture is intentionally structured and should NOT be continuously redesigned based on the dataset.

Dataset inspection is allowed to tune:

- thresholds
- top-K values
- model hyperparameters
- retrieval parameters
- feature weights
- candidate limits
- pruning thresholds
- validation strategy details
- narrowly defined implementation choices.

However, dataset inspection must NOT arbitrarily reorder, remove, bypass, or redefine the core stages below.

If the actual dataset provides strong empirical evidence that a narrowly defined architectural substitution is necessary, make the smallest possible change, validate it using training/validation data, and document the reason.

Do not redesign the entire pipeline simply because another architecture looks interesting.

The required baseline flow is:

```text
RAW DATA
   ↓
DATA AUDIT / PROFILING
   ↓
NORMALIZATION + MULTIPLE REPRESENTATIONS
   ↓
RETRIEVAL / BLOCKING
   ├── Exact normalized-name retrieval
   ├── Exact normalized-address retrieval
   ├── Character TF-IDF retrieval
   ├── Word TF-IDF retrieval
   └── Multilingual embedding retrieval
   ↓
CANDIDATE UNION
   ↓
CHEAP DETERMINISTIC FILTERING
   ↓
STAGE-1 CANDIDATE PRUNING
   └── LightGBM / GBDT
   ↓
CANDIDATE-GENERATION POSTPROCESSING
   ↓
FINAL CANDIDATE SET
   ↓
candidate_pairs.tsv
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FREEZE CANDIDATE SET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ↓
RICH FINAL PAIR FEATURES
   ↓
FINAL MATCHING MODEL(S)
   ├── LightGBM
   ├── CatBoost
   └── Transformer / Cross-Encoder if justified
   ↓
MODEL SCORES
   ↓
ENSEMBLING / CALIBRATION IF VALIDATED
   ↓
ENTITY-LEVEL MATCH DECISION
   ├── ZERO MATCHES
   ├── ONE MATCH
   └── MULTIPLE MATCHES
   ↓
FINAL MATCH POSTPROCESSING
   ↓
matching_results.tsv
```

This stage ordering is extremely important.

---

# 2. CRITICAL REQUIREMENT: SEMANTICS OF `candidate_pairs.tsv`

## THIS MUST NOT BE MISINTERPRETED

`candidate_pairs.tsv` is **NOT** the raw output of the first blocking or retrieval stage.

It is the **FINAL candidate set that will actually be given to the final matching model for inference**.

Therefore, all candidate-generation operations must be completed BEFORE `candidate_pairs.tsv` is created.

This includes:

1. all retrieval methods,
2. all blocking methods,
3. candidate union,
4. cheap deterministic filtering,
5. Stage-1 LightGBM/GBDT candidate pruning,
6. candidate deduplication,
7. validated candidate-count restrictions,
8. candidate-generation cleanup,
9. any other operation whose purpose is to decide whether a pair reaches the final model.

Only after ALL candidate-generation operations are complete should the system write:

```text
candidate_pairs.tsv
```

Then the candidate set becomes **frozen**.

The final matching model must score **exactly the pairs present in `candidate_pairs.tsv`**.

No new candidate pairs may be added after this point.

No additional retrieval may occur after this point.

No hidden second blocking stage may occur after this point.

No final-model step may silently retrieve additional candidates.

---

## Required invariant

The system must satisfy:

```text
matched pairs in matching_results.tsv
        ⊆
pairs in candidate_pairs.tsv
```

If a pair appears in `matching_results.tsv`, that exact pair must already exist in `candidate_pairs.tsv`.

If a predicted match does not exist in `candidate_pairs.tsv`, that is a pipeline bug.

---

## Correct implementation semantics

Use a structure conceptually equivalent to:

```python
candidate_pairs = generate_retrieval_candidates(...)

candidate_pairs = union_candidates(candidate_pairs)

candidate_pairs = apply_cheap_filters(candidate_pairs)

candidate_pairs = stage1_candidate_pruning(candidate_pairs)

candidate_pairs = final_candidate_generation_postprocessing(
    candidate_pairs
)

# This is now the FINAL candidate set.
write_candidate_pairs(candidate_pairs)

# Freeze candidate_pairs from this point onward.

final_features = build_rich_pair_features(candidate_pairs)

scores = final_matching_model.predict(final_features)

matches = entity_level_decision(scores)

matches = final_match_postprocessing(matches)

write_matching_results(matches)
```

Do not move the writing of `candidate_pairs.tsv` earlier in this process.

---

# 3. DATASET AUDIT — INSPECT THE ACTUAL DATA FIRST

After reading the PS, inspect the actual dataset thoroughly.

Do not blindly assume the dataset has the characteristics you expect.

Create a data audit covering:

### File-level analysis

For every input file determine:

- filename
- number of rows
- number of columns
- column names
- data types
- delimiter
- encoding
- missing values
- duplicate rows
- duplicate IDs
- unique ID counts
- source-specific ID patterns.

### Entity-level analysis

Analyze:

- Source-1 entity count
- Source-2 entity count
- training pair count
- test Source-1 count
- match cardinality
- number of Source-1 records matching zero Source-2 records
- one-to-one behavior
- one-to-many behavior
- many-to-one behavior if applicable.

### Field-level analysis

For every relevant field:

- missingness
- unique values
- length distribution
- token counts
- character distributions
- language/script patterns
- numeric content
- punctuation
- abbreviations
- formatting inconsistencies.

Analyze fields such as:

- business/company name
- address
- city
- state/region
- postal code
- country
- phone
- website/domain
- category
- any other fields specified by the PS.

Do not assume all fields exist.

Adapt to the actual schema while preserving the required architecture.

---

# 4. COUNTRY DISTRIBUTION — IMPORTANT FRANCE CASE

Explicitly inspect country distributions in:

- training
- validation
- test
- Source 1
- Source 2.

The dataset contains an important distribution issue:

> **France appears in the test data but is absent from the training data.**

Treat this as an open-set/generalization issue.

Do NOT:

- hard-code a list of training countries,
- assume all test countries appear in training,
- discard France,
- exclude France from inference,
- create a France-specific test rule,
- manually assign France-specific matches,
- use external data to solve France.

Instead:

- represent country as an open-set feature,
- support unseen categorical values safely,
- use country compatibility as a feature,
- preserve country information during normalization,
- allow exact country agreement to contribute to matching,
- do not make country an absolute hard constraint unless training/validation evidence strongly supports such a rule.

The system must generalize to unseen countries naturally.

France must be handled by the same general pipeline.

---

# 5. NORMALIZATION

Build robust but conservative normalization.

Maintain the original raw fields.

Do NOT overwrite the original data.

For important text fields create:

- raw representation
- normalized representation
- token representation
- character representation where useful.

Normalization may include:

- Unicode normalization
- lowercasing
- whitespace normalization
- punctuation normalization
- controlled removal of formatting noise
- standardization of obvious abbreviations
- normalization of repeated spaces
- normalization of numeric formatting
- consistent handling of accents where justified.

Be conservative.

Do not remove information simply because it looks inconvenient.

For example:

- apartment numbers
- building numbers
- postal codes
- suite numbers
- unit numbers
- business suffixes
- meaningful numeric tokens

may contain important entity-resolution information.

If abbreviation mappings are introduced, document them.

---

# 6. RETRIEVAL / BLOCKING

Implement multiple complementary candidate-generation channels.

Every retrieval channel below (6.1–6.5) must run against **both Source 2 and Source 3 independently** — they are separate files with separate ID spaces (`S2-` / `S3-`), and a Source 1 entity may match records from either or both. Do not treat Source 3 as an optional or secondary target: it is a mandatory, equally-weighted retrieval target from the start, not something to add only if validation later shows it necessary.

At minimum, evaluate:

## 6.1 Exact normalized name

Retrieve Source-2 and Source-3 candidates sharing the same normalized name.

Track:

- candidate count
- retrieval success
- ambiguity
- candidate rank.

---

## 6.2 Exact normalized address

Retrieve candidates using normalized address.

Do not require the entire address to match if the dataset shows formatting variation.

---

## 6.3 Character TF-IDF

Use character n-gram TF-IDF retrieval.

Inspect the actual data to choose suitable:

- n-gram range
- analyzer
- minimum document frequency
- top-K
- similarity threshold.

Do not blindly use one fixed value.

However, do not endlessly search over hyperparameters.

Use a compact validation-driven search.

---

## 6.4 Word TF-IDF

Use word-level TF-IDF retrieval where useful.

Evaluate:

- word n-grams
- stopword handling
- minimum frequency
- top-K.

Again, use the dataset and validation results to choose reasonable parameters.

---

## 6.5 Multilingual embeddings

Use multilingual embeddings where computationally feasible and justified by the actual data.

Embeddings should help with:

- spelling variation
- transliteration
- multilingual names
- semantic similarity
- address variation.

Do not assume embeddings automatically outperform lexical retrieval.

Measure their actual contribution to candidate recall and downstream performance.

---

# 7. RETRIEVAL PROVENANCE

For every candidate pair retain metadata showing how it was retrieved.

For example:

```text
exact_name_match
exact_address_match
char_tfidf_match
word_tfidf_match
embedding_match
```

Also retain where useful:

- retrieval rank
- retrieval similarity
- retrieval channel
- number of channels retrieving the pair
- best retrieval score
- per-channel scores.

These are valuable downstream features.

Do not throw away retrieval provenance after generating candidates.

---

# 8. CANDIDATE RECALL MUST BE MEASURED

Using training data with known ground truth, measure whether the true Source-2 match appears in the candidate set.

Measure recall after each major candidate-generation stage:

```text
Retrieval recall
        ↓
Recall after candidate union
        ↓
Recall after cheap filtering
        ↓
Recall after Stage-1 pruning
        ↓
FINAL CANDIDATE RECALL
```

The goal of candidate generation is **high recall**.

A candidate-pruning stage that produces a smaller candidate set but removes many true matches is unacceptable.

Do not optimize candidate count without monitoring recall.

---

## 8.1 CANDIDATE-SET SIZE IS ALSO A SCORED RANKING CRITERION

Recall is the primary requirement, but it is not the only objective for `candidate_pairs.tsv`.

The PS states explicitly:

> Candidate generation counts toward the final ranking. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation, beyond the public/private leaderboard.

Therefore, once a candidate-generation configuration has been validated to preserve the required recall level, treat **average / median candidates per Source-1 entity** as a genuine secondary optimization target — not merely a monitoring statistic.

Concretely:

- Among configurations with statistically equivalent validation recall, prefer the one with the smaller mean/median candidate count per Source-1 entity.
- Do not stop at "recall is fine, ship it" — after recall is validated as sufficient, run a compact, validation-driven search (top-K, thresholds, retrieval-channel pruning) to tighten the candidate set further, exactly as you would tune any other retrieval parameter (see §1).
- Never trade meaningful recall for a smaller candidate set — recall remains the hard constraint (§8's rule still applies) — but do not leave an unnecessarily loose candidate set on the table once recall is satisfied, since it will be penalized independently of your F0.5 leaderboard score.
- Record this trade-off explicitly in the candidate-generation report and decision log (§33): recall at each candidate-set size considered, and why the final size was chosen.

Do not confuse this with the Stage-1 pruning warning above: that warning says "don't shrink the set at the cost of recall." This section says "once recall is satisfied, shrinking further is rewarded, so keep trying."

---

# 9. CANDIDATE UNION

Combine candidates from all retrieval channels.

Use pair identity such as:

```text
(source1_id, source2_id)
```

to deduplicate candidates.

Preserve all relevant retrieval metadata.

If multiple retrieval methods independently retrieve the same pair, record that agreement.

Multi-channel agreement should become a useful feature.

---

# 10. CHEAP DETERMINISTIC FILTERING

After candidate union, apply inexpensive filters where justified.

Examples may include:

- impossible ID combinations
- obviously incompatible values
- invalid records
- extreme contradictions
- impossible numeric patterns
- clearly incompatible country information if validated.

Do NOT introduce aggressive hard rules without validation.

Every deterministic rule must be evaluated for:

- candidate recall
- candidate reduction
- false-negative risk.

The default should be conservative.

---

# 11. STAGE-1 LIGHTGBM / GBDT CANDIDATE PRUNING

Use a first-stage LightGBM/GBDT model to reduce the candidate set.

This is a **candidate-pruning model**, not necessarily the final matching model.

Use relatively cheap features such as:

- exact-name indicators
- exact-address indicators
- character similarity
- token overlap
- length differences
- retrieval similarity
- retrieval rank
- retrieval-channel agreement
- country compatibility
- numeric agreement
- cheap address similarity.

The Stage-1 model should prioritize candidate recall.

Its threshold must be selected using training/validation evidence.

Do not tune this threshold using hidden test labels.

---

# 12. CANDIDATE-GENERATION POSTPROCESSING

After Stage-1 LightGBM pruning, complete all remaining candidate-generation operations.

This may include:

- deduplication
- removing invalid pairs
- enforcing validated candidate-count limits
- retaining necessary high-confidence retrieval candidates if justified
- handling edge cases
- ensuring every relevant Source-1 record is represented
- final candidate-set consistency checks.

This is still part of **candidate generation**.

Therefore it occurs BEFORE `candidate_pairs.tsv`.

---

# 13. WRITE `candidate_pairs.tsv` ONLY NOW

Only now create:

```text
candidate_pairs.tsv
```

At this point:

> `candidate_pairs.tsv` = the exact set of pairs that will be passed to the final matching model.

Freeze it.

Do not change it later.

Do not add candidates later.

Do not remove candidates later unless the PS explicitly requires such a modification to the candidate file itself.

The final matching stage must operate on this exact candidate set.

---

# 14. RICH PAIR FEATURE ENGINEERING — AFTER FREEZING CANDIDATES

Now compute richer pairwise features.

Possible features include:

## String similarity

- Levenshtein similarity
- normalized edit distance
- Jaro
- Jaro-Winkler
- longest common subsequence
- character overlap
- token overlap
- token Jaccard.

## TF-IDF similarity

- character TF-IDF cosine
- word TF-IDF cosine.

## Embedding similarity

- multilingual embedding cosine similarity
- field-specific embedding similarity
- combined representations where validated.

## Address features

Break addresses into components when possible:

- street
- building number
- apartment/unit
- postal code
- city
- state/region
- country.

Create component-level comparisons.

## Numeric features

Compare:

- building numbers
- postal codes
- phone digits
- unit numbers
- other meaningful numeric tokens.

## Retrieval features

Include:

- retrieval rank
- retrieval similarity
- best channel
- number of agreeing retrieval channels
- exact-match indicators.

## Ambiguity features

Calculate:

- number of candidates for Source-1
- candidate rank
- difference between top candidate and second candidate
- local candidate density
- frequency of the normalized name
- frequency of the normalized address.

Do not use any feature that leaks the target label.

---

# 15. HARD NEGATIVE GENERATION

Training only on random negatives is insufficient.

Construct hard negatives from realistic candidate pairs, such as:

- same/similar business names
- same city
- same address components
- same postal code
- similar names with different businesses
- candidates retrieved by embeddings but incorrect
- candidates retrieved by TF-IDF but incorrect
- candidates surviving Stage-1 pruning but ultimately incorrect.

The purpose is to teach the model to distinguish plausible but incorrect entities.

Do not use test labels.

---

# 16. TRAIN/VALIDATION SPLITTING

Avoid leakage.

Do not randomly split pair rows if the same Source-1 entities can appear across train and validation.

Prefer entity-level splitting.

For example:

```text
Source-1 entities
        ↓
train entities
validation entities
```

Then construct training/validation candidate pairs accordingly.

Ensure that:

- the same Source-1 entity does not leak across splits,
- target labels do not leak,
- retrieval statistics do not use validation labels,
- target-derived statistics are computed only within the appropriate training partition.

---

# 17. LEAKAGE CONTROL

Audit every feature and preprocessing step for leakage.

Do not use:

- test labels
- hidden ground truth
- future target information
- manually inferred test matches
- target-derived statistics calculated using validation/test entities improperly.

Retrieval itself must not use target labels.

Validation must simulate the real inference setting as closely as possible.

---

# 18. FINAL MATCHING MODELS

After `candidate_pairs.tsv` has been frozen, train/evaluate final matching models.

Evaluate candidates such as:

### Model A — LightGBM

Use the rich pairwise features.

### Model B — CatBoost

Useful for nonlinear interactions and heterogeneous feature types.

### Model C — Transformer / Cross-Encoder

Use a multilingual transformer or DeBERTa-style cross-encoder if:

- allowed by the PS,
- computationally feasible,
- licensing is appropriate,
- validation indicates meaningful benefit.

**License/size constraint:** the PS requires the final model to be MIT/Apache 2.0 licensed and up to 8 billion parameters. The PS wording does not make explicit whether this constraint applies only to the final matching model or to every model in the pipeline. To stay safely compliant, apply the MIT/Apache-2.0-and-≤8B-parameters constraint to **every** model used anywhere in the pipeline — including any multilingual embedding model used in retrieval (§6.5) — not just the final matching model. Verify and document the license and parameter count of every model choice before adopting it.

Do not automatically assume a transformer is superior.

---

# 19. MODEL COMPARISON

Compare models using validation data.

Report:

- precision
- recall
- F0.5
- candidate recall
- false positives
- false negatives
- inference cost
- training cost.

Do not select a model merely because it is more sophisticated.

The selection must be based on validation evidence and the official evaluation objective.

---

# 20. ENSEMBLING

If multiple models provide complementary information, evaluate an ensemble.

Possible approaches:

- weighted probability averaging
- rank averaging
- calibrated score averaging
- stacking if properly validated.

Weights must be determined using training/validation data.

Do not tune ensemble weights on hidden test labels.

Do not add an ensemble merely because it sounds sophisticated.

Use it only if it improves validation performance robustly.

---

# 21. SCORE CALIBRATION

Calibration is optional.

Consider:

- Platt scaling
- isotonic regression
- other appropriate calibration methods.

Only use calibration if it improves decision quality on validation.

Do not automatically calibrate.

---

# 22. ENTITY-LEVEL MATCH DECISION

The final system must make decisions at the Source-1 entity level.

For each Source-1 record:

1. collect its candidate pairs,
2. obtain final model scores,
3. rank candidates,
4. apply the validated threshold,
5. determine whether the entity has:
   - zero matches,
   - one match,
   - multiple matches.

Do not force every Source-1 record to have exactly one match unless the PS explicitly requires that behavior.

The actual training data should be inspected to understand the expected cardinality.

---

# 23. THRESHOLD OPTIMIZATION

Optimize the final decision threshold on validation data.

The primary metric is:

```text
F0.5
```

with:

```text
F0.5 =
(1.25 × Precision × Recall)
/
(0.25 × Precision + Recall)
```

Because F0.5 weights precision more heavily than recall, evaluate the precision/recall tradeoff carefully.

Do not simply maximize accuracy.

Do not choose the threshold from test predictions.

---

# 24. MARGIN / AMBIGUITY RULES

Where useful, evaluate rules such as:

```text
top_score > threshold
AND
top_score - second_score > margin
```

These rules may help when multiple Source-2 candidates are highly similar.

However:

- do not assume margin rules are always beneficial,
- validate them,
- tune margins only on training/validation data,
- do not create arbitrary special cases.

---

# 25. S2 / S3 / SOURCE-SPECIFIC BEHAVIOR

Note: this section is about whether to build *separate model logic* per source — it does NOT relax the requirement in §6 that every retrieval channel must run against both Source 2 and Source 3 from the start. Retrieving against both sources is mandatory from day one; a unified vs. source-specific *model/pipeline* is the open question below.

If the PS defines multiple source types or subsets such as S2/S3, initially use a unified architecture.

Only introduce source-specific model behavior if validation demonstrates that it is necessary.

Do not create separate pipelines simply because the sources look different.

If source-specific handling is introduced:

1. document why,
2. validate it,
3. compare against the unified baseline,
4. keep the change minimal.

---

# 26. FRANCE / UNSEEN-COUNTRY ROBUSTNESS

Again, the test contains France while training does not.

The pipeline must therefore:

- support unseen countries,
- avoid categorical encoding failures,
- avoid training-country hard-coding,
- avoid dropping France,
- avoid France-specific rules,
- use country information as a generalizable feature,
- preserve multilingual text handling.

If the implementation uses categorical encoders, ensure unseen categories are handled safely.

If using embeddings, ensure multilingual support is appropriate.

---

# 27. NO EXTERNAL BUSINESS LOOKUP

Do NOT use external business/entity information to solve the matching problem.

Do not use:

- Google Search
- Google Maps
- business directories
- company registries
- external entity databases
- geocoding services
- external APIs
- manually searched company websites
- manually verified addresses
- external datasets containing business identities.

The system must solve the task using the provided data and permitted methods.

---

# 28. NO TEST OVERFITTING

You may inspect the test dataset for:

- schema
- row counts
- missingness
- country distribution
- field distributions
- text characteristics
- computational requirements.

You may NOT use test data to derive hidden matching decisions.

Never create rules such as:

```text
if test_id == ...
```

or:

```text
if business_name == ...
```

for individual test records.

Never manually inspect a test company and determine its correct Source-2 match.

Never create country-specific test hacks.

Never tune thresholds directly against hidden test performance.

The test set is for final inference.

---

# 29. CONTROLLED DATASET-DRIVEN ADAPTATION

The dataset should influence parameter choices, but not cause uncontrolled architecture drift.

You ARE expected to inspect the actual dataset and determine reasonable values for:

- character n-gram ranges
- word n-gram ranges
- retrieval top-K
- embedding top-K
- similarity thresholds
- candidate limits
- LightGBM pruning threshold
- final matching threshold
- margin threshold
- model hyperparameters.

Use validation evidence.

You are NOT expected to blindly use textbook defaults.

However, do not repeatedly redesign the entire architecture after every observation.

Use the following principle:

> **Fixed architecture, adaptive parameters.**

If a major architectural change is genuinely necessary:

1. establish a baseline,
2. identify the dataset evidence,
3. propose the smallest change,
4. validate it,
5. compare it against the baseline,
6. document the decision.

---

# 30. COMPUTATIONAL EFFICIENCY

The dataset may be large.

Do NOT construct a full Cartesian product:

```text
Source1 × (Source2 ∪ Source3)
```

unless the actual dataset is demonstrably tiny and the PS permits it.

Prefer:

- inverted indexes
- sparse TF-IDF matrices
- ANN retrieval
- top-K retrieval
- vectorized operations
- batching
- caching
- precomputed normalized fields
- reusable embeddings
- efficient joins
- parallel processing where safe.

The expensive models should operate only on the final candidate set.

This is another reason why `candidate_pairs.tsv` must be the final candidate set before final inference.

---

# 31. REPRODUCIBILITY

Make the entire pipeline reproducible.

Set seeds where applicable.

Create configuration files for:

- random seed
- retrieval parameters
- top-K
- similarity thresholds
- model hyperparameters
- candidate pruning thresholds
- final threshold
- margin threshold.

Do not scatter magic numbers throughout the code.

Every important parameter should be configurable.

---

# 32. PROJECT STRUCTURE

Build directly into the exact folder layout required by the PS for the final submission zip. Do not use a generic structure that will need to be reorganized later — construct the project so it already matches what gets zipped as `<team_name>_submission.zip`:

```text
project/
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   ├── preprocessing/
│       │   ├── blocking/
│       │   ├── retrieval/
│       │   ├── candidate_pruning/
│       │   ├── features/
│       │   ├── models/
│       │   ├── inference/
│       │   ├── evaluation/
│       │   └── validation/
│       │
│       ├── data/
│       │   ├── raw/
│       │   ├── processed/
│       │   └── splits/
│       │
│       ├── configs/
│       ├── models/
│       │
│       ├── reports/
│       │   ├── data_profile.md
│       │   ├── candidate_generation_report.md
│       │   ├── validation_report.md
│       │   └── decision_log.md
│       │
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

Keep components modular.

Do not create one enormous Python script containing the entire pipeline.

At final packaging time, `project/` should zip up directly as `<team_name>_submission.zip` with no restructuring needed — `output/` and `Documentation_template.md` at the top level, and the runnable pipeline under `code/business_entity_resolution/`.

---

# 33. REQUIRED REPORTS

Generate a data profiling report containing:

- dataset sizes
- schemas
- missingness
- duplicates
- country distribution
- text statistics
- training match statistics.

Generate a candidate-generation report containing:

- retrieval methods
- top-K values
- candidate counts
- recall after each stage
- pruning results
- final candidate count
- average candidates per Source-1 entity.

Generate a validation report containing:

- split strategy
- model performance
- precision
- recall
- F0.5
- threshold analysis
- model comparison
- ensemble comparison if applicable.

Generate a decision log containing:

- important design decisions
- dataset evidence
- alternatives considered
- validation results
- final choice.

---

# 34. LOG EVERY IMPORTANT STAGE

During execution, print/log metrics such as:

```text
Source-1 rows:
Source-2 rows:

Initial retrieval candidates:
Candidates after union:
Candidates after cheap filtering:
Candidates after Stage-1 pruning:
Final candidate pairs:

Candidate recall after retrieval:
Candidate recall after filtering:
Candidate recall after Stage-1:
Final candidate recall:

Average candidates per S1:
Median candidates per S1:
Maximum candidates per S1:
S1 entities with zero candidates:
```

Then after final matching:

```text
Validation precision:
Validation recall:
Validation F0.5:
Selected threshold:
Selected margin:
```

---

# 35. FINAL FILE VALIDATION

Before declaring the pipeline complete, validate:

## `candidate_pairs.tsv`

Check:

- correct delimiter
- correct columns
- valid Source-1 IDs
- valid Source-2 IDs
- no invalid pairs
- no unintended duplicates
- every required Source-1 entity handled correctly
- candidate set corresponds exactly to final model input.

## `matching_results.tsv`

Check:

- required schema
- valid IDs
- no invalid Source-2 IDs
- every matched pair exists in `candidate_pairs.tsv`
- no duplicate predictions unless explicitly allowed
- correct zero/one/many representation
- correct delimiter
- correct ordering if the PS specifies one.

---

# 36. CRITICAL FINAL INVARIANTS

Before final submission, verify all of the following:

### Invariant 1

```text
candidate_pairs.tsv
=
the final candidate set before final model inference
```

### Invariant 2

```text
final model input
=
candidate_pairs.tsv
```

### Invariant 3

```text
matching_results.tsv matched pairs
⊆
candidate_pairs.tsv
```

### Invariant 4

No candidate-generation stage occurs after `candidate_pairs.tsv`.

### Invariant 5

No new candidate pair is created during final model inference.

### Invariant 6

No test labels are used.

### Invariant 7

No external business lookup is used.

### Invariant 8

France/unseen countries are supported without special-case test hacks.

### Invariant 9

All important thresholds are determined from training/validation data.

### Invariant 10

The pipeline is reproducible.

### Invariant 11

Every model used anywhere in the pipeline (final matching model, Stage-1 GBDT, any transformer/cross-encoder, any multilingual embedding model) is MIT/Apache 2.0 licensed and has at most 8 billion parameters.

---

# 37. OFFICIAL VALIDATOR

The PS provides an official validator at `utils/validate_submission.py` (stdlib only, no dependencies). Run it from the `student_resource/` directory using the exact invocation given in the PS:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

1. run it,
2. fix all format violations it reports,
3. re-run until it prints `PASS` (exit code 0),
4. do this before every leaderboard upload, not just once at the end.

Note that this validator only checks file format/structure against the rules in the PS — it does not compute your F0.5 score. Do not assume that a file is correct merely because it looks correct manually.

---

# 38. IMPLEMENTATION ORDER

Work in this order:

### Phase 1 — Understand

1. Read PS completely.
2. Extract requirements.
3. Inspect dataset.
4. Produce data profile.

### Phase 2 — Build representations

5. Build normalization.
6. Build text representations.
7. Build country/address/numeric representations.

### Phase 3 — Candidate generation

8. Implement exact retrieval.
9. Implement character TF-IDF retrieval.
10. Implement word TF-IDF retrieval.
11. Implement multilingual embedding retrieval.
12. Combine candidates.
13. Measure candidate recall.
14. Implement cheap filtering.
15. Implement Stage-1 LightGBM.
16. Measure candidate recall again.
17. Perform final candidate-generation postprocessing.
18. Write and freeze `candidate_pairs.tsv`.

### Phase 4 — Final matching

19. Build rich pair features.
20. Generate hard negatives.
21. Train final LightGBM.
22. Train/evaluate CatBoost.
23. Evaluate transformer/cross-encoder if justified.
24. Compare models.
25. Evaluate ensemble if useful.
26. Tune final threshold.
27. Tune margin if useful.
28. Implement entity-level decision.

### Phase 5 — Finalization

29. Run final inference.
30. Run final-match postprocessing.
31. Generate `matching_results.tsv`.
32. Validate output files.
33. Run official validator.
34. Generate reports.
35. Document final methodology.
36. Provide reproducibility instructions.

---

# 39. IMPORTANT DISTINCTION BETWEEN THE TWO POSTPROCESSING STAGES

Never confuse these.

## Candidate-generation postprocessing

Occurs:

```text
BEFORE candidate_pairs.tsv
```

Purpose:

> Decide exactly which pairs are allowed to reach the final model.

It may modify the candidate set.

---

## Final-match postprocessing

Occurs:

```text
AFTER final model scoring
```

Purpose:

> Convert model scores into final entity-level predictions and enforce output consistency.

It must NOT create new candidate pairs.

This distinction must be preserved in both the code and documentation.

---

# 40. DO NOT OVERFIT TO THE DATASET

The actual dataset should inform the system, but do not build a pipeline that only works because of accidental characteristics of this particular dataset.

Avoid:

- test-ID rules
- hard-coded company names
- hard-coded country hacks
- hard-coded expected match IDs
- manually selected examples used as rules
- arbitrary thresholds chosen after inspecting test outputs
- excessive hyperparameter search
- memorization of training entities.

Prefer generalizable mechanisms:

- normalized similarity
- retrieval
- learned pairwise features
- multilingual representations
- validation-based thresholds
- robust candidate generation.

---

# 41. USE FOURSQUARE SOLUTIONS AS DESIGN INSPIRATION, NOT AS BLIND CODE

If reference solutions from the Foursquare/entity-resolution context are available, study their useful ideas, especially the strongest approaches and the 1st/9th-ranked approaches discussed in the project context.

Extract ideas such as:

- efficient blocking
- candidate generation
- lexical retrieval
- embedding retrieval
- hard negatives
- pairwise modeling
- ensemble strategies
- validation methodology.

However:

Do not blindly copy their pipeline.

Adapt only ideas that are compatible with:

1. the official PS,
2. the actual Amazon dataset,
3. the required output semantics,
4. computational constraints.

Most importantly, preserve the required meaning of `candidate_pairs.tsv`.

---

# 42. BASELINE-FIRST RULE

Before making sophisticated improvements:

1. build the complete baseline architecture,
2. validate it,
3. record metrics,
4. then introduce improvements one at a time.

Every major improvement should have:

```text
baseline result
        ↓
new method
        ↓
validation result
        ↓
decision
```

Do not stack ten unvalidated improvements simultaneously.

This makes it possible to identify what actually helps.

---

# 43. FAILURE ANALYSIS

After validation, inspect failure cases.

Categorize errors such as:

- spelling variation
- abbreviations
- multilingual names
- address formatting
- missing fields
- duplicate businesses
- same-name businesses
- same-address businesses
- country mismatch
- postal-code mismatch
- embedding false positives
- TF-IDF false positives
- retrieval misses
- Stage-1 pruning false negatives
- final-model false positives.

Use these observations to make narrowly targeted improvements.

Do not respond to every failure with a new hard-coded rule.

---

# 44. FINAL DELIVERABLES

At the end, the project must contain:

1. Complete source code.
2. Reproducible configuration.
3. Trained model artifacts where appropriate.
4. `candidate_pairs.tsv`.
5. `matching_results.tsv`.
6. Data profiling report.
7. Candidate-generation report.
8. Validation report.
9. Decision log.
10. README with exact execution instructions.
11. Requirements/dependency file.
12. Documentation explaining the complete methodology.

---

# 45. FINAL RESPONSE FROM YOU AFTER IMPLEMENTATION

When implementation is complete, report:

### Dataset

- sizes
- key characteristics
- country distribution
- important anomalies.

### Candidate generation

- retrieval methods
- top-K values
- candidate counts
- candidate recall at every stage
- Stage-1 pruning statistics
- final candidate count
- average and median candidates per Source-1 entity, and how this was traded off against recall (this is a scored ranking criterion in the PS, independent of leaderboard F0.5 — see §8.1).

### Final matching

- models evaluated
- validation precision
- validation recall
- validation F0.5
- threshold
- margin
- ensemble details if used.

### Generalization

Explicitly explain how the pipeline handles:

- France appearing in test but not training
- unseen countries
- multilingual data
- missing fields
- ambiguous entities.

### Files

Confirm:

```text
candidate_pairs.tsv
matching_results.tsv
```

and explicitly confirm:

> `candidate_pairs.tsv` is the final frozen candidate set immediately before final model inference.

Also confirm:

> Every matched pair in `matching_results.tsv` exists in `candidate_pairs.tsv`.

---

# 46. FINAL PRIORITY ORDER

When making implementation decisions, use this priority order:

```text
1. Official Problem Statement compliance
2. Correctness
3. Candidate recall
4. Validation F0.5 / matching quality
5. Leakage prevention
6. Generalization
7. Candidate-set compactness per Source-1 entity (once recall is satisfied)
8. Scalability
9. Reproducibility
10. Computational efficiency
11. Architectural sophistication
```

Do not sacrifice correctness or PS compliance for sophistication.

Do not sacrifice candidate recall merely to reduce the candidate set — recall (priority 3) always outranks compactness (priority 7). Compactness is a tie-breaker among configurations that already meet the validated recall bar, not a license to under-retrieve.

Do not sacrifice generalization for test-specific improvements.

Do not redesign the architecture unnecessarily.

---

# 47. MOST IMPORTANT SUMMARY

The entire system must follow this fundamental separation:

```text
                 CANDIDATE GENERATION
                         │
                         ▼
Retrieval → Union → Filtering → Stage-1 LightGBM
                         │
                         ▼
             Candidate-generation postprocessing
                         │
                         ▼
                candidate_pairs.tsv
                         │
                         │
                    FROZEN HERE
                         │
                         ▼
                 FINAL MATCHING
                         │
                         ▼
        Rich features → Final model(s)
                         │
                         ▼
                    Scores
                         │
                         ▼
              Entity-level decision
                         │
                         ▼
             Final-match postprocessing
                         │
                         ▼
              matching_results.tsv
```

**`candidate_pairs.tsv` is the boundary between candidate generation and final matching.**

Everything that determines **which pairs the final model is allowed to consider** must happen before `candidate_pairs.tsv`.

Everything that determines **which of those candidate pairs become final matches** happens after `candidate_pairs.tsv`.

Never violate this boundary.

Now begin by reading the attached Problem Statement completely and inspecting the actual dataset before implementing anything.
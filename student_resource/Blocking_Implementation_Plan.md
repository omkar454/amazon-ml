# Step 2: Candidate Generation / Blocking Engine — Implementation Plan

**Amazon ML Challenge 2026** — *Business Entity Resolution Challenge*  
**Module:** High-Recall Scalable Blocking Engine (`src/blocking.py`, `src/tune_blocking.py`, `src/generate_candidates.py`)  
**Target:** Maximize Candidate Recall ($\ge 95\%+$), Minimize Search Space, Output Validated `output/candidate_pairs.tsv`

---

## 1. Objectives & Overview

The objective of Step 2 is to reduce the massive Source-1 $\times$ Source-2/Source-3 pairwise search space ($\approx 1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairs) down to a compact, high-recall candidate pool fed to the downstream matching classifier.

### Core Guarantees:
1. **Dynamic Country Partitioning**: Open-set grouping supporting `US`, `INDIA`, `FRANCE`, and arbitrary unseen countries with zero hardcoding.
2. **Multi-Signal Candidate Retrieval**:
   - **Signal A**: Character 3-Gram Sparse TF-IDF Cosine Retrieval.
   - **Signal B**: Distinct Brand Token Inverted Index.
   - **Signal C**: Physical Address Number Co-occurrence.
3. **Candidate Union with Signal Origin Tracking**: Multi-signal union ensures candidates do not need to satisfy all three signals, protecting recall ceiling.
4. **Empirical Top-$K$ Hyperparameter Tuning**: Systematic evaluation across $K \in \{5, 10, 15, 20, 30, 50\}$ on held-out validation data to select optimal $K$ based on candidate recall vs. computational budget.
5. **Output Compliance**: Validated generation of `output/candidate_pairs.tsv` adhering to challenge format and passing `utils/validate_submission.py`.

---

## 2. Architecture & Pipeline Flow

```
                      Normalized S1 & S2/S3 Data
                                  │
                                  ▼
                    [Dynamic Country Partitioning]
                       (US / INDIA / FRANCE / ...)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
[Signal A: Char 3-Gram]  [Signal B: Token Index]  [Signal C: Address Num]
  - Sparse TF-IDF Cosine   - Informative Brand       - Physical Premise Digits
  - Typos, URLs, Noise       Words, Inversions       - Abbreviated/Noisy Names
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    [Multi-Signal Candidate Union]
                      (Retains Origin Metadata)
                                  │
                                  ▼
                   [Composite Scoring & Top-K Slicing]
                     (Tuned K on Validation Split)
                                  │
                                  ▼
                     [output/candidate_pairs.tsv]
                   (Checked via Submission Validator)
```

---

## 3. Modular Component Specifications

The blocking engine is structured into modular functions under `code/business_entity_resolution/src/`:

### 3.1. `country_partition(df_s1, df_targets)`
- Dynamically extracts unique country labels from `df_s1['country']`.
- Partitions both Source 1 and Target (Source 2 + Source 3) datasets into disjoint subsets.
- Eliminates 65.8% of irrelevant comparisons with **0.00% recall loss**.

### 3.2. `generate_3gram_tfidf_candidates(s1_records, target_records, top_n_tfidf)`
- **Feature Space**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, max_features=150,000, sublinear_tf=True)`.
- **Target Vectorization**: Fits on `clean_name` of target records.
- **Search Engine**: Chunked sparse matrix dot product with threshold filtering ($\text{cosine similarity} \ge 0.25$).
- **Coverage**: Character typos, OCR errors, concatenated domain URLs (`siiainvestments`), transliterated phonetic variations.

### 3.3. `generate_token_candidates(s1_records, target_records, top_n_token)`
- **Inverted Index Construction**:
  - Tokenizes `clean_name` into meaningful words (length $\ge 3$).
  - Filters out high-frequency stop tokens (`pvt`, `ltd`, `llc`, `inc`, `corp`, `co`, `and`, `the`, `of`, `services`, `enterprises`).
  - Maps `token -> [target_indices]`.
- **Retrieval**: Matches informative brand words and scores targets using token specificity weights.
- **Coverage**: Word-order inversions (`lowe galata` $\leftrightarrow$ `galatalowe`), DBA trade names (`Painters Local` $\leftrightarrow$ `formerly Painters Local`), and prefix noise.

### 3.4. `generate_address_number_candidates(s1_records, target_records, top_n_addr)`
- **Index Construction**:
  - Maps primary address number + state/city tokens to target IDs.
- **Filtering**:
  - Requires minimal token compatibility to avoid cross-business building confusion.
- **Coverage**: Recovers true matches where names are severely damaged/abbreviated but physical premises match.

### 3.5. `union_candidates(candidates_a, candidates_b, candidates_c)`
- Computes the union: $\text{Candidates}(S_1) = \text{Candidates}_A(S_1) \cup \text{Candidates}_B(S_1) \cup \text{Candidates}_C(S_1)$.
- Attaches signal origin flags (`from_tfidf`, `from_token`, `from_address`) for diagnostic tracking.
- Deduplicates candidate pairs per Source 1 entity.

### 3.6. `rank_candidates(candidate_pool, k)`
- Evaluates composite ranking score:
  $$\text{Score}(S_1, T) = \text{Score}_{\text{tfidf}} + 0.35 \cdot \text{Score}_{\text{token}} + 0.25 \cdot \mathbb{I}(\text{Number Match}) + 0.20 \cdot \text{Jaccard}_{\text{address}}$$
- Slices the top $K$ candidates per anchor. Singletons with 0 qualifying candidates retain an empty candidate list.

---

## 4. Empirical Hyperparameter Tuning ($K$) on Validation Split

### 4.1. Validation Setup
- **Held-out Split**: 100,000 Source 1 entities from `train_source1_normalized.tsv` with corresponding labels from `train_ground_truth.tsv`.
- **Target Search Pool**: Full train Source 2 and Source 3 normalized records (~10.3M records).

### 4.2. Swept Configurations
We evaluate candidate generation performance across:
$$K \in \{5, 10, 15, 20, 30, 50\}$$

### 4.3. Evaluation Metrics:
1. **Candidate Recall**:
   $$\text{Candidate Recall} = \frac{|\text{True Matches Found in Candidate Set}|}{|\text{Total Ground Truth True Matches}|}$$
2. **Reduction Ratio**:
   $$\text{Reduction Ratio} = 1 - \frac{|\text{Total Candidate Pairs}|}{|S_1| \times |S_{2+3}|}$$
3. **Average Candidates per S1 Entity**:
   $$\bar{C} = \frac{\text{Total Candidate Pairs}}{|S_1|}$$
4. **Signal Contribution Matrix**: Breakdown of recall contributed by TF-IDF, Token Index, and Address Numbers.
5. **Runtime & Memory Usage**: Throughput (queries/second) and peak RAM consumption.

---

## 5. Candidate Output Generation & Validation

### Output File: `output/candidate_pairs.tsv`
- **Format**: `source1_entity_id \t candidate_entity_ids` (comma-separated target IDs, tab-separated columns).
- **Rules Enforced**:
  - Exactly 1 row per test Source 1 entity (1,732,544 rows).
  - Empty string for entities with zero candidates.
  - No duplicate IDs within any candidate list.
  - S2- and S3- prefixed IDs only.
- **Local Validator**: Verified automatically with:
  ```bash
  python utils/validate_submission.py \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
  ```

---

## 6. Execution Roadmap

1. **Step 2.1**: Implement `code/business_entity_resolution/src/blocking.py` with all modular components.
2. **Step 2.2**: Implement `code/business_entity_resolution/src/tune_blocking.py` and run empirical tuning on validation data.
3. **Step 2.3**: Analyze tuning results, select optimal $K$, and generate candidate set diagnostic reports.
4. **Step 2.4**: Implement `code/business_entity_resolution/src/generate_candidates.py` to produce `output/candidate_pairs.tsv` and validate with `validate_submission.py`.

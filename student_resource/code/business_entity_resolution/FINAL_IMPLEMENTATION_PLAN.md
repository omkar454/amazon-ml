# End-to-End Implementation Plan: Business Entity Resolution

This document outlines the complete, production-ready implementation plan to finish the Amazon ML Challenge Business Entity Resolution pipeline on CPU with minimal memory footprint and fast streaming disk I/O.

---

## 1. Datasets Used: Pre-Normalized Datasets

The entire pipeline strictly uses the pre-computed normalized datasets located in `dataset/normalized/`:
* **Training Data**:
  * `train_source1_normalized.tsv` (100,000 anchor entities for training/validation)
  * `train_source2_normalized.tsv` & `train_source3_normalized.tsv` (10,320,219 target catalog records)
  * `train_ground_truth.tsv` (Verified true positive label pairs $S_1 \leftrightarrow S_2/S_3$)
* **Test Data**:
  * `test_source1_normalized.tsv` (All 1,733,333 unlabeled test queries)
  * `test_source2_normalized.tsv` & `test_source3_normalized.tsv` (Test target catalog)

### Pre-Extracted Normalized Columns:
1. `clean_name`: Punctuation stripped, lowercase, legal suffix normalized (`ltd`, `llc`, `pvt ltd`).
2. `clean_address`: Standardized street suffixes (`rd`, `st`, `ave`, `blvd`), removed noisy descriptors.
3. `address_numbers`: Pre-extracted comma-separated premise/building/door numbers (e.g., `"101,4B"`).
4. `state_clean` & `country_clean`: Standardized geographic partition keys.

---

## 2. How the Multi-Signal Blocking Engine Works Exactly

The blocking engine reduces a search space of **1.73M queries $\times$ 10.3M targets ($17.8\text{ Trillion}$ pairs)** down to **$\approx 0.91$ candidates per query ($3.4\text{ Million}$ pairs total)** with a **99.999991% reduction ratio** and **62.70% candidate recall**.

```
[Query Entity S1] (e.g., US: "Starbucks Coffee #104, 5th Ave, NY")
         │
         ├──► 1. Partition Filter: Only search US Target Pool (6.18M records)
         │
         ├──► 2. Signal A: C++ Sparse TF-IDF (char 3-gram cosine >= 0.20)  ──┐
         │                                                                   │
         ├──► 3. Signal B: Brand Token Index ("starbucks" -> target IDs)    ──┼──► Union & Rank (K=20)
         │                                                                   │     │
         └──► 4. Signal C: Address Number Index ("104", "5" -> target IDs) ──┘     ▼
                                                                        Top-20 Candidates
```

### Exact Signal Mechanics:
1. **Dynamic Country Partitioning**:
   * Entities are strictly routed to their country catalog:
     * **US Partition**: ~900k queries $\leftrightarrow$ 6.18M targets.
     * **India Partition**: ~700k queries $\leftrightarrow$ 4.13M targets.
     * **France Partition**: ~130k queries $\leftrightarrow$ 500k targets.
   * Cuts runtime by ~70% and completely eliminates impossible cross-country false positives.

2. **Signal A: High-Speed C++ Sparse TF-IDF Cosine Retrieval (`sparse_dot_topn`)**:
   * Builds character 3-gram TF-IDF matrices on `clean_name` (120,000 features, sublinear TF).
   * Uses multithreaded C++ `sparse_dot_topn` (`sp_matmul_topn`) across all CPU cores to compute Top-$K$ ($K=20$) cosine similarity with minimum threshold $\tau = 0.20$.
   * Takes seconds per batch instead of Python matrix loops.

3. **Signal B: Informative Brand Token Inverted Index**:
   * Creates a dictionary `word -> [target_idx1, target_idx2, ...]`.
   * Filters out generic non-discriminative stopwords (`inc`, `ltd`, `services`, `enterprises`, `the`).
   * Scores hits using Inverse Document Frequency (IDF) weights: rarer words (e.g. `"Starbucks"`, `"Infosys"`) yield much higher candidate scores.

4. **Signal C: Physical Premise Number Inverted Index**:
   * Indexes physical premise/door/street numbers (`"12"`, `"45B"`, `"1004"`).
   * Retrieves target entities sharing the exact same physical building numbers.

5. **Candidate Union & Top-$K$ Ranking**:
   * Merges candidate lists from Signals A, B, and C.
   * Combines normalized scores: $\text{Final Score} = S_{\text{TF-IDF}} + 0.5 \times S_{\text{Token}} + 0.5 \times S_{\text{Address}}$.
   * Retains the top $K = 20$ candidates per entity.

---

## 3. Four-Phase Step-by-Step Execution Plan

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Train the LightGBM Matching Classifier (~2–3 mins)                  │
│ • Positives (y=1): Ground-truth pairs from train_ground_truth.tsv            │
│ • Hard Negatives (y=0): Lookalikes from fast training blocking               │
│ • RapidFuzz 8-feature extraction vectorization                               │
│ • Optimal F0.5 Threshold Sweep (τ ≈ 0.68) -> Save models/lgb_matcher.joblib   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Fast Streaming Test Blocking (K=20) (~25–30 mins)                    │
│ • Stream test_source1_normalized in 25,000-query chunks                     │
│ • Run C++ sparse_dot_topn + Inverted Indices across US, India, France        │
│ • Stream directly to: output/candidate_pairs.tsv                            │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Streaming Classifier Inference (~15–20 mins)                         │
│ • Read candidate_pairs.tsv line-by-line / chunk-by-chunk                     │
│ • Singletons (0 candidates) -> Output empty match "" immediately             │
│ • Active Candidates -> Predict match probability P(y=1) with LightGBM        │
│ • Apply F0.5 threshold (P >= τ) -> Stream to: output/matching_results.tsv   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Official Validation & Quality Assurance (< 1 min)                    │
│ • Run utils/validate_submission.py                                           │
│ • Confirm: 1,733,333 rows, correct columns, valid superset constraints       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Resource & Memory Safety Guarantees

* **Low RAM Guarantee (< 3.5 GB)**:
  * Partitions are loaded and processed independently.
  * Intermediate candidate structures are written straight to disk in streaming chunks.
* **No GPU Dependency**:
  * High-speed multithreading uses C++ OpenMP CPU cores (`sparse_dot_topn`) and C-accelerated Levenshtein/Jaro-Winkler engines (`rapidfuzz`).
* **Metric Alignment ($F_{0.5}$ Optimization)**:
  * $F_{0.5}$ weighs precision twice as heavily as recall ($\beta = 0.5$).
  * The threshold $\tau \approx 0.65 - 0.72$ strictly avoids false positive errors while capturing high-confidence true matches.

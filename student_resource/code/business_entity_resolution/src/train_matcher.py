"""
Training Script for LightGBM Matcher
Builds a high-precision pairwise matching classifier:
1. Gathers true positive pairs from train_ground_truth.tsv
2. Gathers hard negative candidate pairs from multi-signal blocking lookalikes
3. Extracts 10 rich RapidFuzz + metadata features
4. Trains LightGBM with GroupKFold CV and tunes optimal F0.5 decision threshold
5. Saves trained model and metadata to models/lgb_matcher.joblib
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score
import lightgbm as lgb

from blocking import (
    country_partition,
    generate_3gram_tfidf_candidates,
    generate_token_candidates,
    generate_address_number_candidates,
    union_and_rank_candidates
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "addr_ratio",
    "addr_token_set",
    "num_match",
    "num_jaccard",
    "country_match",
    "len_diff"
]


def extract_pair_features(n1, n2, a1, a2, nums1_str, nums2_str, s1_ctry, t_ctry) -> list:
    """Fast extraction of 10 numerical similarity features between two entity records."""
    # Name similarities
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    f_name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    # Address similarities
    f_addr_ratio = fuzz.ratio(a1, a2) / 100.0
    f_addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
    
    # Address Premise numbers
    set1 = set(n.strip() for n in nums1_str.split(",") if n.strip())
    set2 = set(n.strip() for n in nums2_str.split(",") if n.strip())
    
    f_num_match = 1.0 if (set1 and set2 and (set1 & set2)) else 0.0
    f_num_jaccard = (len(set1 & set2) / len(set1 | set2)) if (set1 and set2) else 0.0
    
    # Geographic matches
    f_country_match = 1.0 if (s1_ctry and t_ctry and s1_ctry == t_ctry) else 0.0
    
    # Length difference
    f_len_diff = abs(len(n1) - len(n2))
    
    return [
        f_name_ratio,
        f_name_token_set,
        f_name_token_sort,
        f_name_partial,
        f_addr_ratio,
        f_addr_token_set,
        f_num_match,
        f_num_jaccard,
        f_country_match,
        f_len_diff
    ]


def build_training_dataset(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    ground_truth_dict: dict,
    max_hard_neg_per_anchor: int = 5
) -> pd.DataFrame:
    """
    Constructs a balanced, high-quality training dataset with:
    - True Positives (y=1) from ground_truth_dict
    - Hard Negatives (y=0) from multi-signal candidate generation lookalikes
    """
    print("\n" + "=" * 80)
    print("BUILDING LABELED TRAINING DATASET (POSITIVES + HARD NEGATIVES)")
    print("=" * 80)
    
    s1_ids_set = set(df_s1["entity_id"])
    
    # 1. Gather Positives (y=1)
    positive_pairs = []
    for s1_id, true_targets in ground_truth_dict.items():
        if s1_id not in s1_ids_set:
            continue
        for tgt_id in true_targets:
            positive_pairs.append((s1_id, tgt_id, 1))
                
    print(f"Total True Positive Pairs (y=1) from ground truth: {len(positive_pairs):,}")
    
    # 2. Gather Hard Negatives (y=0) via country-partitioned blocking
    print("Extracting Hard Negatives via multi-signal blocking on training anchors...")
    partitions = country_partition(df_s1, df_targets)
    
    hard_negatives = []
    
    for country, part in partitions.items():
        s1_sub = part["s1"]
        tgt_sub = part["targets"]
        
        print(f"\n  --- Partition: {country} (S1: {len(s1_sub):,}, Targets: {len(tgt_sub):,}) ---")
        cands_tfidf = generate_3gram_tfidf_candidates(s1_sub, tgt_sub, top_n_tfidf=10, min_similarity=0.20)
        cands_token = generate_token_candidates(s1_sub, tgt_sub, top_n_token=10)
        cands_addr = generate_address_number_candidates(s1_sub, tgt_sub, top_n_addr=10)
        
        part_cands = union_and_rank_candidates(
            list(s1_sub["entity_id"]),
            cands_tfidf,
            cands_token,
            cands_addr,
            k=15
        )
        
        for s1_id, c_list in part_cands.items():
            true_tgts = ground_truth_dict.get(s1_id, set())
            neg_count = 0
            for c_id in c_list:
                if c_id not in true_tgts:
                    hard_negatives.append((s1_id, c_id, 0))
                    neg_count += 1
                    if neg_count >= max_hard_neg_per_anchor:
                        break
                        
    print(f"\nTotal Hard Negative Pairs (y=0) generated: {len(hard_negatives):,}")
    
    # 3. Combine pairs
    all_pairs = positive_pairs + hard_negatives
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    
    # 4. Only index the ACTIVE targets that appear in all_pairs (ultra-low memory!)
    print(f"Filtering metadata to {len(all_pairs):,} active pair entities...")
    t_idx = time.time()
    active_s1_ids = set(p[0] for p in all_pairs)
    active_tgt_ids = set(p[1] for p in all_pairs)
    
    df_active_s1 = df_s1[df_s1["entity_id"].isin(active_s1_ids)]
    df_active_tgt = df_targets[df_targets["entity_id"].isin(active_tgt_ids)]
    
    s1_names = dict(zip(df_active_s1["entity_id"], df_active_s1["clean_name"].fillna("")))
    s1_addrs = dict(zip(df_active_s1["entity_id"], df_active_s1["clean_address"].fillna("")))
    s1_nums = dict(zip(df_active_s1["entity_id"], df_active_s1["address_numbers"].fillna("")))
    s1_countries = dict(zip(df_active_s1["entity_id"], df_active_s1["country"].fillna("")))
    
    tgt_names = dict(zip(df_active_tgt["entity_id"], df_active_tgt["clean_name"].fillna("")))
    tgt_addrs = dict(zip(df_active_tgt["entity_id"], df_active_tgt["clean_address"].fillna("")))
    tgt_nums = dict(zip(df_active_tgt["entity_id"], df_active_tgt["address_numbers"].fillna("")))
    tgt_countries = dict(zip(df_active_tgt["entity_id"], df_active_tgt["country"].fillna("")))
    print(f"Indexed {len(s1_names):,} active S1 and {len(tgt_names):,} active targets in {time.time()-t_idx:.2f}s!")
    
    # 5. Extract Features
    print(f"\nExtracting RapidFuzz features for all {len(all_pairs):,} labeled pairs...")
    t0 = time.time()
    
    feature_rows = []
    labels = []
    s1_ids = []
    tgt_ids = []
    
    for s1_id, tgt_id, label in all_pairs:
        if s1_id not in s1_names or tgt_id not in tgt_names:
            continue
        n1 = s1_names.get(s1_id, "")
        n2 = tgt_names.get(tgt_id, "")
        a1 = s1_addrs.get(s1_id, "")
        a2 = tgt_addrs.get(tgt_id, "")
        num1 = s1_nums.get(s1_id, "")
        num2 = tgt_nums.get(tgt_id, "")
        c1 = str(s1_countries.get(s1_id, "")).lower()
        c2 = str(tgt_countries.get(tgt_id, "")).lower()
        
        feats = extract_pair_features(n1, n2, a1, a2, num1, num2, c1, c2)
        feature_rows.append(feats)
        labels.append(label)
        s1_ids.append(s1_id)
        tgt_ids.append(tgt_id)
        
    df_dataset = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    df_dataset["label"] = labels
    df_dataset["s1_id"] = s1_ids
    df_dataset["tgt_id"] = tgt_ids
    
    elapsed = time.time() - t0
    print(f"Feature extraction completed in {elapsed:.2f}s ({len(df_dataset)/elapsed:,.0f} pairs/sec)!")
    return df_dataset


def train_lightgbm_matcher(df_train: pd.DataFrame):
    """
    Trains LightGBM classifier with GroupKFold cross validation
    and tunes the optimal F0.5 decision threshold.
    """
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM MATCHER WITH GROUP-KFOLD CV & F0.5 TUNING")
    print("=" * 80)
    
    X = df_train[FEATURE_NAMES].values
    y = df_train["label"].values
    groups = df_train["s1_id"].values
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df_train))
    
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    print(f"Running 5-Fold GroupKFold Cross-Validation on {len(df_train):,} pairs...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        
    # Sweep threshold to maximize F0.5 (beta=0.5 penalizes false positives 2x)
    print("\nSweeping Decision Thresholds to Optimize F0.5 Metric:")
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F0.5 Score':>12} | {'F1 Score':>10}")
    print("-" * 65)
    
    best_thresh = 0.50
    best_f05 = 0.0
    
    for thresh in np.arange(0.30, 0.92, 0.05):
        pred_binary = (oof_preds >= thresh).astype(int)
        p = precision_score(y, pred_binary, zero_division=0)
        r = recall_score(y, pred_binary, zero_division=0)
        f05 = fbeta_score(y, pred_binary, beta=0.5, zero_division=0)
        f1 = fbeta_score(y, pred_binary, beta=1.0, zero_division=0)
        
        star = " ★ BEST" if f05 > best_f05 else ""
        print(f"{thresh:10.2f} | {p*100:9.2f}% | {r*100:9.2f}% | {f05*100:11.2f}% | {f1*100:9.2f}%{star}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print("-" * 65)
    print(f"Optimal F0.5 Threshold: {best_thresh:.2f} (OOF F0.5 = {best_f05*100:.2f}%)")
    
    # Train final model on all data
    print("\nTraining Final LightGBM model on all labeled pairs...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)
    
    # Print Feature Importances
    print("\nFeature Importances:")
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]:<20}: {importances[idx]:5d}")
        
    # Save Model Artifact
    model_path = os.path.join(MODEL_DIR, "lgb_matcher.joblib")
    model_artifact = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "optimal_threshold": float(best_thresh),
        "best_f05": float(best_f05)
    }
    joblib.dump(model_artifact, model_path)
    print(f"\nModel artifact successfully saved to: {model_path}")
    return final_model, best_thresh


def main():
    print("=" * 80)
    print("LIGHTGBM MATCHING MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Load normalized datasets
    s1_path = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
    s2_path = os.path.join(NORM_DIR, "train_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "train_source3_normalized.tsv")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    
    print("Loading normalized training data...")
    # Load 30,000 S1 records for fast, balanced training set generation
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=30000, dtype=str, keep_default_na=False)
    
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    print(f"Loaded {len(df_s1):,} S1 Anchors and {len(df_targets):,} Target Catalog records.")
    
    # Load ground truth dictionary
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    ground_truth_dict = defaultdict(set)
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if matched_str:
            for tid in matched_str.split(","):
                if tid.strip():
                    ground_truth_dict[s1_id].add(tid.strip())
                    
    # 2. Build Dataset
    df_dataset = build_training_dataset(df_s1, df_targets, ground_truth_dict)
    
    # 3. Train & Tune LightGBM
    train_lightgbm_matcher(df_dataset)
    
    total_time = time.time() - t0
    print(f"\nEntire Training Pipeline Finished in {total_time:.2f}s ({total_time/60:.2f} minutes)!")


if __name__ == "__main__":
    main()

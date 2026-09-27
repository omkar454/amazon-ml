"""
Hybrid Model Training Pipeline (LightGBM + RapidFuzz + Transformer Embeddings)
Builds a high-precision pairwise matching classifier with 11 features:
1. Gathers true positive pairs from train_ground_truth.tsv
2. Gathers hard negative candidate pairs from multi-signal blocking
3. Computes 10 RapidFuzz features + 11th Transformer Cosine Similarity feature
4. Performs 5-Fold GroupKFold Cross-Validation and tunes optimal F0.5 decision threshold
5. Saves trained model and calibrated threshold to models/lgb_hybrid_matcher.joblib
"""

import os
import gc
import sys
import json
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
    build_target_blocking_index,
    query_blocking_index_chunk
)
from embed_entities import get_embedding_engine, encode_texts_batch, format_entity_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

HYBRID_FEATURE_NAMES = [
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "addr_ratio",
    "addr_token_set",
    "num_match",
    "num_jaccard",
    "country_match",
    "len_diff",
    "embed_cosine_sim"
]


def extract_pair_features(
    n1: str, n2: str,
    a1: str, a2: str,
    nums1_str: str, nums2_str: str,
    s1_ctry: str, t_ctry: str,
    cosine_sim: float
) -> list:
    """Fast extraction of 11 hybrid features (10 lexical + 1 transformer cosine similarity)."""
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    f_name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    f_addr_ratio = fuzz.ratio(a1, a2) / 100.0
    f_addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
    
    set1 = set(n.strip() for n in nums1_str.split(",") if n.strip()) if nums1_str else set()
    set2 = set(n.strip() for n in nums2_str.split(",") if n.strip()) if nums2_str else set()
    
    f_num_match = 1.0 if (set1 and set2 and (set1 & set2)) else 0.0
    f_num_jaccard = (len(set1 & set2) / len(set1 | set2)) if (set1 and set2) else 0.0
    
    f_country_match = 1.0 if (s1_ctry and t_ctry and s1_ctry == t_ctry) else 0.0
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
        f_len_diff,
        float(cosine_sim)
    ]


def build_hybrid_training_dataset(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    ground_truth_dict: dict,
    engine_type: str,
    model,
    max_hard_neg_per_anchor: int = 5
) -> pd.DataFrame:
    """Constructs labeled training pairs and calculates all 11 hybrid features."""
    print("\n" + "=" * 80)
    print("BUILDING LABELED HYBRID DATASET (POSITIVES + HARD NEGATIVES)")
    print("=" * 80)
    
    s1_ids_set = set(df_s1["entity_id"])
    
    # 1. Positives
    positive_pairs = []
    for s1_id, true_targets in ground_truth_dict.items():
        if s1_id not in s1_ids_set:
            continue
        for tgt_id in true_targets:
            positive_pairs.append((s1_id, tgt_id, 1))
            
    print(f"Total True Positive Pairs (y=1): {len(positive_pairs):,}")
    
    # 2. Hard Negatives via Blocking
    print("Extracting Hard Negatives via multi-signal blocking...")
    partitions = country_partition(df_s1, df_targets)
    hard_negatives = []
    
    for country, part in partitions.items():
        s1_sub = part["s1"]
        tgt_sub = part["targets"]
        
        if len(tgt_sub) == 0:
            continue
            
        index_bundle = build_target_blocking_index(tgt_sub)
        s1_ids = s1_sub["entity_id"].values
        s1_names = s1_sub["clean_name"].values
        s1_nums = s1_sub["address_numbers"].values if "address_numbers" in s1_sub.columns else np.array([""] * len(s1_sub))
        
        cands_results = query_blocking_index_chunk(s1_ids, s1_names, s1_nums, index_bundle, k=15)
        
        for s1_id, c_list in cands_results:
            true_tgts = ground_truth_dict.get(s1_id, set())
            neg_count = 0
            for c_id in c_list:
                if c_id not in true_tgts:
                    hard_negatives.append((s1_id, c_id, 0))
                    neg_count += 1
                    if neg_count >= max_hard_neg_per_anchor:
                        break
                        
    print(f"Total Hard Negative Pairs (y=0): {len(hard_negatives):,}")
    
    all_pairs = positive_pairs + hard_negatives
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    
    # 3. Filter active entities for memory efficiency
    active_s1_ids = set(p[0] for p in all_pairs)
    active_tgt_ids = set(p[1] for p in all_pairs)
    
    df_active_s1 = df_s1[df_s1["entity_id"].isin(active_s1_ids)].copy()
    df_active_tgt = df_targets[df_targets["entity_id"].isin(active_tgt_ids)].copy()
    
    s1_names = dict(zip(df_active_s1["entity_id"], df_active_s1["clean_name"].fillna("")))
    s1_addrs = dict(zip(df_active_s1["entity_id"], df_active_s1["clean_address"].fillna("")))
    s1_nums = dict(zip(df_active_s1["entity_id"], df_active_s1["address_numbers"].fillna("")))
    s1_countries = dict(zip(df_active_s1["entity_id"], df_active_s1["country"].fillna("")))
    
    tgt_names = dict(zip(df_active_tgt["entity_id"], df_active_tgt["clean_name"].fillna("")))
    tgt_addrs = dict(zip(df_active_tgt["entity_id"], df_active_tgt["clean_address"].fillna("")))
    tgt_nums = dict(zip(df_active_tgt["entity_id"], df_active_tgt["address_numbers"].fillna("")))
    tgt_countries = dict(zip(df_active_tgt["entity_id"], df_active_tgt["country"].fillna("")))
    
    # 4. Generate Embeddings for Active Training Entities
    print(f"\nEncoding {len(df_active_s1):,} active S1 and {len(df_active_tgt):,} active Target entities...")
    t_emb = time.time()
    
    s1_texts = [format_entity_text(r["clean_name"], r["clean_address"], r["country"]) for _, r in df_active_s1.iterrows()]
    s1_vecs_arr = encode_texts_batch(engine_type, model, s1_texts, batch_size=256)
    s1_vec_dict = {eid: s1_vecs_arr[i].astype(np.float32) for i, eid in enumerate(df_active_s1["entity_id"])}
    
    tgt_texts = [format_entity_text(r["clean_name"], r["clean_address"], r["country"]) for _, r in df_active_tgt.iterrows()]
    tgt_vecs_arr = encode_texts_batch(engine_type, model, tgt_texts, batch_size=256)
    tgt_vec_dict = {eid: tgt_vecs_arr[i].astype(np.float32) for i, eid in enumerate(df_active_tgt["entity_id"])}
    
    print(f"Computed embeddings in {time.time()-t_emb:.2f}s!")
    
    # 5. Extract Features
    print(f"Extracting 11 Hybrid features for {len(all_pairs):,} labeled pairs...")
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
        
        # Cosine similarity
        v1 = s1_vec_dict.get(s1_id)
        v2 = tgt_vec_dict.get(tgt_id)
        cosine_sim = float(np.dot(v1, v2)) if (v1 is not None and v2 is not None) else 0.0
        
        feats = extract_pair_features(n1, n2, a1, a2, num1, num2, c1, c2, cosine_sim)
        feature_rows.append(feats)
        labels.append(label)
        s1_ids.append(s1_id)
        tgt_ids.append(tgt_id)
        
    df_dataset = pd.DataFrame(feature_rows, columns=HYBRID_FEATURE_NAMES)
    df_dataset["label"] = labels
    df_dataset["s1_id"] = s1_ids
    df_dataset["tgt_id"] = tgt_ids
    
    print(f"Dataset construction complete in {time.time()-t0:.2f}s ({len(df_dataset):,} pairs)!")
    return df_dataset


def train_hybrid_lightgbm(df_train: pd.DataFrame):
    """5-Fold GroupKFold CV with F0.5 Threshold Calibration."""
    print("\n" + "=" * 80)
    print("TRAINING HYBRID LIGHTGBM (11 FEATURES) WITH GROUP-KFOLD & F0.5 TUNING")
    print("=" * 80)
    
    X = df_train[HYBRID_FEATURE_NAMES].values
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
        
    # Sweep threshold
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
        
        star = " * BEST" if f05 > best_f05 else ""
        print(f"{thresh:10.2f} | {p*100:9.2f}% | {r*100:9.2f}% | {f05*100:11.2f}% | {f1*100:9.2f}%{star}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print("-" * 65)
    print(f"Optimal F0.5 Threshold: {best_thresh:.2f} (OOF F0.5 = {best_f05*100:.2f}%)")
    
    # Final Model
    print("\nTraining Final Hybrid LightGBM model on all pairs...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)
    
    print("\nFeature Importances (11 Features):")
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for idx in sorted_idx:
        print(f"  {HYBRID_FEATURE_NAMES[idx]:<22}: {importances[idx]:5d}")
        
    model_path = os.path.join(MODEL_DIR, "lgb_hybrid_matcher.joblib")
    model_artifact = {
        "model": final_model,
        "feature_names": HYBRID_FEATURE_NAMES,
        "optimal_threshold": float(best_thresh),
        "best_f05": float(best_f05)
    }
    joblib.dump(model_artifact, model_path)
    print(f"\nHybrid model artifact saved to: {model_path}")
    return final_model, best_thresh


def main():
    print("=" * 80)
    print("HYBRID LIGHTGBM MODEL TRAINING PIPELINE (11 FEATURES)")
    print("=" * 80)
    
    t0 = time.time()
    
    s1_path = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
    s2_path = os.path.join(NORM_DIR, "train_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "train_source3_normalized.tsv")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=30000, dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    ground_truth_dict = defaultdict(set)
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if matched_str:
            for tid in matched_str.split(","):
                if tid.strip():
                    ground_truth_dict[s1_id].add(tid.strip())
                    
    engine_type, model = get_embedding_engine()
    
    df_dataset = build_hybrid_training_dataset(df_s1, df_targets, ground_truth_dict, engine_type, model)
    train_hybrid_lightgbm(df_dataset)
    
    total_time = time.time() - t0
    print(f"\nEntire Hybrid Training Finished in {total_time/60:.2f} minutes!")


if __name__ == "__main__":
    main()

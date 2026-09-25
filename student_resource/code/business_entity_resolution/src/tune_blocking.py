"""
Hyperparameter Tuning Script for Candidate Generation (Blocking)
Evaluates Candidate Recall, Pair Counts, and Reduction Ratio across K in {5, 10, 15, 20, 30, 50}
on a held-out validation set of 100,000 Source 1 records against full S2/S3 targets.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from blocking import (
    country_partition,
    generate_3gram_tfidf_candidates,
    generate_token_candidates,
    generate_address_number_candidates,
    union_and_rank_candidates,
    evaluate_candidate_recall
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

VAL_SIZE = 100000
K_VALUES = [5, 10, 15, 20, 30, 50]

def load_validation_data():
    print("=" * 80)
    print(f"1. LOADING VALIDATION DATA ({VAL_SIZE:,} S1 Anchors against full S2/S3 Targets)")
    print("=" * 80)
    
    t0 = time.time()
    
    # Load 100k S1 normalized records
    s1_path = os.path.join(NORM_DIR, "train_source1_normalized.tsv")
    print(f"Loading {VAL_SIZE:,} records from {os.path.basename(s1_path)}...")
    df_s1_val = pd.read_csv(s1_path, sep="\t", nrows=VAL_SIZE, dtype=str, keep_default_na=False)
    
    # Load ground truth for these 100k records
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    print(f"Loading ground truth labels from {os.path.basename(gt_path)}...")
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=VAL_SIZE, dtype=str, keep_default_na=False)
    
    val_s1_ids = set(df_s1_val['entity_id'])
    ground_truth_dict = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        if s1_id in val_s1_ids:
            if m_str.strip():
                ground_truth_dict[s1_id] = set(x.strip() for x in m_str.split(",") if x.strip())
            else:
                ground_truth_dict[s1_id] = set()
                
    # Load full S2 and S3 normalized targets
    s2_path = os.path.join(NORM_DIR, "train_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "train_source3_normalized.tsv")
    
    print(f"Loading target catalog from {os.path.basename(s2_path)} & {os.path.basename(s3_path)}...")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    elapsed = time.time() - t0
    print(f"Validation setup complete in {elapsed:.2f}s!")
    print(f"  Validation S1 Entities: {len(df_s1_val):,}")
    print(f"  Total Ground Truth Matches in Split: {sum(len(v) for v in ground_truth_dict.values()):,}")
    print(f"  Target Pool (S2+S3): {len(df_targets):,} records")
    
    return df_s1_val, df_targets, ground_truth_dict

def main():
    df_s1_val, df_targets, ground_truth_dict = load_validation_data()
    
    print("\n" + "=" * 80)
    print("2. RUNNING MULTI-SIGNAL BLOCKING ENGINE ON VALIDATION SET")
    print("=" * 80)
    
    # 1. Dynamic Country Partitioning
    partitions = country_partition(df_s1_val, df_targets)
    print(f"Discovered {len(partitions)} Country Partitions: {list(partitions.keys())}")
    
    all_tfidf_cands = {}
    all_token_cands = {}
    all_addr_cands = {}
    
    total_blocking_time = 0
    
    for country, part in partitions.items():
        s1_sub = part["s1"]
        tgt_sub = part["targets"]
        
        print(f"\n--- Processing Partition: {country} (S1: {len(s1_sub):,}, Targets: {len(tgt_sub):,}) ---")
        t_p_start = time.time()
        
        # Max top-N during retrieval to allow sweeping K up to 50
        max_k = max(K_VALUES)
        
        # Signal A: TF-IDF
        cands_tfidf = generate_3gram_tfidf_candidates(s1_sub, tgt_sub, top_n_tfidf=max_k, min_similarity=0.20)
        all_tfidf_cands.update(cands_tfidf)
        
        # Signal B: Brand Tokens
        cands_token = generate_token_candidates(s1_sub, tgt_sub, top_n_token=max_k)
        all_token_cands.update(cands_token)
        
        # Signal C: Address Numbers
        cands_addr = generate_address_number_candidates(s1_sub, tgt_sub, top_n_addr=20)
        all_addr_cands.update(cands_addr)
        
        elapsed_p = time.time() - t_p_start
        total_blocking_time += elapsed_p
        print(f"Partition {country} finished in {elapsed_p:.2f}s")
        
    print(f"\nAll blocking signals extracted in {total_blocking_time:.2f}s!")
    
    print("\n" + "=" * 80)
    print("3. EMPIRICAL TOP-K HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print(f"{'K':>4} | {'Recall':>9} | {'Found Matches':>14} | {'Total Pairs':>13} | {'Avg Pairs/Entity':>17} | {'Reduction Ratio':>16}")
    print("-" * 85)
    
    val_s1_ids = list(df_s1_val['entity_id'])
    
    results = []
    for k in K_VALUES:
        candidate_dict = union_and_rank_candidates(
            val_s1_ids,
            all_tfidf_cands,
            all_token_cands,
            all_addr_cands,
            k=k
        )
        
        metrics = evaluate_candidate_recall(
            candidate_dict,
            ground_truth_dict,
            total_s1_count=len(df_s1_val),
            total_target_count=len(df_targets)
        )
        
        results.append((k, metrics))
        print(f"{k:4d} | {metrics['candidate_recall']*100:8.2f}% | {metrics['found_true_matches']:14,d} | {metrics['total_candidate_pairs']:13,d} | {metrics['avg_candidates_per_entity']:17.2f} | {metrics['reduction_ratio']*100:15.6f}%")
        
    print("=" * 85)
    
    # Analyze individual blocker contribution at K=20
    print("\n" + "=" * 80)
    print("4. ISOLATED SIGNAL RECALL CONTRIBUTIONS (at K=20)")
    print("=" * 80)
    
    cands_only_tfidf = union_and_rank_candidates(val_s1_ids, all_tfidf_cands, {}, {}, k=20)
    m_tfidf = evaluate_candidate_recall(cands_only_tfidf, ground_truth_dict, len(df_s1_val), len(df_targets))
    
    cands_only_token = union_and_rank_candidates(val_s1_ids, {}, all_token_cands, {}, k=20)
    m_token = evaluate_candidate_recall(cands_only_token, ground_truth_dict, len(df_s1_val), len(df_targets))
    
    cands_only_addr = union_and_rank_candidates(val_s1_ids, {}, {}, all_addr_cands, k=20)
    m_addr = evaluate_candidate_recall(cands_only_addr, ground_truth_dict, len(df_s1_val), len(df_targets))
    
    cands_union = union_and_rank_candidates(val_s1_ids, all_tfidf_cands, all_token_cands, all_addr_cands, k=20)
    m_union = evaluate_candidate_recall(cands_union, ground_truth_dict, len(df_s1_val), len(df_targets))
    
    print(f"  Signal A Only (TF-IDF Cosine)   : Recall = {m_tfidf['candidate_recall']*100:.2f}% | Found = {m_tfidf['found_true_matches']:,}")
    print(f"  Signal B Only (Brand Token Idx) : Recall = {m_token['candidate_recall']*100:.2f}% | Found = {m_token['found_true_matches']:,}")
    print(f"  Signal C Only (Address Numbers) : Recall = {m_addr['candidate_recall']*100:.2f}% | Found = {m_addr['found_true_matches']:,}")
    print(f"  >>> MULTI-SIGNAL UNION (A + B + C): Recall = {m_union['candidate_recall']*100:.2f}% | Found = {m_union['found_true_matches']:,} <<<")
    print("=" * 80)
    print("TUNING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

"""
Hybrid Match Inference Pipeline (LightGBM + RapidFuzz + Transformer Embeddings)
Combines 10 lexical/numerical RapidFuzz features with 1 semantic Transformer Cosine Similarity feature.
Applies optimal F0.5 decision threshold and writes validated 'output/matching_results_hybrid.tsv'.
"""

import os
import gc
import sys
import json
import time
import argparse
import subprocess
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

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


def run_hybrid_prediction_pipeline(
    candidate_file: str = None,
    output_file: str = None,
    model_file: str = None,
    override_threshold: float = None,
    batch_size: int = 50000
):
    print("=" * 80)
    print("HYBRID MATCH PREDICTION PIPELINE (LightGBM + RapidFuzz + Transformer)")
    print("=" * 80)
    
    t0 = time.time()
    
    if candidate_file is None:
        candidate_file = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    if output_file is None:
        output_file = os.path.join(OUT_DIR, "matching_results_hybrid.tsv")
    if model_file is None:
        model_file = os.path.join(MODEL_DIR, "lgb_matcher.joblib")
        
    # Check embedding files
    s1_mmap_path = os.path.join(EMBED_DIR, "s1_embeddings.mmap")
    s1_map_path = os.path.join(EMBED_DIR, "s1_id_map.json")
    tgt_mmap_path = os.path.join(EMBED_DIR, "target_embeddings.mmap")
    tgt_map_path = os.path.join(EMBED_DIR, "target_id_map.json")
    
    has_embeddings = (
        os.path.isfile(s1_mmap_path) and os.path.isfile(s1_map_path) and
        os.path.isfile(tgt_mmap_path) and os.path.isfile(tgt_map_path)
    )
    
    if not has_embeddings:
        print("[Notice] Precomputed embedding files not found in 'embeddings/'.")
        print("Falling back to lexical-only scoring or run 'embed_entities.py' first.")
        
    # Load Model Artifact
    print(f"Loading LightGBM model from {os.path.basename(model_file)}...")
    artifact = joblib.load(model_file)
    model = artifact["model"]
    optimal_threshold = override_threshold if override_threshold is not None else artifact.get("optimal_threshold", 0.75)
    
    print(f"  Model Type: LightGBM (GBDT)")
    print(f"  Optimal Decision Threshold: {optimal_threshold:.2f}")
    
    # Load Embeddings Memmaps if available
    s1_vecs, s1_id_map = None, {}
    tgt_vecs, tgt_id_map = None, {}
    
    if has_embeddings:
        print("Loading disk memmaps for S1 and Target vectors...")
        with open(s1_map_path, "r", encoding="utf-8") as f:
            s1_id_map = json.load(f)
        with open(tgt_map_path, "r", encoding="utf-8") as f:
            tgt_id_map = json.load(f)
            
        dim = 384
        s1_vecs = np.memmap(s1_mmap_path, dtype=np.float16, mode="r", shape=(len(s1_id_map), dim))
        tgt_vecs = np.memmap(tgt_mmap_path, dtype=np.float16, mode="r", shape=(len(tgt_id_map), dim))
        print("  Memmaps loaded into virtual address space successfully.")

    # Collect Active Candidate IDs
    print(f"\nPass 1: Scanning {os.path.basename(candidate_file)} for active candidate targets...")
    t_scan = time.time()
    active_target_ids = set()
    with open(candidate_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                for cid in parts[1].split(","):
                    cid = cid.strip()
                    if cid:
                        active_target_ids.add(cid)
                        
    print(f"  Found {len(active_target_ids):,} active candidate targets in {time.time()-t_scan:.2f}s!")
    
    # Load S1 Strings
    print("\nLoading Normalized Test S1 Dataset...")
    s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    
    s1_names = dict(zip(df_s1["entity_id"], df_s1["clean_name"].fillna("")))
    s1_addrs = dict(zip(df_s1["entity_id"], df_s1["clean_address"].fillna("")))
    s1_nums = dict(zip(df_s1["entity_id"], df_s1["address_numbers"].fillna("")))
    s1_countries = dict(zip(df_s1["entity_id"], df_s1["country"].fillna("")))
    del df_s1
    gc.collect()
    
    # Load Target Strings
    print(f"Loading metadata for {len(active_target_ids):,} active targets from S2 and S3...")
    tgt_names, tgt_addrs, tgt_nums, tgt_countries = {}, {}, {}, {}
    s2_path = os.path.join(NORM_DIR, "test_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "test_source3_normalized.tsv")
    
    for path in [s2_path, s3_path]:
        df_t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df_active = df_t[df_t["entity_id"].isin(active_target_ids)]
        for _, row in df_active.iterrows():
            eid = row["entity_id"]
            tgt_names[eid] = row.get("clean_name", "")
            tgt_addrs[eid] = row.get("clean_address", "")
            tgt_nums[eid] = row.get("address_numbers", "")
            tgt_countries[eid] = row.get("country", "")
        del df_t, df_active
        gc.collect()
        
    del active_target_ids
    gc.collect()
    
    # Chunked Scoring
    print(f"\nProcessing candidate pairs in chunks of {batch_size:,} S1 entities...")
    total_s1 = 0
    total_singletons = 0
    total_matched_entities = 0
    total_pairs_scored = 0
    total_pairs_matched = 0
    
    with open(output_file, "w", encoding="utf-8") as f_out, \
         open(candidate_file, "r", encoding="utf-8") as f_cand:
        
        next(f_cand, None)
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        chunk_lines = []
        for line in f_cand:
            line = line.strip()
            if not line:
                continue
            chunk_lines.append(line)
            
            if len(chunk_lines) >= batch_size:
                s1_cnt, match_cnt, sing_cnt, scored_cnt, m_pair_cnt = _score_and_write_hybrid_chunk(
                    chunk_lines, model, optimal_threshold,
                    s1_names, s1_addrs, s1_nums, s1_countries,
                    tgt_names, tgt_addrs, tgt_nums, tgt_countries,
                    s1_vecs, s1_id_map, tgt_vecs, tgt_id_map,
                    f_out
                )
                total_s1 += s1_cnt
                total_matched_entities += match_cnt
                total_singletons += sing_cnt
                total_pairs_scored += scored_cnt
                total_pairs_matched += m_pair_cnt
                
                chunk_lines = []
                print(f"  Processed {total_s1:,} S1 entities ({total_matched_entities:,} matched, {total_singletons:,} singletons)...")
                
        if chunk_lines:
            s1_cnt, match_cnt, sing_cnt, scored_cnt, m_pair_cnt = _score_and_write_hybrid_chunk(
                chunk_lines, model, optimal_threshold,
                s1_names, s1_addrs, s1_nums, s1_countries,
                tgt_names, tgt_addrs, tgt_nums, tgt_countries,
                s1_vecs, s1_id_map, tgt_vecs, tgt_id_map,
                f_out
            )
            total_s1 += s1_cnt
            total_matched_entities += match_cnt
            total_singletons += sing_cnt
            total_pairs_scored += scored_cnt
            total_pairs_matched += m_pair_cnt
            
    total_elapsed = time.time() - t0
    sing_pct = (total_singletons / total_s1 * 100) if total_s1 > 0 else 0
    match_pct = (total_matched_entities / total_s1 * 100) if total_s1 > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"Hybrid Matching Results Complete in {total_elapsed/60:.2f} minutes!")
    print(f"  Total S1 Entities Processed: {total_s1:,}")
    print(f"  Singletons (Empty Match): {total_singletons:,} ({sing_pct:.1f}%)")
    print(f"  Matched S1 Entities: {total_matched_entities:,} ({match_pct:.1f}%)")
    print(f"  Total Confirmed Match Pairs: {total_pairs_matched:,}")
    print(f"  Output Saved to: {output_file}")
    print("=" * 80)


def _score_and_write_hybrid_chunk(
    chunk_lines: list,
    model,
    threshold: float,
    s1_names: dict, s1_addrs: dict, s1_nums: dict, s1_countries: dict,
    tgt_names: dict, tgt_addrs: dict, tgt_nums: dict, tgt_countries: dict,
    s1_vecs, s1_id_map: dict, tgt_vecs, tgt_id_map: dict,
    f_out
):
    pairs_to_score = []
    pair_meta = []
    s1_items = []
    
    for line in chunk_lines:
        parts = line.split("\t")
        s1_id = parts[0]
        cand_str = parts[1] if len(parts) > 1 else ""
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        
        s1_idx = len(s1_items)
        s1_items.append(s1_id)
        
        if not cands:
            continue
            
        n1 = s1_names.get(s1_id, "")
        a1 = s1_addrs.get(s1_id, "")
        num1 = s1_nums.get(s1_id, "")
        c1 = str(s1_countries.get(s1_id, "")).lower()
        
        # S1 embedding vector lookup
        s1_v = None
        if s1_vecs is not None and s1_id in s1_id_map:
            s1_v = np.array(s1_vecs[s1_id_map[s1_id]], dtype=np.float32)
            
        for tgt_id in cands:
            n2 = tgt_names.get(tgt_id, "")
            a2 = tgt_addrs.get(tgt_id, "")
            num2 = tgt_nums.get(tgt_id, "")
            c2 = str(tgt_countries.get(tgt_id, "")).lower()
            
            # Dot-product cosine similarity
            cosine_sim = 0.0
            if s1_v is not None and tgt_vecs is not None and tgt_id in tgt_id_map:
                tgt_v = np.array(tgt_vecs[tgt_id_map[tgt_id]], dtype=np.float32)
                cosine_sim = float(np.dot(s1_v, tgt_v))
                
            feats = extract_pair_features(n1, n2, a1, a2, num1, num2, c1, c2, cosine_sim)
            # If model was trained on 10 features, slice first 10, else use all 11
            n_features_model = getattr(model, "n_features_in_", 10)
            if n_features_model == 10:
                feats = feats[:10]
                
            pairs_to_score.append(feats)
            pair_meta.append((s1_idx, tgt_id))
            
    scored_cnt = len(pairs_to_score)
    s1_matches = defaultdict(list)
    match_pair_cnt = 0
    
    if pairs_to_score:
        X_batch = np.array(pairs_to_score, dtype=np.float32)
        probs = model.predict_proba(X_batch)[:, 1]
        
        for (s1_idx, tgt_id), prob in zip(pair_meta, probs):
            if prob >= threshold:
                s1_matches[s1_idx].append((tgt_id, float(prob)))
                match_pair_cnt += 1
                
    matched_cnt = 0
    sing_cnt = 0
    for s1_idx, s1_id in enumerate(s1_items):
        matches = s1_matches.get(s1_idx, [])
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            matched_str = ",".join(t[0] for t in matches)
            f_out.write(f"{s1_id}\t{matched_str}\n")
            matched_cnt += 1
        else:
            f_out.write(f"{s1_id}\t\n")
            sing_cnt += 1
            
    return len(chunk_lines), matched_cnt, sing_cnt, scored_cnt, match_pair_cnt


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Match Inference Pipeline")
    parser.add_argument("--candidate-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--model-file", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()
    
    run_hybrid_prediction_pipeline(
        candidate_file=args.candidate_file,
        output_file=args.output_file,
        model_file=args.model_file,
        override_threshold=args.threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

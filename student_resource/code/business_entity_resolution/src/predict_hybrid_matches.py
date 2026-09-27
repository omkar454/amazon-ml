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
import lightgbm as lgb
import sklearn
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


def load_normalized_tsv_records(file_path: str, active_ids: set = None):
    """Ultra-fast streaming loader for normalized TSV files with optional ID filtering (compact tuple dict)."""
    meta = {}
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        header_line = next(f, None)
        if not header_line:
            return meta
        header = header_line.rstrip("\r\n").split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("clean_name") if "clean_name" in header else 1
        addr_idx = header.index("clean_address") if "clean_address" in header else 3
        num_idx = header.index("address_numbers") if "address_numbers" in header else 4
        ctry_idx = header.index("country") if "country" in header else 6
        
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[id_idx]
            if active_ids is None or eid in active_ids:
                name = parts[name_idx] if name_idx < len(parts) else ""
                addr = parts[addr_idx] if addr_idx < len(parts) else ""
                nums = parts[num_idx] if num_idx < len(parts) else ""
                ctry = parts[ctry_idx] if ctry_idx < len(parts) else ""
                meta[eid] = (name, addr, nums, ctry.lower())
    return meta


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
        hybrid_path = os.path.join(MODEL_DIR, "lgb_hybrid_matcher.joblib")
        std_path = os.path.join(MODEL_DIR, "lgb_matcher.joblib")
        model_file = hybrid_path if os.path.isfile(hybrid_path) else std_path
        
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
    
    # 1. Load S1 Strings and Index Mapping
    print("\n[1/3] Loading Normalized Test S1 Dataset...")
    t_s1 = time.time()
    s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
    s1_meta = load_normalized_tsv_records(s1_path)
    s1_id_map = {eid: idx for idx, eid in enumerate(s1_meta.keys())}
    print(f"  Loaded {len(s1_meta):,} S1 query entities in {time.time()-t_s1:.2f}s.")
    
    # 2. Load Embeddings Mapping & Target Strings
    s1_vecs, tgt_vecs, tgt_id_map = None, None, {}
    if has_embeddings:
        print("\n[2/3] Loading target embedding ID map...")
        t_map = time.time()
        with open(tgt_map_path, "r", encoding="utf-8") as f:
            tgt_id_map = json.load(f)
        print(f"  Loaded {len(tgt_id_map):,} target ID mappings in {time.time()-t_map:.2f}s.")
        
        dim = 384
        s1_vecs = np.memmap(s1_mmap_path, dtype=np.float16, mode="r", shape=(len(s1_id_map), dim))
        tgt_vecs = np.memmap(tgt_mmap_path, dtype=np.float16, mode="r", shape=(len(tgt_id_map), dim))
        print("  Memmaps loaded into virtual address space successfully.")
    
    # 3. Load Target Metadata for Active Target IDs
    print(f"\n[3/3] Loading metadata for {len(tgt_id_map):,} active targets from S2 and S3...")
    t_tgt = time.time()
    tgt_meta = {}
    s2_path = os.path.join(NORM_DIR, "test_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "test_source3_normalized.tsv")
    
    for path in [s2_path, s3_path]:
        print(f"  Filtering {os.path.basename(path)}...")
        meta = load_normalized_tsv_records(path, tgt_id_map)
        tgt_meta.update(meta)
        del meta
        gc.collect()
        
    print(f"  Loaded {len(tgt_meta):,} active target records in {time.time()-t_tgt:.2f}s!")
    
    # Chunked Scoring
    print(f"\nProcessing candidate pairs in chunks of {batch_size:,} S1 entities...")
    total_s1 = 0
    total_singletons = 0
    total_matched_entities = 0
    total_pairs_scored = 0
    total_pairs_matched = 0
    
    with open(output_file, "w", encoding="utf-8", newline="\n") as f_out, \
         open(candidate_file, "r", encoding="utf-8", newline="") as f_cand:
        
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
                    s1_meta, tgt_meta,
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
                s1_meta, tgt_meta,
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
    
    # Run Submission Validator
    val_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    if os.path.isfile(val_script):
        print("\n" + "=" * 80)
        print("RUNNING SUBMISSION VALIDATOR (validate_submission.py)")
        print("=" * 80)
        raw_test_dir = os.path.join(BASE_DIR, "dataset", "test")
        cmd = [
            sys.executable,
            val_script,
            "--matching", output_file,
            "--candidate", candidate_file,
            "--test-dir", raw_test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Stderr:", res.stderr)
        if res.returncode == 0:
            print(">>> VALIDATOR STATUS: PASS (Exit Code 0) <<<")


def _score_and_write_hybrid_chunk(
    chunk_lines: list,
    model,
    threshold: float,
    s1_meta: dict,
    tgt_meta: dict,
    s1_vecs, s1_id_map: dict, tgt_vecs, tgt_id_map: dict,
    f_out
):
    # 1. Parse lines
    parsed = []
    needed_tids = set()
    s1_items = []
    
    for line in chunk_lines:
        parts = line.split("\t")
        s1_id = parts[0]
        cand_str = parts[1] if len(parts) > 1 else ""
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        s1_idx = len(s1_items)
        s1_items.append(s1_id)
        if cands:
            parsed.append((s1_idx, s1_id, cands))
            needed_tids.update(cands)
            
    # 2. Batch-prefetch target vectors in sequential disk order (Ultra fast!)
    tgt_vec_cache = {}
    if tgt_vecs is not None and tgt_id_map:
        valid_tids = [tid for tid in needed_tids if tid in tgt_id_map]
        if valid_tids:
            indices = [tgt_id_map[tid] for tid in valid_tids]
            order = np.argsort(indices)
            sorted_indices = [indices[o] for o in order]
            sorted_tids = [valid_tids[o] for o in order]
            
            # Slice in sequential disk order
            batch_vecs = np.array(tgt_vecs[sorted_indices], dtype=np.float32)
            for tid, v in zip(sorted_tids, batch_vecs):
                tgt_vec_cache[tid] = v
                del v
            del batch_vecs, sorted_indices, sorted_tids, order, valid_tids
            
    # 3. Extract features
    pairs_to_score = []
    pair_meta = []
    empty_tuple = ("", "", "", "")
    n_features_model = getattr(model, "n_features_in_", 11)
    
    for s1_idx, s1_id, cands in parsed:
        n1, a1, num1, c1 = s1_meta.get(s1_id, empty_tuple)
        s1_v = None
        if s1_vecs is not None and s1_id in s1_id_map:
            s1_v = np.array(s1_vecs[s1_id_map[s1_id]], dtype=np.float32)
            
        for tgt_id in cands:
            n2, a2, num2, c2 = tgt_meta.get(tgt_id, empty_tuple)
            
            cosine_sim = 0.0
            if s1_v is not None and tgt_id in tgt_vec_cache:
                cosine_sim = float(np.dot(s1_v, tgt_vec_cache[tgt_id]))
                
            feats = extract_pair_features(n1, n2, a1, a2, num1, num2, c1, c2, cosine_sim)
            if n_features_model == 10:
                feats = feats[:10]
                
            pairs_to_score.append(feats)
            pair_meta.append((s1_idx, tgt_id))
            
    del tgt_vec_cache, parsed, needed_tids
    
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

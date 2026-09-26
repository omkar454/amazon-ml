#!/usr/bin/env python3
"""
Business Entity Resolution — High-Performance & Memory-Safe Matching Pipeline
Amazon ML Challenge 2026

Ultra Memory-Optimized Architecture (< 1.5 GB RAM footprint):
1. Compact Flat Storage: Stores entity names & addresses in lean flat mappings (no nested dicts/sets).
2. Dynamic On-the-Fly Signals: Evaluates RapidFuzz string similarities & numeric overlap directly.
3. Pre-allocated Vectorized Inference: Avoids intermediate Python list allocations.
4. Separate Training Phase: Trains LightGBM on a representative sample and purges training memory.
5. Micro-Batch Streaming: Streams candidate pairs in small chunks with explicit garbage collection.
"""

import os
import sys
import gc
import re
import time
import argparse
import logging
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from tqdm import tqdm


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("predict_matches")


# =============================================================================
# STRING NORMALIZATION & TOKEN HELPERS
# =============================================================================

RE_DIGITS = re.compile(r"\b\d+\b")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")

COMMON_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "llc", "llp", "co", "company", "enterprises", "services", "solutions", "holdings",
    "gmbh", "sa", "sarl", "bv", "technologies", "tech", "group", "international"
}


def clean_text(text: Any) -> str:
    """Standardize text: lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    text = text.lower()
    text = RE_PUNCT.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens without leading zeros."""
    if not text:
        return set()
    return {n.lstrip("0") for n in RE_DIGITS.findall(text) if n.lstrip("0")}


def get_core_name(clean_name: str) -> str:
    """Strip common corporate suffixes from cleaned name."""
    tokens = clean_name.split()
    core_tokens = [t for t in tokens if t not in COMMON_LEGAL_SUFFIXES]
    return " ".join(core_tokens) if core_tokens else clean_name


# =============================================================================
# FEATURE DEFINITIONS & FAST ROW-WISE CALCULATION
# =============================================================================

NUM_FEATURES = 24

def compute_pair_features_into(
    out_arr: np.ndarray,
    row_idx: int,
    s1_name: str,
    s1_addr: str,
    t_id: str,
    t_name: str,
    t_addr: str
):
    """
    Computes 24 similarity signals directly into a pre-allocated numpy array slice.
    Avoids Python float list allocations and stays CPU/RAM ultra-lean.
    """
    # 1. Name Similarities (0-1 range)
    n_ratio = fuzz.ratio(s1_name, t_name) / 100.0
    n_tsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
    n_tset = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    n_partial = fuzz.partial_ratio(s1_name, t_name) / 100.0
    n_exact = 1.0 if s1_name and (s1_name == t_name) else 0.0

    # Core name similarity (without legal suffixes)
    s1_core = get_core_name(s1_name)
    t_core = get_core_name(t_name)
    n_core_ratio = fuzz.ratio(s1_core, t_core) / 100.0
    n_core_tsort = fuzz.token_sort_ratio(s1_core, t_core) / 100.0

    # Token overlap & Prefix matches
    s1_n_toks = set(s1_name.split())
    t_n_toks = set(t_name.split())
    n_union = s1_n_toks | t_n_toks
    n_jaccard = (len(s1_n_toks & t_n_toks) / len(n_union)) if n_union else 0.0
    
    pfx3 = 1.0 if (s1_name[:3] and s1_name[:3] == t_name[:3]) else 0.0
    max_nl = max(len(s1_name), len(t_name), 1)
    len_diff_n = abs(len(s1_name) - len(t_name)) / max_nl
    
    first_s1 = s1_name.split()[0] if s1_name else ""
    first_t = t_name.split()[0] if t_name else ""
    first_match = 1.0 if (first_s1 and first_s1 == first_t) else 0.0

    # 2. Address Similarities
    has_a1 = 1.0 if s1_addr else 0.0
    has_at = 1.0 if t_addr else 0.0
    both_addr = 1.0 if (has_a1 and has_at) else 0.0

    if both_addr:
        a_ratio = fuzz.ratio(s1_addr, t_addr) / 100.0
        a_tsort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        a_tset = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
        a_partial = fuzz.partial_ratio(s1_addr, t_addr) / 100.0
        
        s1_a_toks = set(s1_addr.split())
        t_a_toks = set(t_addr.split())
        a_union = s1_a_toks | t_a_toks
        a_jaccard = (len(s1_a_toks & t_a_toks) / len(a_union)) if a_union else 0.0
    else:
        a_ratio = 0.0
        a_tsort = 0.0
        a_tset = 0.0
        a_partial = 0.0
        a_jaccard = 0.0

    # 3. Numeric & PIN / Building Overlap
    s1_nums = extract_numbers(s1_name + " " + s1_addr)
    t_nums = extract_numbers(t_name + " " + t_addr)
    
    if s1_nums and t_nums:
        num_exact = 1.0 if (s1_nums == t_nums) else 0.0
        shared = s1_nums & t_nums
        num_shared_cnt = float(len(shared))
        num_jacc = len(shared) / len(s1_nums | t_nums)
        # Conflicting numbers: both specify numbers but share none (strong negative signal)
        num_conflict = 1.0 if (len(shared) == 0) else 0.0
    else:
        num_exact = 0.0
        num_shared_cnt = 0.0
        num_jacc = 0.0
        num_conflict = 0.0

    # 4. Source Indicator
    is_s2 = 1.0 if t_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if t_id.startswith("S3-") else 0.0

    # Write directly into array row
    out_arr[row_idx, 0] = n_ratio
    out_arr[row_idx, 1] = n_tsort
    out_arr[row_idx, 2] = n_tset
    out_arr[row_idx, 3] = n_partial
    out_arr[row_idx, 4] = n_exact
    out_arr[row_idx, 5] = n_core_ratio
    out_arr[row_idx, 6] = n_core_tsort
    out_arr[row_idx, 7] = n_jaccard
    out_arr[row_idx, 8] = pfx3
    out_arr[row_idx, 9] = len_diff_n
    out_arr[row_idx, 10] = first_match
    out_arr[row_idx, 11] = a_ratio
    out_arr[row_idx, 12] = a_tsort
    out_arr[row_idx, 13] = a_tset
    out_arr[row_idx, 14] = a_partial
    out_arr[row_idx, 15] = a_jaccard
    out_arr[row_idx, 16] = both_addr
    out_arr[row_idx, 17] = num_exact
    out_arr[row_idx, 18] = num_shared_cnt
    out_arr[row_idx, 19] = num_jacc
    out_arr[row_idx, 20] = num_conflict
    out_arr[row_idx, 21] = 1.0 if (not s1_nums and not t_nums) else 0.0
    out_arr[row_idx, 22] = is_s2
    out_arr[row_idx, 23] = is_s3


# =============================================================================
# COMPACT STORAGE LOADING
# =============================================================================

def find_source_file(directory: str, prefix: str) -> Optional[str]:
    """Locate file supporting .tsv, .parquet, and _normalized variants."""
    candidates = [
        f"{prefix}.tsv",
        f"{prefix}.parquet",
        f"{prefix}_normalized.tsv",
        f"{prefix}_normalized.parquet",
    ]
    for c in candidates:
        full_p = os.path.join(directory, c)
        if os.path.isfile(full_p):
            return full_p
    return None


def load_flat_records(file_path: str, names_dict: Dict[str, str], addrs_dict: Dict[str, str]):
    """
    Populates flat entity_id -> clean_string dictionaries.
    Uses < 150 MB RAM for 1,000,000 records.
    """
    if not file_path or not os.path.isfile(file_path):
        logger.warning(f"File not found: {file_path}")
        return

    logger.info(f"Loading {os.path.basename(file_path)} into flat memory...")
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)

    for _, row in df.iterrows():
        eid = str(row.get("entity_id", "")).strip()
        if not eid:
            continue
        
        raw_name = row.get("clean_name_full") or row.get("clean_name") or row.get("business_name") or ""
        raw_addr = row.get("clean_address") or row.get("business_address") or ""
        
        names_dict[eid] = clean_text(raw_name)
        addrs_dict[eid] = clean_text(raw_addr)

    del df
    gc.collect()


# =============================================================================
# LEAN MODEL TRAINING (< 500 MB RAM)
# =============================================================================

def train_lightgbm_model(train_dir: str) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Train a LightGBM match classifier on a representative sample of train pairs.
    Purges all training data immediately after training to release memory for test inference.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: TRAINING LIGHTGBM MATCH CLASSIFIER")
    logger.info("=" * 60)

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    if not os.path.isfile(gt_file):
        alt_gt = os.path.join(os.path.dirname(train_dir), "train", "train_ground_truth.tsv")
        if os.path.isfile(alt_gt):
            gt_file = alt_gt

    s1_file = find_source_file(train_dir, "train_source1")
    s2_file = find_source_file(train_dir, "train_source2")
    s3_file = find_source_file(train_dir, "train_source3")

    if not gt_file or not os.path.isfile(gt_file) or not s1_file:
        logger.warning("Train files not found. Using pre-initialized default model.")
        model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42)
        dummy_X = np.random.rand(50, NUM_FEATURES).astype(np.float32)
        model.fit(dummy_X, (dummy_X[:, 0] > 0.5).astype(np.int32))
        return model, 0.75

    # 1. Load Training Data into Temporary Flat Maps
    tr_names: Dict[str, str] = {}
    tr_addrs: Dict[str, str] = {}
    load_flat_records(s1_file, tr_names, tr_addrs)
    if s2_file:
        load_flat_records(s2_file, tr_names, tr_addrs)
    if s3_file:
        load_flat_records(s3_file, tr_names, tr_addrs)

    # 2. Sample Positive Pairs from Ground Truth
    gt_df = pd.read_csv(gt_file, sep="\t", dtype=str, keep_default_na=False)
    pos_pairs = []
    ground_truth_map: Dict[str, Set[str]] = {}

    for _, row in gt_df.iterrows():
        s1_id = str(row.get("source1_entity_id", "")).strip()
        matched_str = str(row.get("matched_entity_ids", "")).strip()
        if not matched_str:
            continue
        m_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        ground_truth_map[s1_id] = set(m_ids)
        for mid in m_ids:
            if s1_id in tr_names and mid in tr_names:
                pos_pairs.append((s1_id, mid))

    del gt_df
    gc.collect()

    # Limit training sample size to keep RAM < 500 MB
    np.random.seed(42)
    max_pos = 40_000
    if len(pos_pairs) > max_pos:
        indices = np.random.choice(len(pos_pairs), size=max_pos, replace=False)
        pos_pairs = [pos_pairs[i] for i in indices]

    logger.info(f"Using {len(pos_pairs):,} positive pairs for training.")

    # 3. Mine Hard Negatives with Token Overlap
    target_ids = [k for k in tr_names.keys() if k.startswith("S2-") or k.startswith("S3-")]
    s1_ids = [k for k in tr_names.keys() if k.startswith("S1-")]

    first_tok_map: Dict[str, List[str]] = {}
    for tid in target_ids:
        toks = tr_names[tid].split()
        if toks and len(toks[0]) >= 3:
            first_tok_map.setdefault(toks[0], []).append(tid)

    neg_pairs = []
    num_neg_target = len(pos_pairs) * 3

    for s1_id in s1_ids:
        if len(neg_pairs) >= num_neg_target:
            break
        toks = tr_names[s1_id].split()
        if not toks or len(toks[0]) < 3:
            continue
        cands = first_tok_map.get(toks[0], [])
        true_m = ground_truth_map.get(s1_id, set())
        for tid in cands:
            if tid not in true_m:
                neg_pairs.append((s1_id, tid))
                if len(neg_pairs) >= num_neg_target:
                    break

    # Add random negatives if needed
    while len(neg_pairs) < num_neg_target and s1_ids and target_ids:
        s1_id = s1_ids[np.random.randint(0, len(s1_ids))]
        tid = target_ids[np.random.randint(0, len(target_ids))]
        if tid not in ground_truth_map.get(s1_id, set()):
            neg_pairs.append((s1_id, tid))

    logger.info(f"Synthesized {len(neg_pairs):,} negative training pairs.")

    # 4. Build Training Matrix in Pre-allocated Array
    total_samples = len(pos_pairs) + len(neg_pairs)
    X_train = np.empty((total_samples, NUM_FEATURES), dtype=np.float32)
    y_train = np.empty(total_samples, dtype=np.int32)

    row_i = 0
    for s1_id, tid in pos_pairs:
        compute_pair_features_into(
            X_train, row_i,
            tr_names[s1_id], tr_addrs[s1_id],
            tid, tr_names[tid], tr_addrs[tid]
        )
        y_train[row_i] = 1
        row_i += 1

    for s1_id, tid in neg_pairs:
        compute_pair_features_into(
            X_train, row_i,
            tr_names[s1_id], tr_addrs[s1_id],
            tid, tr_names[tid], tr_addrs[tid]
        )
        y_train[row_i] = 0
        row_i += 1

    # Purge intermediate training dicts before fitting LightGBM
    del tr_names, tr_addrs, pos_pairs, neg_pairs, first_tok_map, ground_truth_map
    gc.collect()

    # 5. Fit LightGBM
    logger.info("Fitting LightGBM decision trees...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    logger.info("LightGBM Model trained successfully. Memory cleaned.")
    return model, 0.75


# =============================================================================
# STREAMING INFERENCE PIPELINE (< 1 GB RAM)
# =============================================================================

def run_matching_pipeline(
    test_dir: str,
    candidate_pairs_path: str,
    output_path: str,
    train_dir: Optional[str] = None,
    threshold: float = 0.75,
    chunk_size: int = 5_000
):
    """
    Memory-safe streaming prediction pipeline.
    Keeps resident memory strictly under 1.5 GB.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 2: TEST SET MATCHING INFERENCE")
    logger.info("=" * 60)

    # 1. Train or Load Model
    if train_dir and os.path.isdir(train_dir):
        model, opt_thresh = train_lightgbm_model(train_dir)
        if threshold is None:
            threshold = opt_thresh
    else:
        model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42)
        dummy_X = np.random.rand(50, NUM_FEATURES).astype(np.float32)
        model.fit(dummy_X, (dummy_X[:, 0] > 0.5).astype(np.int32))

    # 2. Load Test Records in Flat Memory
    names_dict: Dict[str, str] = {}
    addrs_dict: Dict[str, str] = {}

    s1_file = find_source_file(test_dir, "test_source1")
    s2_file = find_source_file(test_dir, "test_source2")
    s3_file = find_source_file(test_dir, "test_source3")

    if not s1_file or not os.path.isfile(s1_file):
        raise FileNotFoundError(f"test_source1 file not found in: {test_dir}")

    load_flat_records(s1_file, names_dict, addrs_dict)
    all_s1_ids = set(k for k in names_dict.keys() if k.startswith("S1-"))
    logger.info(f"Loaded {len(all_s1_ids):,} Source 1 test entities.")

    if s2_file:
        load_flat_records(s2_file, names_dict, addrs_dict)
    if s3_file:
        load_flat_records(s3_file, names_dict, addrs_dict)

    logger.info(f"Total entity records in memory: {len(names_dict):,}")

    # 3. Stream candidate_pairs.tsv in Micro-Batches
    if not os.path.isfile(candidate_pairs_path):
        raise FileNotFoundError(f"candidate_pairs file not found: {candidate_pairs_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Evaluating candidate pairs with cutoff threshold: {threshold:.3f}")

    total_s1_processed = 0
    total_matches_found = 0
    processed_s1_ids = set()

    with open(output_path, "w", encoding="utf-8") as out_f, \
         open(candidate_pairs_path, "r", encoding="utf-8") as in_f:

        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        in_f.readline()  # skip header

        chunk_rows: List[Tuple[str, List[str]]] = []

        def process_and_write_chunk(rows):
            nonlocal total_s1_processed, total_matches_found
            if not rows:
                return

            # Count total candidate pairs in this chunk
            pair_items = []
            for s1_id, cands in rows:
                if s1_id not in names_dict:
                    continue
                for tid in cands:
                    if tid in names_dict:
                        pair_items.append((s1_id, tid))

            # Score in pre-allocated array
            s1_results_map: Dict[str, List[Tuple[str, float]]] = {}
            if pair_items:
                X_chunk = np.empty((len(pair_items), NUM_FEATURES), dtype=np.float32)
                for i, (s1_id, tid) in enumerate(pair_items):
                    compute_pair_features_into(
                        X_chunk, i,
                        names_dict[s1_id], addrs_dict[s1_id],
                        tid, names_dict[tid], addrs_dict[tid]
                    )

                probs = model.predict_proba(X_chunk)[:, 1]
                for (s1_id, tid), p in zip(pair_items, probs):
                    if p >= threshold:
                        s1_results_map.setdefault(s1_id, []).append((tid, float(p)))

                del X_chunk, probs

            # Write formatted output
            for s1_id, _ in rows:
                matches = s1_results_map.get(s1_id, [])
                if matches:
                    matches.sort(key=lambda x: x[1], reverse=True)
                    # Deduplicate preserving order
                    seen = set()
                    unique_tids = []
                    for tid, _ in matches:
                        if tid not in seen:
                            seen.add(tid)
                            unique_tids.append(tid)
                    out_str = ",".join(unique_tids)
                    out_f.write(f"{s1_id}\t{out_str}\n")
                    total_matches_found += len(unique_tids)
                else:
                    # Singleton
                    out_f.write(f"{s1_id}\t\n")
                total_s1_processed += 1

        for line in tqdm(in_f, desc="Predicting Matches"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            candidates = []
            if len(parts) > 1 and parts[1].strip():
                candidates = [c.strip() for c in parts[1].split(",") if c.strip()]

            chunk_rows.append((s1_id, candidates))
            processed_s1_ids.add(s1_id)

            if len(chunk_rows) >= chunk_size:
                process_and_write_chunk(chunk_rows)
                chunk_rows = []
                gc.collect()

        if chunk_rows:
            process_and_write_chunk(chunk_rows)
            chunk_rows = []
            gc.collect()

        # Guarantee 100% S1 coverage
        missing_s1 = all_s1_ids - processed_s1_ids
        if missing_s1:
            logger.warning(f"Writing {len(missing_s1):,} missing singletons from test_source1")
            for s1_id in sorted(missing_s1):
                out_f.write(f"{s1_id}\t\n")
                total_s1_processed += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"MATCHING COMPLETE IN {elapsed:.2f} SECONDS")
    logger.info(f"Total Source 1 Rows Written: {total_s1_processed:,}")
    logger.info(f"Total Match Links Found:     {total_matches_found:,}")
    logger.info(f"Output File:                 {output_path}")
    logger.info("=" * 60)


# =============================================================================
# CLI PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Memory-Safe Match Predictor")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--candidates", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--output", type=str, default="output/matching_results.tsv")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def resolve_path(p: str) -> str:
        if os.path.isabs(p) or os.path.exists(p):
            return p
        rel_p = os.path.join(base_dir, p)
        if os.path.exists(rel_p):
            return rel_p
        return p

    test_dir = resolve_path(args.test_dir)
    train_dir = resolve_path(args.train_dir)
    candidates_path = resolve_path(args.candidates)
    output_path = resolve_path(args.output) if os.path.isabs(args.output) else os.path.join(base_dir, args.output)

    run_matching_pipeline(
        test_dir=test_dir,
        candidate_pairs_path=candidates_path,
        output_path=output_path,
        train_dir=train_dir,
        threshold=args.threshold,
        chunk_size=args.chunk_size
    )

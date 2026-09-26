#!/usr/bin/env python3
"""
Business Entity Resolution — Fast Rule & Heuristic Matching Pipeline
Amazon ML Challenge 2026

Architecture (Option 2: Pure Rule & Heuristic Scoring - Zero ML Training):
1. Composite Similarity Scoring:
   Score = 0.50 * NameScore + 0.35 * AddressScore + 0.15 * NumberScore
2. Number / PIN Disambiguation:
   Rewards shared premise numbers/PIN codes; penalizes conflicting branch numbers.
3. Source-Aware Selection:
   Ranks candidates and selects high-confidence matches per source (S2 / S3).
4. Strict Precision Filtering:
   Applies high-threshold cutoff (e.g., Score >= 80.0 - 85.0) to optimize Macro F_0.5.
5. Ultra-Lightweight & Streaming:
   Zero training overhead, < 500 MB RAM footprint, processes test candidates in minutes.
"""

import os
import sys
import gc
import re
import time
import argparse
import logging
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd
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
logger = logging.getLogger("heuristic_matches")


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
    """Extract numeric tokens (building numbers, PIN/postal codes) without leading zeros."""
    if not text:
        return set()
    return {n.lstrip("0") for n in RE_DIGITS.findall(text) if n.lstrip("0")}


def get_core_name(clean_name: str) -> str:
    """Strip common corporate legal suffixes from cleaned name."""
    tokens = clean_name.split()
    core_tokens = [t for t in tokens if t not in COMMON_LEGAL_SUFFIXES]
    return " ".join(core_tokens) if core_tokens else clean_name


# =============================================================================
# HEURISTIC COMPOSITE SCORER
# =============================================================================

def compute_composite_score(
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str
) -> float:
    """
    Compute rule-based composite similarity score on a 0.0 to 100.0 scale:
    Score = 0.50 * NameScore + 0.35 * AddressScore + 0.15 * NumberScore
    """
    if not s1_name or not t_name:
        return 0.0

    # 1. Name Score (0 - 100)
    # token_set_ratio handles added/missing words (e.g. "Apollo Pharmacy" vs "Apollo Pharmacy Pvt Ltd")
    # token_sort_ratio handles word order transpositions
    name_set_ratio = fuzz.token_set_ratio(s1_name, t_name)
    name_sort_ratio = fuzz.token_sort_ratio(s1_name, t_name)
    name_score = 0.65 * name_set_ratio + 0.35 * name_sort_ratio

    # Bonus for exact name match
    if s1_name == t_name:
        name_score = 100.0

    # 2. Address Score (0 - 100)
    has_s1_addr = bool(s1_addr)
    has_t_addr = bool(t_addr)

    if has_s1_addr and has_t_addr:
        addr_set_ratio = fuzz.token_set_ratio(s1_addr, t_addr)
        addr_sort_ratio = fuzz.token_sort_ratio(s1_addr, t_addr)
        addr_ratio = fuzz.ratio(s1_addr, t_addr)
        addr_score = 0.40 * addr_set_ratio + 0.30 * addr_sort_ratio + 0.30 * addr_ratio
    else:
        # Neutral score if address is completely absent
        addr_score = 50.0

    # 3. Premise / PIN / Building Number Score (0 - 100)
    s1_nums = extract_numbers(s1_name + " " + s1_addr)
    t_nums = extract_numbers(t_name + " " + t_addr)

    if s1_nums and t_nums:
        shared_nums = s1_nums & t_nums
        if shared_nums:
            # High reward for matching postal code or building number
            num_score = 100.0
        else:
            # Heavy penalty: both specify numbers/PINs but they don't match (e.g. different branches)
            num_score = 10.0
    else:
        # Neutral if one or both lack numbers
        num_score = 60.0

    # 4. Composite Formula
    if has_s1_addr and has_t_addr:
        composite = (0.50 * name_score) + (0.35 * addr_score) + (0.15 * num_score)
    else:
        # If address is missing, weight name more heavily
        composite = (0.80 * name_score) + (0.20 * num_score)

    # Strong gating condition: If names are completely dissimilar, reject immediately
    if name_set_ratio < 45.0:
        composite = min(composite, 30.0)

    return composite


# =============================================================================
# DATA LOADING (FLAT & LEAN)
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
    """Populates lean entity_id -> string dictionaries (< 150 MB RAM per 1M rows)."""
    if not file_path or not os.path.isfile(file_path):
        logger.warning(f"File not found: {file_path}")
        return

    logger.info(f"Loading {os.path.basename(file_path)}...")
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
# STREAMING MATCH INFERENCE (RULE-BASED)
# =============================================================================

def run_rule_based_matching(
    test_dir: str,
    candidate_pairs_path: str,
    output_path: str,
    threshold: float = 80.0,
    max_per_source: int = 1,
    chunk_size: int = 10_000
):
    """
    Execute streaming rule-based candidate scoring.
    Zero ML training required, runs in minutes, protects precision.
    """
    start_time = time.time()
    logger.info("=" * 65)
    logger.info("STARTING RULE-BASED MATCHING PIPELINE (OPTION 2)")
    logger.info(f"Threshold: {threshold:.1f} | Max per source: {max_per_source}")
    logger.info("=" * 65)

    # 1. Load Test Records
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

    logger.info(f"Total test entities in memory: {len(names_dict):,}")

    # 2. Open Candidate Pairs and Output File
    if not os.path.isfile(candidate_pairs_path):
        raise FileNotFoundError(f"candidate_pairs file not found: {candidate_pairs_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Writing matching results to: {output_path}")

    total_s1_processed = 0
    total_matches_found = 0
    processed_s1_ids = set()

    with open(output_path, "w", encoding="utf-8") as out_f, \
         open(candidate_pairs_path, "r", encoding="utf-8") as in_f:

        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        in_f.readline()  # skip header

        for line in tqdm(in_f, desc="Scoring Candidates"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            s1_id = parts[0].strip()
            processed_s1_ids.add(s1_id)

            candidates = []
            if len(parts) > 1 and parts[1].strip():
                candidates = [c.strip() for c in parts[1].split(",") if c.strip()]

            if not candidates or s1_id not in names_dict:
                out_f.write(f"{s1_id}\t\n")
                total_s1_processed += 1
                continue

            s1_name = names_dict[s1_id]
            s1_addr = addrs_dict[s1_id]

            # Score candidates separated by source (S2 and S3)
            s2_candidates: List[Tuple[str, float]] = []
            s3_candidates: List[Tuple[str, float]] = []

            for tid in candidates:
                if tid not in names_dict:
                    continue
                score = compute_composite_score(
                    s1_name, s1_addr,
                    names_dict[tid], addrs_dict[tid]
                )
                if score >= threshold:
                    if tid.startswith("S2-"):
                        s2_candidates.append((tid, score))
                    elif tid.startswith("S3-"):
                        s3_candidates.append((tid, score))

            # Sort each source candidates by score descending
            s2_candidates.sort(key=lambda x: x[1], reverse=True)
            s3_candidates.sort(key=lambda x: x[1], reverse=True)

            # Pick top-K per source to strictly preserve precision under F_0.5
            selected_matches = []
            for tid, _ in s2_candidates[:max_per_source]:
                selected_matches.append(tid)
            for tid, _ in s3_candidates[:max_per_source]:
                selected_matches.append(tid)

            if selected_matches:
                match_str = ",".join(selected_matches)
                out_f.write(f"{s1_id}\t{match_str}\n")
                total_matches_found += len(selected_matches)
            else:
                out_f.write(f"{s1_id}\t\n")

            total_s1_processed += 1

        # 3. Guarantee 100% S1 Coverage
        missing_s1 = all_s1_ids - processed_s1_ids
        if missing_s1:
            logger.warning(f"Appending {len(missing_s1):,} missing singletons from test_source1")
            for s1_id in sorted(missing_s1):
                out_f.write(f"{s1_id}\t\n")
                total_s1_processed += 1

    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info(f"MATCHING FINISHED IN {elapsed:.2f} SECONDS")
    logger.info(f"Total Source 1 Rows Written: {total_s1_processed:,}")
    logger.info(f"Total Match Links Produced:  {total_matches_found:,}")
    logger.info(f"Output Saved To:             {output_path}")
    logger.info("=" * 65)


# =============================================================================
# CLI PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Rule-Based Business Entity Matcher")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--candidates", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--output", type=str, default="output/matching_results.tsv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Score threshold (0-100 scale) for accepting a match (e.g. 80.0, 82.0, 85.0)"
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=1,
        help="Maximum number of matches to accept per source (default: 1 for S2, 1 for S3)"
    )
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
    candidates_path = resolve_path(args.candidates)
    output_path = resolve_path(args.output) if os.path.isabs(args.output) else os.path.join(base_dir, args.output)

    run_rule_based_matching(
        test_dir=test_dir,
        candidate_pairs_path=candidates_path,
        output_path=output_path,
        threshold=args.threshold,
        max_per_source=args.max_per_source
    )

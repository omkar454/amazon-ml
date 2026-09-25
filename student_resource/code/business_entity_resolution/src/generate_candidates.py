"""
Final Candidate Pairs Generation Pipeline
Runs full country-partitioned multi-signal candidate generation across all 1.73M test Source 1 records
(US, India, and France) and outputs validated 'output/candidate_pairs.tsv'.
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
from blocking import (
    country_partition,
    generate_3gram_tfidf_candidates,
    generate_token_candidates,
    generate_address_number_candidates,
    union_and_rank_candidates
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)

def generate_test_candidates(k: int = 20, min_sim: float = 0.20):
    print("=" * 80)
    print("TEST CANDIDATE GENERATION ENGINE (candidate_pairs.tsv)")
    print(f"Top-K Hyperparameter: {k}")
    print(f"Minimum Similarity Threshold: {min_sim}")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Load normalized test datasets
    s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
    s2_path = os.path.join(NORM_DIR, "test_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "test_source3_normalized.tsv")
    
    print(f"Loading Test S1 records from {os.path.basename(s1_path)}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    
    print(f"Loading Test S2 and S3 target catalog...")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    print(f"  Test S1 Queries: {len(df_s1):,}")
    print(f"  Test S2+S3 Targets: {len(df_targets):,}")
    
    # 2. Dynamic Country Partitioning (US, India, France)
    partitions = country_partition(df_s1, df_targets)
    print(f"\nDiscovered {len(partitions)} Country Partitions: {list(partitions.keys())}")
    
    all_final_candidates = {}
    
    for country, part in partitions.items():
        s1_sub = part["s1"]
        tgt_sub = part["targets"]
        
        print(f"\n=======================================================")
        print(f"Processing Country Partition: {country}")
        print(f"  S1 Query Anchors: {len(s1_sub):,}")
        print(f"  Target Catalog  : {len(tgt_sub):,}")
        print(f"=======================================================")
        
        t_p_start = time.time()
        
        # Signal A: Char 3-gram TF-IDF
        cands_tfidf = generate_3gram_tfidf_candidates(s1_sub, tgt_sub, top_n_tfidf=k, min_similarity=min_sim)
        
        # Signal B: Brand Tokens
        cands_token = generate_token_candidates(s1_sub, tgt_sub, top_n_token=k)
        
        # Signal C: Address Numbers
        cands_addr = generate_address_number_candidates(s1_sub, tgt_sub, top_n_addr=15)
        
        # Union & Top-K Slicing
        part_s1_ids = list(s1_sub['entity_id'])
        part_candidates = union_and_rank_candidates(
            part_s1_ids,
            cands_tfidf,
            cands_token,
            cands_addr,
            k=k
        )
        
        all_final_candidates.update(part_candidates)
        elapsed_p = time.time() - t_p_start
        print(f"Partition {country} finished in {elapsed_p:.2f}s ({len(s1_sub)/elapsed_p:,.0f} queries/sec)")
        
    # 3. Write candidate_pairs.tsv
    out_file = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    print(f"\nWriting candidate set to {out_file}...")
    
    # Maintain exact order of test_source1.tsv
    all_test_s1_ids = df_s1['entity_id'].values
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            cands = all_final_candidates.get(s1_id, [])
            cand_str = ",".join(cands)
            f.write(f"{s1_id}\t{cand_str}\n")
            
    total_elapsed = time.time() - t0
    total_pairs = sum(len(c) for c in all_final_candidates.values())
    avg_pairs = total_pairs / len(df_s1) if len(df_s1) > 0 else 0
    
    print(f"\nCandidate Generation Complete in {total_elapsed/60:.2f} minutes!")
    print(f"  Total S1 Entities Processed: {len(df_s1):,}")
    print(f"  Total Generated Candidate Pairs: {total_pairs:,}")
    print(f"  Average Candidate Pairs per Entity: {avg_pairs:.2f}")
    print(f"  Output Saved to: {out_file}")
    
    # 4. Run Validator
    val_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    if os.path.isfile(val_script):
        print("\n" + "=" * 80)
        print("RUNNING SUBMISSION VALIDATOR (validate_submission.py)")
        print("=" * 80)
        cmd = [
            sys.executable,
            val_script,
            "--candidate", out_file,
            "--test-dir", TEST_DIR
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Stderr:", res.stderr)
        if res.returncode == 0:
            print(">>> VALIDATOR STATUS: PASS (Exit Code 0) <<<")
        else:
            print(">>> VALIDATOR STATUS: WARNING/FAIL <<<")

def main():
    parser = argparse.ArgumentParser(description="Generate candidate pairs for test set")
    parser.add_argument("--k", type=int, default=20, help="Top-K candidates per S1 entity (default: 20)")
    parser.add_argument("--min-sim", type=float, default=0.20, help="Minimum TF-IDF cosine similarity (default: 0.20)")
    args = parser.parse_args()
    
    generate_test_candidates(k=args.k, min_sim=args.min_sim)

if __name__ == "__main__":
    main()

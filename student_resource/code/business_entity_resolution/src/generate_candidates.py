"""
Final Candidate Pairs Generation Pipeline (Optimized & Chunked)
Processes 1.73M test Source 1 records across all country partitions (US, India, France)
using chunked multi-signal sparse blocking with strict RAM limits (< 1.8 GB peak).
Outputs validated 'output/candidate_pairs.tsv'.
"""

import os
import gc
import sys
import time
import argparse
import subprocess
import pandas as pd
import numpy as np
from blocking import build_target_blocking_index, query_blocking_index_chunk

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)


def generate_test_candidates(k: int = 20, min_sim: float = 0.18, chunk_size: int = 50000):
    print("=" * 80)
    print("HIGH-SPEED CANDIDATE GENERATION ENGINE (candidate_pairs.tsv)")
    print(f"Top-K Hyperparameter: {k}")
    print(f"Minimum Similarity: {min_sim}")
    print(f"Query Chunk Size: {chunk_size:,}")
    print("=" * 80)
    
    t0 = time.time()
    
    # 1. Load S1 query metadata (only entity_id, clean_name, address_numbers, country)
    s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
    print(f"Loading Test S1 records from {os.path.basename(s1_path)}...")
    df_s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "clean_name", "address_numbers", "country"]
    )
    total_s1 = len(df_s1)
    print(f"Total Test S1 Entities: {total_s1:,}")
    
    # Identify unique countries
    countries = list(df_s1["country"].unique())
    print(f"Discovered Country Partitions: {countries}")
    
    # 2. Process Country Partitions One-by-One (Zero-RAM Leakage)
    temp_files = {}
    
    for country in countries:
        c_str = str(country).strip()
        print(f"\n" + "=" * 60)
        print(f"PROCESSING COUNTRY PARTITION: {c_str}")
        print("=" * 60)
        
        t_country_start = time.time()
        
        # Filter S1 records for this country
        df_s1_country = df_s1[df_s1["country"] == country].reset_index(drop=True)
        n_country_s1 = len(df_s1_country)
        print(f"  S1 Query Anchors: {n_country_s1:,}")
        
        # Load only targets for this country from S2 and S3
        print(f"  Loading S2 and S3 target catalog for country '{c_str}'...")
        s2_path = os.path.join(NORM_DIR, "test_source2_normalized.tsv")
        s3_path = os.path.join(NORM_DIR, "test_source3_normalized.tsv")
        
        tgt_cols = ["entity_id", "clean_name", "address_numbers", "country"]
        
        df_s2_c = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, usecols=tgt_cols)
        df_s2_c = df_s2_c[df_s2_c["country"] == country]
        
        df_s3_c = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, usecols=tgt_cols)
        df_s3_c = df_s3_c[df_s3_c["country"] == country]
        
        df_targets_c = pd.concat([df_s2_c, df_s3_c], ignore_index=True)
        del df_s2_c, df_s3_c
        gc.collect()
        
        n_targets = len(df_targets_c)
        print(f"  Target Catalog Entities: {n_targets:,}")
        
        if n_targets == 0:
            print(f"  [Warning] No target records found for country {c_str}!")
            # Write empty candidates
            temp_file = os.path.join(OUT_DIR, f"temp_cands_{c_str}.tsv")
            temp_files[c_str] = temp_file
            with open(temp_file, "w", encoding="utf-8") as f:
                for s1_id in df_s1_country["entity_id"].values:
                    f.write(f"{s1_id}\t\n")
            continue
            
        # Build multi-signal blocking index
        index_bundle = build_target_blocking_index(df_targets_c)
        del df_targets_c
        gc.collect()
        
        # Stream S1 queries in chunks
        temp_file = os.path.join(OUT_DIR, f"temp_cands_{c_str}.tsv")
        temp_files[c_str] = temp_file
        
        s1_ids_all = df_s1_country["entity_id"].values
        s1_names_all = df_s1_country["clean_name"].values
        s1_nums_all = df_s1_country["address_numbers"].values
        
        num_chunks = (n_country_s1 + chunk_size - 1) // chunk_size
        print(f"  Executing candidate retrieval across {num_chunks} chunk(s)...")
        
        with open(temp_file, "w", encoding="utf-8") as f_tmp:
            for c_idx in range(num_chunks):
                start_i = c_idx * chunk_size
                end_i = min(n_country_s1, (c_idx + 1) * chunk_size)
                
                chunk_ids = s1_ids_all[start_i:end_i]
                chunk_names = s1_names_all[start_i:end_i]
                chunk_nums = s1_nums_all[start_i:end_i]
                
                t_chunk = time.time()
                chunk_results = query_blocking_index_chunk(
                    chunk_ids, chunk_names, chunk_nums,
                    index_bundle, k=k, min_sim=min_sim
                )
                
                # Write chunk directly to disk
                for s1_id, cands in chunk_results:
                    cand_str = ",".join(cands)
                    f_tmp.write(f"{s1_id}\t{cand_str}\n")
                    
                elapsed_chunk = time.time() - t_chunk
                rate = len(chunk_ids) / elapsed_chunk if elapsed_chunk > 0 else 0
                print(f"    Chunk {c_idx+1}/{num_chunks}: Processed {end_i:,}/{n_country_s1:,} queries in {elapsed_chunk:.2f}s ({rate:,.0f} q/s)")
                
        # Clean up country memory
        del index_bundle, df_s1_country, s1_ids_all, s1_names_all, s1_nums_all
        gc.collect()
        
        c_elapsed = time.time() - t_country_start
        print(f"  Partition {c_str} Complete in {c_elapsed:.2f}s ({c_elapsed/60:.2f} min)!")
        
    # 3. Assemble final candidate_pairs.tsv in exact original S1 order
    out_file = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    print("\n" + "=" * 80)
    print(f"Assembling final ordered candidate set into: {out_file}...")
    print("=" * 80)
    
    t_merge = time.time()
    
    # Read temp partitions into fast disk/memory lookup
    partition_cands = {}
    for c_str, path in temp_files.items():
        print(f"  Reading partition '{c_str}' from {os.path.basename(path)}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0]
                cand_str = parts[1] if len(parts) > 1 else ""
                partition_cands[s1_id] = cand_str
                
        # Clean up temp file
        try:
            os.remove(path)
        except OSError:
            pass
            
    # Write final candidate_pairs.tsv in exact order of test_source1.tsv
    total_pairs = 0
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in df_s1["entity_id"].values:
            cands_str = partition_cands.get(s1_id, "")
            if cands_str:
                total_pairs += len(cands_str.split(","))
            f_out.write(f"{s1_id}\t{cands_str}\n")
            
    del partition_cands, df_s1
    gc.collect()
    
    total_elapsed = time.time() - t0
    avg_pairs = total_pairs / total_s1 if total_s1 > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"Candidate Generation Complete in {total_elapsed/60:.2f} minutes!")
    print(f"  Total S1 Entities Processed: {total_s1:,}")
    print(f"  Total Candidate Pairs Generated: {total_pairs:,}")
    print(f"  Average Candidate Pairs per Entity: {avg_pairs:.2f}")
    print(f"  Output Saved to: {out_file}")
    print("=" * 80)
    
    # 4. Run Validator
    val_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    if os.path.isfile(val_script):
        print("\n" + "=" * 80)
        print("RUNNING SUBMISSION VALIDATOR (validate_submission.py)")
        print("=" * 80)
        raw_test_dir = os.path.join(BASE_DIR, "dataset", "test")
        norm_dir = os.path.join(BASE_DIR, "dataset", "normalized")
        chk_dir = raw_test_dir if os.path.isdir(raw_test_dir) else norm_dir
        cmd = [
            sys.executable,
            val_script,
            "--candidate", out_file,
            "--test-dir", chk_dir
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
    parser.add_argument("--min-sim", type=float, default=0.18, help="Minimum similarity threshold (default: 0.18)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size (default: 50,000)")
    args = parser.parse_args()
    
    generate_test_candidates(k=args.k, min_sim=args.min_sim, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()

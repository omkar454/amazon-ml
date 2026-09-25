"""
Multi-core High-Performance Dataset Normalizer
Processes all Train and Test TSV files in parallel streaming batches
and saves normalized datasets to 'dataset/normalized/' in fast Parquet format (or TSV if parquet engine unavailable).
"""

import os
import sys
import time
import multiprocessing as mp
import pandas as pd
import numpy as np
from normalizer import normalize_record

# Input directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUT_DIR = os.path.join(BASE_DIR, "dataset", "normalized")

os.makedirs(OUT_DIR, exist_ok=True)

# Determine preferred storage format (Parquet if pyarrow/fastparquet is available, else TSV)
USE_PARQUET = True
try:
    import pyarrow
except ImportError:
    try:
        import fastparquet
    except ImportError:
        USE_PARQUET = False

def process_batch(rows):
    """Worker function to normalize a batch of (entity_id, name, address, country) tuples."""
    results = []
    for eid, name, addr, country in rows:
        norm = normalize_record(name, addr, country)
        results.append((
            eid,
            norm["clean_name"],
            norm["clean_name_full"],
            norm["clean_address"],
            ",".join(norm["address_numbers"]),
            norm["has_address"],
            norm["country"]
        ))
    return results

def normalize_file(input_path, output_path, num_workers=None, chunk_size=200000):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    file_name = os.path.basename(input_path)
    print(f"\n[{file_name}] Starting normalization with {num_workers} parallel workers...")
    start_time = time.time()
    
    total_processed = 0
    first_chunk = True
    
    # Process in streaming chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size)):
        t_chunk_start = time.time()
        
        tuples = list(zip(
            chunk['entity_id'].values,
            chunk['business_name'].values,
            chunk['business_address'].values,
            chunk['country'].values
        ))
        
        sub_batch_size = max(1, len(tuples) // (num_workers * 4))
        sub_batches = [tuples[i:i + sub_batch_size] for i in range(0, len(tuples), sub_batch_size)]
        
        with mp.Pool(processes=num_workers) as pool:
            batch_results = pool.map(process_batch, sub_batches)
            
        flat_results = [item for sublist in batch_results for item in sublist]
        
        df_norm = pd.DataFrame(flat_results, columns=[
            "entity_id",
            "clean_name",
            "clean_name_full",
            "clean_address",
            "address_numbers",
            "has_address",
            "country"
        ])
        
        if USE_PARQUET:
            parquet_path = output_path.replace(".tsv", ".parquet")
            if first_chunk:
                df_norm.to_parquet(parquet_path, index=False)
                first_chunk = False
            else:
                existing_df = pd.read_parquet(parquet_path)
                combined_df = pd.concat([existing_df, df_norm], ignore_index=True)
                combined_df.to_parquet(parquet_path, index=False)
        else:
            tsv_path = output_path.replace(".parquet", ".tsv")
            mode = "w" if first_chunk else "a"
            header = first_chunk
            df_norm.to_csv(tsv_path, sep="\t", index=False, mode=mode, header=header)
            first_chunk = False
            
        total_processed += len(chunk)
        elapsed_chunk = time.time() - t_chunk_start
        rate_chunk = len(chunk) / elapsed_chunk
        print(f"  Processed chunk {chunk_idx + 1:2d} ({total_processed:,} records total) | {rate_chunk:,.0f} rec/sec")
        
    total_elapsed = time.time() - start_time
    avg_rate = total_processed / total_elapsed
    final_dest = output_path.replace(".tsv", ".parquet") if USE_PARQUET else output_path.replace(".parquet", ".tsv")
    print(f"[{file_name}] Complete! {total_processed:,} records normalized in {total_elapsed:.2f}s ({avg_rate:,.0f} rec/sec).")
    print(f"  Saved to: {final_dest}")

def main():
    print("=" * 80)
    print("UNIVERSAL DATASET NORMALIZATION PIPELINE")
    print(f"CPU Cores Available: {mp.cpu_count()}")
    print(f"Storage Format: {'Parquet (.parquet)' if USE_PARQUET else 'Tab-Separated (.tsv)'}")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 80)
    
    files_to_process = [
        (os.path.join(TRAIN_DIR, "train_source1.tsv"), os.path.join(OUT_DIR, "train_source1_normalized.parquet")),
        (os.path.join(TRAIN_DIR, "train_source2.tsv"), os.path.join(OUT_DIR, "train_source2_normalized.parquet")),
        (os.path.join(TRAIN_DIR, "train_source3.tsv"), os.path.join(OUT_DIR, "train_source3_normalized.parquet")),
        (os.path.join(TEST_DIR, "test_source1.tsv"), os.path.join(OUT_DIR, "test_source1_normalized.parquet")),
        (os.path.join(TEST_DIR, "test_source2.tsv"), os.path.join(OUT_DIR, "test_source2_normalized.parquet")),
        (os.path.join(TEST_DIR, "test_source3.tsv"), os.path.join(OUT_DIR, "test_source3_normalized.parquet")),
    ]
    
    overall_start = time.time()
    for in_path, out_path in files_to_process:
        if not os.path.isfile(in_path):
            print(f"Warning: {in_path} not found. Skipping.")
            continue
        normalize_file(in_path, out_path)
        
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"ALL DATASETS NORMALIZED SUCCESSFULLY IN {overall_elapsed / 60:.2f} MINUTES!")
    print("=" * 80)

if __name__ == "__main__":
    mp.freeze_support()
    main()

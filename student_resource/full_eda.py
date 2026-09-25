import os
import sys
import time
import pandas as pd
import numpy as np
from collections import Counter

train_dir = "dataset/train"
test_dir = "dataset/test"

print("=" * 70)
print("FULL DATASET EDA — STREAMING & ACCUMULATOR ANALYSIS")
print("=" * 70)

def analyze_tsv_file(file_path, chunksize=500000):
    start_time = time.time()
    total_rows = 0
    missing_counts = Counter()
    country_counts = Counter()
    name_len_total = 0
    addr_len_total = 0
    non_ascii_name_count = 0
    non_ascii_addr_count = 0
    max_name_len = 0
    max_addr_len = 0
    
    col_names = None
    
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False):
        if col_names is None:
            col_names = list(chunk.columns)
            
        nrows = len(chunk)
        total_rows += nrows
        
        for col in chunk.columns:
            # Check empty strings
            empty_count = (chunk[col] == "").sum()
            missing_counts[col] += empty_count
            
        if 'country' in chunk.columns:
            country_counts.update(chunk['country'].value_counts().to_dict())
            
        if 'business_name' in chunk.columns:
            names = chunk['business_name']
            lens = names.str.len()
            name_len_total += lens.sum()
            max_name_len = max(max_name_len, lens.max())
            non_ascii_name_count += names.str.contains(r'[^\x00-\x7F]').sum()
            
        if 'business_address' in chunk.columns:
            addrs = chunk['business_address']
            lens = addrs.str.len()
            addr_len_total += lens.sum()
            max_addr_len = max(max_addr_len, lens.max())
            non_ascii_addr_count += addrs.str.contains(r'[^\x00-\x7F]').sum()
            
    elapsed = time.time() - start_time
    return {
        "rows": total_rows,
        "cols": col_names,
        "missing": dict(missing_counts),
        "countries": dict(country_counts),
        "avg_name_len": name_len_total / total_rows if total_rows else 0,
        "max_name_len": max_name_len,
        "non_ascii_name_pct": (non_ascii_name_count / total_rows * 100) if total_rows else 0,
        "avg_addr_len": addr_len_total / total_rows if total_rows else 0,
        "max_addr_len": max_addr_len,
        "non_ascii_addr_pct": (non_ascii_addr_count / total_rows * 100) if total_rows else 0,
        "time_sec": elapsed
    }

files_to_analyze = [
    ("Train Source 1", os.path.join(train_dir, "train_source1.tsv")),
    ("Train Source 2", os.path.join(train_dir, "train_source2.tsv")),
    ("Train Source 3", os.path.join(train_dir, "train_source3.tsv")),
    ("Test Source 1", os.path.join(test_dir, "test_source1.tsv")),
    ("Test Source 2", os.path.join(test_dir, "test_source2.tsv")),
    ("Test Source 3", os.path.join(test_dir, "test_source3.tsv")),
]

for label, fpath in files_to_analyze:
    print(f"\nAnalyzing {label} ({os.path.basename(fpath)})...")
    res = analyze_tsv_file(fpath)
    print(f"  Total Rows: {res['rows']:,} (processed in {res['time_sec']:.2f}s)")
    print(f"  Missing values: {res['missing']}")
    print(f"  Countries: {res['countries']}")
    print(f"  Business Name: Avg Char Len = {res['avg_name_len']:.1f}, Max Len = {res['max_name_len']}, Non-ASCII = {res['non_ascii_name_pct']:.2f}%")
    print(f"  Business Address: Avg Char Len = {res['avg_addr_len']:.1f}, Max Len = {res['max_addr_len']}, Non-ASCII = {res['non_ascii_addr_pct']:.2f}%")

print("\n" + "=" * 70)
print("ANALYZING FULL GROUND TRUTH (train_ground_truth.tsv)")
print("=" * 70)

gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
total_gt_rows = 0
singleton_count = 0
match_hist = Counter()
total_matches = 0
total_s2_matches = 0
total_s3_matches = 0
max_matches = 0

start_gt = time.time()
for chunk in pd.read_csv(gt_path, sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    total_gt_rows += len(chunk)
    for m_str in chunk['matched_entity_ids']:
        if not m_str:
            singleton_count += 1
            match_hist[0] += 1
        else:
            ids = [x.strip() for x in m_str.split(",") if x.strip()]
            num_m = len(ids)
            match_hist[num_m] += 1
            total_matches += num_m
            max_matches = max(max_matches, num_m)
            
            s2_c = sum(1 for x in ids if x.startswith("S2-"))
            s3_c = sum(1 for x in ids if x.startswith("S3-"))
            total_s2_matches += s2_c
            total_s3_matches += s3_c

elapsed_gt = time.time() - start_gt
print(f"Ground truth total rows: {total_gt_rows:,} (processed in {elapsed_gt:.2f}s)")
print(f"Total True Match Pairs: {total_matches:,} (S2: {total_s2_matches:,}, S3: {total_s3_matches:,})")
print(f"Singletons (0 matches): {singleton_count:,} ({singleton_count / total_gt_rows * 100:.2f}%)")
print(f"Entities with >= 1 match: {total_gt_rows - singleton_count:,} ({(total_gt_rows - singleton_count) / total_gt_rows * 100:.2f}%)")
print(f"Average matches per entity: {total_matches / total_gt_rows:.3f}")
print(f"Average matches among non-singletons: {total_matches / (total_gt_rows - singleton_count):.3f}")
print(f"Max matches for a single entity: {max_matches}")

print("\nMatch Count Histogram (matches per S1 entity):")
for k in sorted(match_hist.keys()):
    cnt = match_hist[k]
    pct = cnt / total_gt_rows * 100
    bar = "#" * int(pct / 2)
    print(f"  {k:2d} matches: {cnt:9,d} ({pct:5.2f}%) | {bar}")

print("\n" + "=" * 70)
print("CROSS-COUNTRY GROUND TRUTH CONSISTENCY CHECK")
print("=" * 70)

# Build S1 country map from train_source1
print("Building S1 country mapping...")
s1_country_map = {}
for chunk in pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", chunksize=1000000, usecols=['entity_id', 'country'], dtype=str):
    s1_country_map.update(dict(zip(chunk['entity_id'], chunk['country'])))

# Build S2 & S3 country map
print("Building S2 & S3 country mapping...")
target_country_map = {}
for chunk in pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", chunksize=1000000, usecols=['entity_id', 'country'], dtype=str):
    target_country_map.update(dict(zip(chunk['entity_id'], chunk['country'])))
for chunk in pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", chunksize=1000000, usecols=['entity_id', 'country'], dtype=str):
    target_country_map.update(dict(zip(chunk['entity_id'], chunk['country'])))

print("Verifying cross-country match purity in Ground Truth...")
cross_country_mismatches = 0
total_checked = 0

for chunk in pd.read_csv(gt_path, sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    for s1_id, m_str in zip(chunk['source1_entity_id'], chunk['matched_entity_ids']):
        if not m_str:
            continue
        c1 = s1_country_map.get(s1_id)
        if not c1:
            continue
        for mid in m_str.split(","):
            mid = mid.strip()
            if not mid:
                continue
            c2 = target_country_map.get(mid)
            if c2 and c1 != c2:
                cross_country_mismatches += 1
            total_checked += 1

print(f"Total evaluated pairs: {total_checked:,}")
print(f"Cross-country mismatches found: {cross_country_mismatches}")
if cross_country_mismatches == 0:
    print(">>> 100% PROVEN: ZERO cross-country matches exist. Entities NEVER match across different countries! <<<")
else:
    print(f"Warning: {cross_country_mismatches} pairs had different country labels.")

print("\nFULL EDA COMPLETED SUCCESSFULLY.")

import pandas as pd
import numpy as np
import os
import re
from collections import Counter

train_dir = "dataset/train"
test_dir = "dataset/test"

print("=" * 60)
print("1. LOADING HEAD SAMPLES & SCHEMA INSPECTION")
print("=" * 60)

# Load a sample from each train file
df_s1_sample = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=50000)
df_s2_sample = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", nrows=50000)
df_s3_sample = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", nrows=50000)
df_gt_sample = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", nrows=100000)

df_test_s1_sample = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", nrows=50000)
df_test_s2_sample = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", nrows=50000)
df_test_s3_sample = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", nrows=50000)

print("Source 1 train shape sample:", df_s1_sample.shape)
print("Source 1 columns:", df_s1_sample.columns.tolist())
print("\nMissing values in S1 sample:")
print(df_s1_sample.isnull().sum())

print("\nMissing values in S2 sample:")
print(df_s2_sample.isnull().sum())

print("\nMissing values in S3 sample:")
print(df_s3_sample.isnull().sum())

print("\n" + "=" * 60)
print("2. COUNTRY DISTRIBUTION")
print("=" * 60)
print("Train S1 countries (%):")
print(df_s1_sample['country'].value_counts(normalize=True) * 100)

print("\nTest S1 countries (%):")
print(df_test_s1_sample['country'].value_counts(normalize=True) * 100)

print("\nTest S2 countries (%):")
print(df_test_s2_sample['country'].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("3. GROUND TRUTH MATCH DISTRIBUTION (100k sample)")
print("=" * 60)

# Fill nulls in matched_entity_ids with empty string
df_gt_sample['matched_entity_ids'] = df_gt_sample['matched_entity_ids'].fillna("")

def parse_matches(m_str):
    if not m_str.strip():
        return []
    return [x.strip() for x in m_str.split(",")]

match_lists = df_gt_sample['matched_entity_ids'].apply(parse_matches)
match_counts = match_lists.apply(len)

num_singletons = (match_counts == 0).sum()
print(f"Total S1 entities in sample: {len(df_gt_sample)}")
print(f"Singletons (0 matches): {num_singletons} ({num_singletons / len(df_gt_sample) * 100:.2f}%)")
print(f"Entities with matches: {len(df_gt_sample) - num_singletons} ({(len(df_gt_sample) - num_singletons) / len(df_gt_sample) * 100:.2f}%)")
print(f"Match count summary:\n{match_counts.describe()}")

# Source 2 vs Source 3 match split
s2_matches = match_lists.apply(lambda lst: sum(1 for x in lst if x.startswith("S2-")))
s3_matches = match_lists.apply(lambda lst: sum(1 for x in lst if x.startswith("S3-")))

print(f"\nAverage S2 matches per entity: {s2_matches.mean():.3f} (max: {s2_matches.max()})")
print(f"Average S3 matches per entity: {s3_matches.mean():.3f} (max: {s3_matches.max()})")

print("\n" + "=" * 60)
print("4. REAL MATCH EXAMPLES (INSPECTING NOISE PATTERNS)")
print("=" * 60)

# Index samples by entity_id
s1_dict = df_s1_sample.set_index('entity_id').to_dict('index')
s2_dict = df_s2_sample.set_index('entity_id').to_dict('index')
s3_dict = df_s3_sample.set_index('entity_id').to_dict('index')

examples_found = 0
for _, row in df_gt_sample.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = parse_matches(row['matched_entity_ids'])
    if s1_id in s1_dict and m_ids:
        # Check if we have at least one match record in s2_dict or s3_dict
        found_matches = []
        for mid in m_ids:
            if mid in s2_dict:
                found_matches.append((mid, s2_dict[mid]))
            elif mid in s3_dict:
                found_matches.append((mid, s3_dict[mid]))
        
        if found_matches:
            examples_found += 1
            s1_info = s1_dict[s1_id]
            print(f"\n--- Example {examples_found} ---")
            print(f"[Anchor S1] ID: {s1_id} | Country: {s1_info['country']}")
            print(f"  Name   : {s1_info['business_name']}")
            print(f"  Address: {s1_info['business_address']}")
            for mid, minfo in found_matches:
                print(f"  [Match {mid[:2]}] ID: {mid} | Country: {minfo['country']}")
                print(f"    Name   : {minfo['business_name']}")
                print(f"    Address: {minfo['business_address']}")
            
            if examples_found >= 10:
                break

print("\nEDA script completed successfully.")

"""
Final Candidate Pairs Generation Pipeline

Generates candidate pairs for the Amazon ML Challenge
Business Entity Resolution test dataset.

Pipeline:

    Test S1 + S2 + S3
            ↓
    Country Partitioning
            ↓
    ┌─────────────────────────────┐
    │ Signal A: E5 + FAISS IVF-PQ│
    │ Signal B: Brand Tokens      │
    │ Signal C: Address Numbers  │
    └─────────────────────────────┘
            ↓
       Union + Ranking
            ↓
          Top-K
            ↓
    candidate_pairs.tsv

Current mode:
    10,000 S1 smoke test

After successful smoke test:
    Remove/comment the `head(10_000)` line
    to process the complete test S1 dataset.
"""

import os
import sys
import time
import argparse
import subprocess
import re

import pandas as pd

from blocking import (
    country_partition,
    generate_embedding_candidates,
    generate_token_candidates,
    generate_address_number_candidates,
    union_and_rank_candidates,
)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = "/content/drive/MyDrive/"

NORM_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "normalized"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    "signal_a"
)


# Create required directories
os.makedirs(
    OUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def generate_test_candidates(
    k: int = 20,
    min_sim: float = 0.20
):

    print("=" * 80)
    print(
        "TEST CANDIDATE GENERATION ENGINE "
        "(candidate_pairs.tsv)"
    )
    print(
        f"Top-K Hyperparameter: {k}"
    )
    print(
        f"Minimum Similarity Threshold: {min_sim}"
    )
    print("=" * 80)

    total_start_time = time.time()

    # =========================================================================
    # 1. LOAD NORMALIZED S1
    # =========================================================================

    s1_path = os.path.join(
        NORM_DIR,
        "test_source1_normalized.tsv"
    )

    s2_path = os.path.join(
        NORM_DIR,
        "test_source2_normalized.tsv"
    )

    s3_path = os.path.join(
        NORM_DIR,
        "test_source3_normalized.tsv"
    )

    print(
        f"Loading Test S1 records from "
        f"{os.path.basename(s1_path)}..."
    )

    df_s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # =========================================================================
    # TEMPORARY 10K SMOKE TEST
    # =========================================================================
    #
    # KEEP THIS FOR THE FIRST TEST.
    #
    # After US + FRANCE + INDIA all process successfully,
    # remove/comment this line to process the complete dataset.
    #
    # =========================================================================

    # df_s1 = df_s1.head(10_000)

    # =========================================================================
    # 2. LOAD S2 + S3
    # =========================================================================

    print(
        "Loading Test S2 and S3 target catalog..."
    )

    df_s2 = pd.read_csv(
        s2_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df_s3 = pd.read_csv(
        s3_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df_targets = pd.concat(
        [
            df_s2,
            df_s3
        ],
        ignore_index=True
    )

    print(
        f"  Test S1 Queries: "
        f"{len(df_s1):,}"
    )

    print(
        f"  Test S2+S3 Targets: "
        f"{len(df_targets):,}"
    )

    # =========================================================================
    # 3. COUNTRY PARTITIONING
    # =========================================================================

    partitions = country_partition(
        df_s1,
        df_targets
    )

    print(
        f"\nDiscovered "
        f"{len(partitions)} Country Partitions: "
        f"{list(partitions.keys())}"
    )

    # =========================================================================
    # STORAGE FOR ALL COUNTRY RESULTS
    # =========================================================================

    all_final_candidates = {}

    # =========================================================================
    # 4. PROCESS EACH COUNTRY
    # =========================================================================

    for country, part in partitions.items():

        s1_sub = part["s1"]

        tgt_sub = part["targets"]

        print("\n" + "=" * 60)

        print(
            f"Processing Country Partition: "
            f"{country}"
        )

        print(
            f"  S1 Query Anchors: "
            f"{len(s1_sub):,}"
        )

        print(
            f"  Target Catalog  : "
            f"{len(tgt_sub):,}"
        )

        print("=" * 60)

        partition_start = time.time()

        # =====================================================================
        # SAFETY CHECK
        # =====================================================================

        if len(s1_sub) == 0:

            print(
                f"  No S1 records for {country}. "
                f"Skipping."
            )

            continue

        if len(tgt_sub) == 0:

            print(
                f"  No target records for {country}. "
                f"Skipping."
            )

            # Still make sure every S1 gets an empty candidate list
            for s1_id in s1_sub["entity_id"]:
                all_final_candidates[
                    str(s1_id)
                ] = []

            continue

        # =====================================================================
        # COUNTRY-SPECIFIC SIGNAL A CHECKPOINT
        # =====================================================================

        country_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(country).strip()
        )

        country_checkpoint_dir = os.path.join(
            CHECKPOINT_DIR,
            country_name
        )

        os.makedirs(
            country_checkpoint_dir,
            exist_ok=True
        )

        print(
            f"  Signal A checkpoint: "
            f"{country_checkpoint_dir}"
        )

        # =====================================================================
        # SIGNAL A
        # MULTILINGUAL E5 + FAISS IVF-PQ
        # =====================================================================

        cands_embedding = generate_embedding_candidates(
            s1_sub,
            tgt_sub,
            top_k=k,
            target_chunk_size=100_000,
            query_chunk_size=10_000,
            checkpoint_dir=country_checkpoint_dir
        )

        # =====================================================================
        # SIGNAL B
        # BRAND TOKEN INVERTED INDEX
        # =====================================================================

        cands_token = generate_token_candidates(
            s1_sub,
            tgt_sub,
            top_n_token=k
        )

        # =====================================================================
        # SIGNAL C
        # ADDRESS NUMBER INDEX
        # =====================================================================

        cands_addr = generate_address_number_candidates(
            s1_sub,
            tgt_sub,
            top_n_addr=15
        )

        # =====================================================================
        # UNION + RANK + TOP-K
        # =====================================================================

        part_s1_ids = list(
            s1_sub["entity_id"]
        )

        part_candidates = union_and_rank_candidates(
            part_s1_ids,
            cands_embedding,
            cands_token,
            cands_addr,
            k=k
        )

        # =====================================================================
        # ADD COUNTRY RESULTS
        # =====================================================================

        all_final_candidates.update(
            part_candidates
        )

        # =====================================================================
        # COUNTRY SUMMARY
        # =====================================================================

        partition_elapsed = (
            time.time()
            - partition_start
        )

        qps = (
            len(s1_sub) / partition_elapsed
            if partition_elapsed > 0
            else 0
        )

        print(
            f"\nPartition {country} finished in "
            f"{partition_elapsed:.2f}s "
            f"({qps:,.0f} queries/sec)"
        )

    # =========================================================================
    # 5. MAKE SURE EVERY S1 HAS AN ENTRY
    # =========================================================================

    for s1_id in df_s1["entity_id"]:

        s1_id = str(s1_id)

        if s1_id not in all_final_candidates:

            all_final_candidates[
                s1_id
            ] = []

    # =========================================================================
    # 6. WRITE candidate_pairs.tsv
    # =========================================================================

    out_file = os.path.join(
        OUT_DIR,
        "candidate_pairs.tsv"
    )

    print(
        f"\nWriting candidate set to "
        f"{out_file}..."
    )

    # Preserve exact S1 order
    all_test_s1_ids = (
        df_s1["entity_id"]
        .astype(str)
        .tolist()
    )

    with open(
        out_file,
        "w",
        encoding="utf-8"
    ) as f:

        # Header
        f.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

        # Rows
        for s1_id in all_test_s1_ids:

            candidates = (
                all_final_candidates.get(
                    s1_id,
                    []
                )
            )

            candidate_string = ",".join(
                candidates
            )

            f.write(
                f"{s1_id}\t"
                f"{candidate_string}\n"
            )

    # =========================================================================
    # 7. FINAL STATISTICS
    # =========================================================================

    total_elapsed = (
        time.time()
        - total_start_time
    )

    total_pairs = sum(
        len(candidates)
        for candidates
        in all_final_candidates.values()
    )

    avg_pairs = (
        total_pairs / len(df_s1)
        if len(df_s1) > 0
        else 0
    )

    print("\n" + "=" * 80)

    print(
        f"Candidate Generation Complete "
        f"in {total_elapsed / 60:.2f} minutes!"
    )

    print(
        f"  Total S1 Entities Processed: "
        f"{len(df_s1):,}"
    )

    print(
        f"  Total Generated Candidate Pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"  Average Candidate Pairs per Entity: "
        f"{avg_pairs:.2f}"
    )

    print(
        f"  Output Saved to: "
        f"{out_file}"
    )

    print("=" * 80)

    # =========================================================================
    # 8. VALIDATOR
    # =========================================================================

    val_script = os.path.join(
        BASE_DIR,
        "utils",
        "validate_submission.py"
    )

    if os.path.isfile(val_script):

        print("\n" + "=" * 80)
        print(
            "RUNNING SUBMISSION VALIDATOR "
            "(validate_submission.py)"
        )
        print("=" * 80)

        cmd = [
            sys.executable,
            val_script,
            "--candidate",
            out_file,
            "--test-dir",
            TEST_DIR
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Validator stdout
        if result.stdout:
            print(
                result.stdout
            )

        # Validator stderr
        if result.stderr:
            print(
                "Stderr:"
            )
            print(
                result.stderr
            )

        if result.returncode == 0:

            print(
                ">>> VALIDATOR STATUS: "
                "PASS (Exit Code 0) <<<"
            )

        else:

            print(
                ">>> VALIDATOR STATUS: "
                "WARNING/FAIL <<<"
            )

    else:

        print(
            "\nValidator script not found:"
        )

        print(
            val_script
        )


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate pairs "
            "for the Amazon ML Challenge "
            "test dataset."
        )
    )

    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help=(
            "Top-K candidates per S1 entity "
            "(default: 20)"
        )
    )

    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.20,
        help=(
            "Legacy similarity parameter. "
            "Currently unused by embedding "
            "Signal A."
        )
    )

    args = parser.parse_args()

    generate_test_candidates(
        k=args.k,
        min_sim=args.min_sim
    )


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":

    main()
"""
Memory-Safe Final Candidate Pairs Generation Pipeline
Amazon ML Challenge - Business Entity Resolution

Pipeline:

    Test S1 + S2 + S3
            ↓
    Country Partitioning
            ↓
    ┌─────────────────────────────────────┐
    │ Signal A: E5 + FAISS IVF-PQ        │
    │ Signal B: Brand Token Blocking      │
    │ Signal C: Address Number Blocking   │
    └─────────────────────────────────────┘
            ↓
       Union + Ranking
            ↓
          Top-K
            ↓
    candidate_pairs.tsv

Memory Strategy
---------------
Processing is performed in batches.

Only one S1 batch is held for final candidate merging at a time.

Existing FAISS target/query checkpoints are reused.
"""

import os
import sys
import time
import argparse
import subprocess
import re
import gc

import pandas as pd

from blocking import (
    country_partition,
    build_or_resume_embedding_index,
    generate_embedding_query_checkpoints,
    stream_embedding_candidates,
    build_token_inverted_index,
    generate_token_candidates_batch,
    build_address_number_index,
    generate_address_number_candidates_batch,
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

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_K = 20

TARGET_CHUNK_SIZE = 100_000

QUERY_CHUNK_SIZE = 10_000

ADDRESS_TOP_K = 15

# Final output filename
OUTPUT_FILE = os.path.join(
    OUT_DIR,
    "candidate_pairs.tsv"
)


# =============================================================================
# MEMORY CLEANUP
# =============================================================================

def cleanup_memory():

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass


# =============================================================================
# COUNTRY CHECKPOINT NAME
# =============================================================================

def get_country_checkpoint_dir(
    country: str
):

    country_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(country).strip()
    )

    checkpoint_dir = os.path.join(
        CHECKPOINT_DIR,
        country_name
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    return checkpoint_dir


# =============================================================================
# LOAD DATA
# =============================================================================

def load_test_data():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING NORMALIZED TEST DATA"
    )

    print(
        "=" * 80
    )

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

    # -----------------------------------------------------------------
    # S1
    # -----------------------------------------------------------------

    print(
        f"\nLoading S1:"
    )

    print(
        s1_path
    )

    df_s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # -----------------------------------------------------------------
    # S2
    # -----------------------------------------------------------------

    print(
        "\nLoading S2:"
    )

    print(
        s2_path
    )

    df_s2 = pd.read_csv(
        s2_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # -----------------------------------------------------------------
    # S3
    # -----------------------------------------------------------------

    print(
        "\nLoading S3:"
    )

    print(
        s3_path
    )

    df_s3 = pd.read_csv(
        s3_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # -----------------------------------------------------------------
    # Combine targets
    # -----------------------------------------------------------------

    df_targets = pd.concat(
        [
            df_s2,
            df_s3
        ],
        ignore_index=True
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"S1 records       : {len(df_s1):,}"
    )

    print(
        f"S2 records       : {len(df_s2):,}"
    )

    print(
        f"S3 records       : {len(df_s3):,}"
    )

    print(
        f"S2 + S3 targets  : {len(df_targets):,}"
    )

    print(
        "-" * 80
    )

    return (
        df_s1,
        df_targets
    )


# =============================================================================
# WRITE OUTPUT HEADER
# =============================================================================

def initialize_output(
    output_file: str
):

    print(
        f"\nInitializing output:"
    )

    print(
        output_file
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )


# =============================================================================
# APPEND BATCH TO OUTPUT
# =============================================================================

def append_candidates_to_output(
    output_file: str,
    s1_ids,
    final_candidates
):

    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as f:

        for s1_id in s1_ids:

            s1_id = str(
                s1_id
            )

            candidates = final_candidates.get(
                s1_id,
                []
            )

            candidate_string = ",".join(
                str(x)
                for x in candidates
            )

            f.write(
                f"{s1_id}\t"
                f"{candidate_string}\n"
            )


# =============================================================================
# PROCESS ONE COUNTRY
# =============================================================================

def process_country(
    country,
    s1_sub,
    target_sub,
    k,
    output_file
):

    country_start = time.time()

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"PROCESSING COUNTRY: {country}"
    )

    print(
        "=" * 80
    )

    print(
        f"S1 records      : {len(s1_sub):,}"
    )

    print(
        f"Target records   : {len(target_sub):,}"
    )

    # -----------------------------------------------------------------
    # Empty partition
    # -----------------------------------------------------------------

    if len(s1_sub) == 0:

        print(
            "No S1 records. Skipping."
        )

        return {
            "country": country,
            "s1_count": 0,
            "pairs": 0,
            "elapsed": 0
        }

    if len(target_sub) == 0:

        print(
            "No targets for this country."
        )

        # Write empty candidate rows
        s1_ids = (
            s1_sub[
                "entity_id"
            ]
            .astype(str)
            .tolist()
        )

        append_candidates_to_output(
            output_file,
            s1_ids,
            {}
        )

        return {
            "country": country,
            "s1_count": len(s1_sub),
            "pairs": 0,
            "elapsed": time.time() - country_start
        }

    # -----------------------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------------------

    checkpoint_dir = get_country_checkpoint_dir(
        country
    )

    print(
        "\nSignal A checkpoint:"
    )

    print(
        checkpoint_dir
    )

    # =================================================================
    # SIGNAL A - BUILD / RESUME FAISS INDEX
    # =================================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "SIGNAL A - FAISS IVF-PQ"
    )

    print(
        "-" * 70
    )

    index = build_or_resume_embedding_index(
        target_sub,
        checkpoint_dir,
        target_chunk_size=TARGET_CHUNK_SIZE
    )

    # =================================================================
    # SIGNAL A - QUERY CHECKPOINTS
    # =================================================================

    generate_embedding_query_checkpoints(
        s1_sub,
        target_sub,
        index,
        checkpoint_dir,
        top_k=k,
        query_chunk_size=QUERY_CHUNK_SIZE
    )

    # -----------------------------------------------------------------
    # Release index from memory before candidate processing.
    #
    # The query checkpoints already contain all FAISS results.
    # -----------------------------------------------------------------

    del index

    cleanup_memory()

    # =================================================================
    # SIGNAL B - BUILD TOKEN INDEX
    # =================================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "SIGNAL B - TOKEN INVERTED INDEX"
    )

    print(
        "-" * 70
    )

    token_index = build_token_inverted_index(
        target_sub
    )

    # =================================================================
    # SIGNAL C - BUILD ADDRESS INDEX
    # =================================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "SIGNAL C - ADDRESS NUMBER INDEX"
    )

    print(
        "-" * 70
    )

    address_index = build_address_number_index(
        target_sub
    )

    # =================================================================
    # STREAM SIGNAL A QUERY CHUNKS
    # =================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STREAMING FINAL CANDIDATE GENERATION"
    )

    print(
        "=" * 80
    )

    print(
        "Only one 10,000-S1 query chunk is processed at a time."
    )

    # ---------------------------------------------------------------
    # Target counters
    # ---------------------------------------------------------------

    total_s1_processed = 0

    total_pairs = 0

    chunk_number = 0

    # ---------------------------------------------------------------
    # Signal A checkpoint stream
    # ---------------------------------------------------------------

    for (
        start_idx,
        end_idx,
        embedding_batch
    ) in stream_embedding_candidates(
        s1_sub,
        target_sub,
        checkpoint_dir,
        top_k=k,
        query_chunk_size=QUERY_CHUNK_SIZE
    ):

        chunk_start_time = time.time()

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"FINAL BATCH #{chunk_number}"
        )

        print(
            f"S1 positions: "
            f"{start_idx:,} -> {end_idx:,}"
        )

        print(
            "-" * 70
        )

        # -----------------------------------------------------------
        # Get corresponding S1 batch
        # -----------------------------------------------------------

        s1_batch = (
            s1_sub
            .iloc[
                start_idx:end_idx
            ]
            .copy()
        )

        s1_ids = (
            s1_batch[
                "entity_id"
            ]
            .astype(str)
            .tolist()
        )

        # -----------------------------------------------------------
        # Signal A dictionary for THIS BATCH ONLY
        # -----------------------------------------------------------

        candidates_a = {}

        for offset, candidate_list in enumerate(
            embedding_batch
        ):

            if offset >= len(s1_ids):
                break

            candidates_a[
                s1_ids[offset]
            ] = candidate_list

        # -----------------------------------------------------------
        # Signal B
        # -----------------------------------------------------------

        candidates_b = (
            generate_token_candidates_batch(
                s1_batch,
                target_sub,
                token_index,
                top_n_token=k
            )
        )

        # -----------------------------------------------------------
        # Signal C
        # -----------------------------------------------------------

        candidates_c = (
            generate_address_number_candidates_batch(
                s1_batch,
                target_sub,
                address_index,
                top_n_addr=ADDRESS_TOP_K
            )
        )

        # -----------------------------------------------------------
        # Union + ranking
        # -----------------------------------------------------------

        final_candidates = (
            union_and_rank_candidates(
                s1_ids,
                candidates_a,
                candidates_b,
                candidates_c,
                k=k
            )
        )

        # -----------------------------------------------------------
        # Write immediately
        # -----------------------------------------------------------

        append_candidates_to_output(
            output_file,
            s1_ids,
            final_candidates
        )

        # -----------------------------------------------------------
        # Statistics
        # -----------------------------------------------------------

        batch_pairs = sum(
            len(
                candidates
            )
            for candidates
            in final_candidates.values()
        )

        total_pairs += batch_pairs

        batch_count = len(
            s1_batch
        )

        total_s1_processed += batch_count

        elapsed = (
            time.time()
            - chunk_start_time
        )

        qps = (
            batch_count / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Batch S1 processed : "
            f"{batch_count:,}"
        )

        print(
            f"Batch pairs        : "
            f"{batch_pairs:,}"
        )

        print(
            f"Average candidates : "
            f"{batch_pairs / batch_count:.2f}"
            if batch_count > 0
            else "Average candidates : 0"
        )

        print(
            f"Batch time         : "
            f"{elapsed:.2f}s"
        )

        print(
            f"Batch throughput   : "
            f"{qps:,.0f} S1/sec"
        )

        print(
            f"Country progress   : "
            f"{total_s1_processed:,} / "
            f"{len(s1_sub):,}"
        )

        print(
            f"Country pairs      : "
            f"{total_pairs:,}"
        )

        # -----------------------------------------------------------
        # CRITICAL: free batch memory
        # -----------------------------------------------------------

        del s1_batch
        del s1_ids

        del embedding_batch

        del candidates_a
        del candidates_b
        del candidates_c

        del final_candidates

        cleanup_memory()

        chunk_number += 1

    # =================================================================
    # CLEAN COUNTRY-SPECIFIC MEMORY
    # =================================================================

    del token_index
    del address_index

    cleanup_memory()

    country_elapsed = (
        time.time()
        - country_start
    )

    country_qps = (
        len(s1_sub) / country_elapsed
        if country_elapsed > 0
        else 0
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"COUNTRY {country} COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"S1 processed : {total_s1_processed:,}"
    )

    print(
        f"Pairs        : {total_pairs:,}"
    )

    print(
        f"Time         : "
        f"{country_elapsed / 60:.2f} minutes"
    )

    print(
        f"Throughput   : "
        f"{country_qps:,.0f} S1/sec"
    )

    print(
        "=" * 80
    )

    return {
        "country": country,
        "s1_count": total_s1_processed,
        "pairs": total_pairs,
        "elapsed": country_elapsed
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_test_candidates(
    k: int = DEFAULT_K,
    min_sim: float = 0.20
):

    total_start_time = time.time()

    print(
        "\n"
        + "#" * 80
    )

    print(
        "# AMAZON ML CHALLENGE"
    )

    print(
        "# MEMORY-SAFE CANDIDATE GENERATION"
    )

    print(
        "#" * 80
    )

    print(
        f"Top-K                 : {k}"
    )

    print(
        f"Target chunk size     : "
        f"{TARGET_CHUNK_SIZE:,}"
    )

    print(
        f"Query chunk size      : "
        f"{QUERY_CHUNK_SIZE:,}"
    )

    print(
        f"Address candidates    : "
        f"{ADDRESS_TOP_K}"
    )

    print(
        f"Output                : "
        f"{OUTPUT_FILE}"
    )

    print(
        "#" * 80
    )

    # =================================================================
    # LOAD DATA
    # =================================================================

    (
        df_s1,
        df_targets
    ) = load_test_data()

    # =================================================================
    # COUNTRY PARTITIONING
    # =================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COUNTRY PARTITIONING"
    )

    print(
        "=" * 80
    )

    partitions = country_partition(
        df_s1,
        df_targets
    )

    print(
        f"\nDiscovered "
        f"{len(partitions)} country partitions:"
    )

    for country, part in partitions.items():

        print(
            f"  {country}: "
            f"S1={len(part['s1']):,}, "
            f"targets={len(part['targets']):,}"
        )

    # =================================================================
    # INITIALIZE OUTPUT
    # =================================================================

    initialize_output(
        OUTPUT_FILE
    )

    # =================================================================
    # PROCESS COUNTRIES
    # =================================================================

    country_results = []

    for country, part in partitions.items():

        result = process_country(
            country=country,
            s1_sub=part["s1"],
            target_sub=part["targets"],
            k=k,
            output_file=OUTPUT_FILE
        )

        country_results.append(
            result
        )

        cleanup_memory()

    # =================================================================
    # FINAL VALIDATION OF OUTPUT COVERAGE
    # =================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CHECKING OUTPUT COVERAGE"
    )

    print(
        "=" * 80
    )

    expected_ids = set(
        df_s1[
            "entity_id"
        ]
        .astype(str)
    )

    output_ids = set()

    try:

        output_df = pd.read_csv(
            OUTPUT_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False
        )

        output_ids = set(
            output_df[
                "source1_entity_id"
            ]
            .astype(str)
        )

        print(
            f"Expected S1 entities : "
            f"{len(expected_ids):,}"
        )

        print(
            f"Output S1 entities   : "
            f"{len(output_ids):,}"
        )

        missing = (
            expected_ids
            - output_ids
        )

        extra = (
            output_ids
            - expected_ids
        )

        print(
            f"Missing S1 entities  : "
            f"{len(missing):,}"
        )

        print(
            f"Unexpected IDs       : "
            f"{len(extra):,}"
        )

        if len(missing) == 0 and len(extra) == 0:

            print(
                "\n>>> OUTPUT COVERAGE: PASS <<<"
            )

        else:

            print(
                "\n>>> OUTPUT COVERAGE: WARNING <<<"
            )

            if missing:

                print(
                    "First missing IDs:"
                )

                for entity_id in list(
                    missing
                )[:10]:

                    print(
                        f"  {entity_id}"
                    )

    except Exception as exc:

        print(
            "Could not perform output coverage check."
        )

        print(
            f"Reason: {exc}"
        )

    # =================================================================
    # TOTAL STATISTICS
    # =================================================================

    total_elapsed = (
        time.time()
        - total_start_time
    )

    total_s1 = sum(
        result[
            "s1_count"
        ]
        for result in country_results
    )

    total_pairs = sum(
        result[
            "pairs"
        ]
        for result in country_results
    )

    avg_pairs = (
        total_pairs / total_s1
        if total_s1 > 0
        else 0
    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        "# CANDIDATE GENERATION COMPLETE"
    )

    print(
        "#" * 80
    )

    print(
        f"Total S1 processed : "
        f"{total_s1:,}"
    )

    print(
        f"Total pairs        : "
        f"{total_pairs:,}"
    )

    print(
        f"Average candidates : "
        f"{avg_pairs:.2f}"
    )

    print(
        f"Total runtime      : "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(
        f"Output file        : "
        f"{OUTPUT_FILE}"
    )

    print(
        "#" * 80
    )

    # =================================================================
    # VALIDATOR
    # =================================================================

    val_script = os.path.join(
        BASE_DIR,
        "utils",
        "validate_submission.py"
    )

    if os.path.isfile(
        val_script
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "RUNNING CANDIDATE VALIDATOR"
        )

        print(
            "=" * 80
        )

        cmd = [
            sys.executable,
            val_script,
            "--candidate",
            OUTPUT_FILE,
            "--test-dir",
            TEST_DIR
        ]

        print(
            "Command:"
        )

        print(
            " ".join(cmd)
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.stdout:

            print(
                "\nValidator output:"
            )

            print(
                result.stdout
            )

        if result.stderr:

            print(
                "\nValidator stderr:"
            )

            print(
                result.stderr
            )

        if result.returncode == 0:

            print(
                "\n>>> VALIDATOR STATUS: PASS <<<"
            )

        else:

            print(
                "\n>>> VALIDATOR STATUS: WARNING/FAIL <<<"
            )

    else:

        print(
            "\nValidator script not found:"
        )

        print(
            val_script
        )

    # =================================================================
    # RELEASE MEMORY
    # =================================================================

    del df_s1
    del df_targets
    del partitions

    cleanup_memory()


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe candidate generation "
            "for the Amazon ML Challenge."
        )
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=(
            "Final Top-K candidates per S1 "
            "(default: 20)"
        )
    )

    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.20,
        help=(
            "Legacy similarity parameter. "
            "Currently unused."
        )
    )

    args = parser.parse_args()

    generate_test_candidates(
        k=args.k,
        min_sim=args.min_sim
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
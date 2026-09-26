"""
High-Recall Scalable Blocking / Candidate Generation Engine

Signals:
    A. Multilingual E5 embeddings + FAISS IVF-PQ ANN retrieval
    B. Distinct brand token inverted index
    C. Physical address number co-occurrence

Pipeline:
    Country Partition
        ↓
    Signal A: Embedding + FAISS
        ↓
    Signal B: Brand Tokens
        ↓
    Signal C: Address Numbers
        ↓
    Union + Ranking
        ↓
    Top-K Candidates
"""

import os
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

IVF_NLIST = 2048
PQ_M = 48
PQ_NBITS = 8
NPROBE = 32

INDEX_VERSION = "ivf-pq-v2"


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed."
            )

        device = "cuda"

        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"

        except Exception:
            device = "cpu"

        print(
            f"[Embedding] Loading "
            f"{EMBEDDING_MODEL}"
        )

        print(
            f"[Embedding] Device: {device}"
        )

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device
        )

    return _embedding_model


# =============================================================================
# COUNTRY PARTITIONING
# =============================================================================

def country_partition(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame
):

    partitions = {}

    if "country" not in df_s1.columns:
        raise ValueError(
            "S1 dataframe must contain 'country'"
        )

    if "country" not in df_targets.columns:
        raise ValueError(
            "Target dataframe must contain 'country'"
        )

    countries = sorted(
        set(df_s1["country"].astype(str))
    )

    for country in countries:

        s1_mask = (
            df_s1["country"].astype(str)
            == str(country)
        )

        target_mask = (
            df_targets["country"].astype(str)
            == str(country)
        )

        s1_sub = df_s1.loc[
            s1_mask
        ].copy()

        target_sub = df_targets.loc[
            target_mask
        ].copy()

        partitions[country] = {
            "s1": s1_sub,
            "targets": target_sub
        }

    return partitions


# =============================================================================
# FAISS CHECKPOINT HELPERS
# =============================================================================

def _atomic_write_faiss_index(
    index,
    path: str
):

    tmp_path = path + ".tmp"

    # If index is on GPU, move it back to CPU
    try:
        cpu_index = faiss.index_gpu_to_cpu(index)
    except Exception:
        cpu_index = index

    faiss.write_index(
        cpu_index,
        tmp_path
    )

    os.replace(
        tmp_path,
        path
    )


def _load_faiss_index(
    path: str
):

    if not os.path.isfile(path):
        return None

    return faiss.read_index(
        path
    )


def _save_metadata(
    metadata: dict,
    path: str
):

    tmp_path = path + ".tmp"

    with open(
        tmp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    os.replace(
        tmp_path,
        path
    )


def _load_metadata(
    path: str
):

    if not os.path.isfile(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =============================================================================
# CREATE GPU INDEX
# =============================================================================

def _create_gpu_index(
    cpu_index
):

    if not faiss.get_num_gpus():
        return cpu_index

    try:

        res = faiss.StandardGpuResources()

        gpu_index = faiss.index_cpu_to_gpu(
            res,
            0,
            cpu_index
        )

        return gpu_index

    except Exception as e:

        print(
            f"[FAISS] GPU conversion failed: "
            f"{e}"
        )

        return cpu_index


# =============================================================================
# TRAIN IVF-PQ
# =============================================================================

def _train_ivf_pq(
    target_texts: List[str]
):

    model = get_embedding_model()

    sample_size = min(
        100_000,
        len(target_texts)
    )

    train_texts = target_texts[
        :sample_size
    ]

    print(
        f"[Signal A] Training IVF-PQ "
        f"on first {sample_size:,} target names"
    )

    embeddings = model.encode(
        [
            "passage: " + str(x)
            for x in train_texts
        ],
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    print(
        f"[Signal A] Embedding dimension: "
        f"{dimension}"
    )

    if dimension % PQ_M != 0:

        raise ValueError(
            f"Embedding dimension {dimension} "
            f"is not divisible by PQ_M={PQ_M}"
        )

    quantizer = faiss.IndexFlatIP(
        dimension
    )

    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        IVF_NLIST,
        PQ_M,
        PQ_NBITS,
        faiss.METRIC_INNER_PRODUCT
    )

    print(
        "[Signal A] Training IVF-PQ..."
    )

    index.train(
        embeddings
    )

    print(
        "[Signal A] IVF-PQ training complete."
    )

    return index


# =============================================================================
# SIGNAL A
# MULTILINGUAL E5 + FAISS IVF-PQ
# =============================================================================

def generate_embedding_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_k: int = 20,
    target_chunk_size: int = 100_000,
    query_chunk_size: int = 10_000,
    checkpoint_dir: str = None
):

    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss is required for "
            "embedding candidate generation."
        )

    print("\n" + "=" * 80)
    print(
        "SIGNAL A — IVF-PQ EMBEDDING BLOCKING"
    )
    print("=" * 80)

    print(
        f"S1 records     : {len(df_s1):,}"
    )

    print(
        f"Target records : {len(df_targets):,}"
    )

    print(
        f"Top-K          : {top_k}"
    )

    print(
        f"Target chunk   : {target_chunk_size:,}"
    )

    print(
        f"Query chunk    : {query_chunk_size:,}"
    )

    print(
        f"nlist          : {IVF_NLIST}"
    )

    print(
        f"PQ m           : {PQ_M}"
    )

    print(
        f"nprobe         : {NPROBE}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    if checkpoint_dir is None:

        checkpoint_dir = "./signal_a_checkpoint"

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    index_path = os.path.join(
        checkpoint_dir,
        "targets_ivfpq.faiss"
    )

    metadata_path = os.path.join(
        checkpoint_dir,
        "metadata.json"
    )

    query_dir = os.path.join(
        checkpoint_dir,
        "query_chunks"
    )

    os.makedirs(
        query_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # VALIDATE INPUT
    # -------------------------------------------------------------------------

    required_s1 = [
        "entity_id",
        "clean_name"
    ]

    required_target = [
        "entity_id",
        "clean_name"
    ]

    for col in required_s1:

        if col not in df_s1.columns:
            raise ValueError(
                f"S1 missing column: {col}"
            )

    for col in required_target:

        if col not in df_targets.columns:
            raise ValueError(
                f"Target missing column: {col}"
            )

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    model = get_embedding_model()

    # -------------------------------------------------------------------------
    # LOAD OR CREATE FAISS INDEX
    # -------------------------------------------------------------------------

    print(
        "\n[Signal A] Loading FAISS checkpoint:"
    )

    print(
        f"            {index_path}"
    )

    index = _load_faiss_index(
        index_path
    )

    metadata = _load_metadata(
        metadata_path
    )

    total_targets = len(
        df_targets
    )

    if index is not None:

        print(
            f"[Signal A] Checkpoint vectors: "
            f"{index.ntotal:,}"
        )

        compatible = True

        if metadata is not None:

            if metadata.get(
                "embedding_model"
            ) != EMBEDDING_MODEL:

                compatible = False

            if metadata.get(
                "total_targets"
            ) != total_targets:

                compatible = False

            if metadata.get(
                "target_chunk_size"
            ) != target_chunk_size:

                compatible = False

            if metadata.get(
                "ivf_nlist"
            ) != IVF_NLIST:

                compatible = False

            if metadata.get(
                "pq_m"
            ) != PQ_M:

                compatible = False

            if metadata.get(
                "pq_nbits"
            ) != PQ_NBITS:

                compatible = False

        if not compatible:

            print(
                "[Signal A] Existing checkpoint "
                "is incompatible."
            )

            print(
                "[Signal A] Rebuilding index."
            )

            index = None

        else:

            print(
                f"[Signal A] Existing index contains "
                f"{index.ntotal:,} vectors."
            )

    # -------------------------------------------------------------------------
    # CREATE INDEX IF NEEDED
    # -------------------------------------------------------------------------

    if index is None:

        target_texts = (
            df_targets["clean_name"]
            .astype(str)
            .tolist()
        )

        index = _train_ivf_pq(
            target_texts
        )

        metadata = {
            "index_version": INDEX_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "total_targets": total_targets,
            "target_chunk_size": target_chunk_size,
            "ivf_nlist": IVF_NLIST,
            "pq_m": PQ_M,
            "pq_nbits": PQ_NBITS,
            "ntotal": 0
        }

    # -------------------------------------------------------------------------
    # PHASE 1
    # TARGET INDEXING
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "PHASE 1 — TARGET INDEXING"
    )
    print("=" * 80)

    indexed_count = int(
        index.ntotal
    )

    if indexed_count > total_targets:

        raise RuntimeError(
            "Checkpoint contains more vectors "
            "than current target dataset."
        )

    if indexed_count < total_targets:

        print(
            f"[Signal A] Resuming from "
            f"{indexed_count:,}"
        )

        gpu_index = _create_gpu_index(
            index
        )

        try:

            gpu_index.nprobe = NPROBE

        except Exception:
            pass

        for start_idx in range(
            indexed_count,
            total_targets,
            target_chunk_size
        ):

            end_idx = min(
                start_idx + target_chunk_size,
                total_targets
            )

            print(
                f"\n[Signal A] Target chunk "
                f"{start_idx:,} → {end_idx:,}"
            )

            chunk_texts = (
                df_targets["clean_name"]
                .iloc[
                    start_idx:end_idx
                ]
                .astype(str)
                .tolist()
            )

            t_chunk = time.time()

            embeddings = model.encode(
                [
                    "passage: " + x
                    for x in chunk_texts
                ],
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            gpu_index.add(
                embeddings
            )

            print(
                f"[Signal A] Added "
                f"{len(embeddings):,} vectors "
                f"in {time.time() - t_chunk:.2f}s"
            )

            # Save checkpoint
            _atomic_write_faiss_index(
                gpu_index,
                index_path
            )

            metadata["ntotal"] = int(
                end_idx
            )

            _save_metadata(
                metadata,
                metadata_path
            )

            print(
                f"[Signal A] Checkpoint saved: "
                f"{end_idx:,}/{total_targets:,}"
            )

        index = _load_faiss_index(
            index_path
        )

    else:

        print(
            "\n[Signal A] ALL TARGETS INDEXED"
        )

    # -------------------------------------------------------------------------
    # PHASE 2
    # QUERY SEARCH
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "PHASE 2 — S1 ANN SEARCH"
    )
    print("=" * 80)

    index = _load_faiss_index(
        index_path
    )

    gpu_index = _create_gpu_index(
        index
    )

    try:

        gpu_index.nprobe = NPROBE

    except Exception:
        pass

    total_queries = len(
        df_s1
    )

    # -------------------------------------------------------------------------
    # SEARCH QUERY CHUNKS
    # -------------------------------------------------------------------------

    for start_idx in range(
        0,
        total_queries,
        query_chunk_size
    ):

        end_idx = min(
            start_idx + query_chunk_size,
            total_queries
        )

        chunk_number = (
            start_idx // query_chunk_size
        )

        chunk_path = os.path.join(
            query_dir,
            f"query_{chunk_number:06d}.npz"
        )

        # Already completed
        if os.path.isfile(
            chunk_path
        ):

            print(
                f"[Signal A] Query chunk "
                f"{chunk_number} already exists. "
                f"Skipping."
            )

            continue

        print(
            f"\n[Signal A] Query chunk "
            f"{chunk_number}"
        )

        print(
            f"           rows "
            f"{start_idx:,} → {end_idx:,}"
        )

        query_texts = (
            df_s1["clean_name"]
            .iloc[
                start_idx:end_idx
            ]
            .astype(str)
            .tolist()
        )

        t_query = time.time()

        query_embeddings = model.encode(
            [
                "query: " + x
                for x in query_texts
            ],
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embeddings = np.asarray(
            query_embeddings,
            dtype=np.float32
        )

        scores, indices = gpu_index.search(
            query_embeddings,
            top_k
        )

        print(
            f"[Signal A] Query completed "
            f"in {time.time() - t_query:.2f}s"
        )

        print(
            f"[Signal A] Progress: "
            f"{end_idx:,}/{total_queries:,} "
            f"({end_idx / total_queries * 100:.2f}%)"
        )

        # ---------------------------------------------------------------------
        # IMPORTANT: SAFE NPZ SAVE
        # ---------------------------------------------------------------------

        tmp_path = chunk_path + ".tmp"

        with open(
            tmp_path,
            "wb"
        ) as f:

            np.savez(
                f,
                scores=scores.astype(
                    np.float32
                ),
                indices=indices.astype(
                    np.int64
                ),
                start_idx=np.int64(
                    start_idx
                ),
                end_idx=np.int64(
                    end_idx
                )
            )

        os.replace(
            tmp_path,
            chunk_path
        )

        print(
            f"[Signal A] Saved query chunk: "
            f"{chunk_path}"
        )

    # -------------------------------------------------------------------------
    # PHASE 3
    # BUILD CANDIDATE DICTIONARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "PHASE 3 — BUILD SIGNAL-A CANDIDATES"
    )
    print("=" * 80)

    candidates = defaultdict(list)

    target_ids = (
        df_targets["entity_id"]
        .astype(str)
        .tolist()
    )

    s1_ids = (
        df_s1["entity_id"]
        .astype(str)
        .tolist()
    )

    query_files = sorted(
        [
            x
            for x in os.listdir(query_dir)
            if x.endswith(".npz")
            and x.startswith("query_")
        ]
    )

    for query_file in query_files:

        path = os.path.join(
            query_dir,
            query_file
        )

        data = np.load(
            path
        )

        scores = data["scores"]
        indices = data["indices"]

        start_idx = int(
            data["start_idx"]
        )

        for local_idx in range(
            len(scores)
        ):

            s1_index = (
                start_idx + local_idx
            )

            if s1_index >= len(s1_ids):
                continue

            s1_id = s1_ids[
                s1_index
            ]

            for j in range(
                len(scores[local_idx])
            ):

                target_index = int(
                    indices[
                        local_idx,
                        j
                    ]
                )

                if target_index < 0:
                    continue

                if target_index >= len(
                    target_ids
                ):
                    continue

                target_id = target_ids[
                    target_index
                ]

                score = float(
                    scores[
                        local_idx,
                        j
                    ]
                )

                candidates[s1_id].append(
                    (
                        target_id,
                        score
                    )
                )

    print(
        f"[Signal A] S1 entities: "
        f"{len(s1_ids):,}"
    )

    print(
        f"[Signal A] Candidate pairs: "
        f"{sum(len(v) for v in candidates.values()):,}"
    )

    print(
        f"[Signal A] Average candidates/S1: "
        f"{sum(len(v) for v in candidates.values()) / len(s1_ids):.2f}"
    )

    print(
        "\n[Signal A] COMPLETE."
    )

    return dict(
        candidates
    )


# =============================================================================
# SIGNAL B
# DISTINCT BRAND TOKEN INVERTED INDEX
# =============================================================================

GENERIC_STOP_TOKENS = {
    "the",
    "and",
    "of",
    "for",
    "a",
    "an",
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "llp",
    "pvt",
    "private",
    "priv",
    "services",
    "service",
    "group",
    "international",
    "india",
    "usa",
    "us"
}


def generate_token_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_token: int = 10,
    max_token_postings: int = 5000
):

    print(
        f"    [Signal B: Token Index] "
        f"Building inverted index on "
        f"{len(df_targets):,} target names..."
    )

    inverted_index = defaultdict(list)

    target_names = (
        df_targets["clean_name"]
        .astype(str)
        .tolist()
    )

    target_ids = (
        df_targets["entity_id"]
        .astype(str)
        .tolist()
    )

    # -------------------------------------------------------------------------
    # BUILD INDEX
    # -------------------------------------------------------------------------

    for idx, name in enumerate(
        target_names
    ):

        tokens = set(
            name.lower().split()
        )

        tokens = {
            token
            for token in tokens
            if token
            and token not in GENERIC_STOP_TOKENS
        }

        for token in tokens:

            inverted_index[token].append(
                idx
            )

    # Remove overly common tokens
    for token in list(
        inverted_index.keys()
    ):

        if len(
            inverted_index[token]
        ) > max_token_postings:

            del inverted_index[token]

    # -------------------------------------------------------------------------
    # QUERY
    # -------------------------------------------------------------------------

    print(
        f"    [Signal B: Token Index] "
        f"Querying {len(df_s1):,} anchors..."
    )

    candidates = defaultdict(list)

    t0 = time.time()

    for _, row in df_s1.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        name = str(
            row["clean_name"]
        )

        tokens = set(
            name.lower().split()
        )

        tokens = {
            token
            for token in tokens
            if token
            and token not in GENERIC_STOP_TOKENS
        }

        scores = defaultdict(float)

        for token in tokens:

            postings = inverted_index.get(
                token,
                []
            )

            if not postings:
                continue

            weight = 1.0 / np.log1p(
                len(postings)
            )

            for target_idx in postings:

                scores[target_idx] += weight

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n_token]

        candidates[s1_id] = [
            (
                target_ids[idx],
                float(score)
            )
            for idx, score in ranked
        ]

    elapsed = time.time() - t0

    anchors_with_candidates = sum(
        1
        for v in candidates.values()
        if v
    )

    print(
        f"    [Signal B: Token Index] "
        f"Complete in {elapsed:.2f}s "
        f"({len(df_s1) / elapsed:,.0f} queries/sec)"
    )

    print(
        f"    Anchors with candidates: "
        f"{anchors_with_candidates:,}"
    )

    return dict(
        candidates
    )


# =============================================================================
# SIGNAL C
# PHYSICAL ADDRESS NUMBER CO-OCCURRENCE
# =============================================================================

def generate_address_number_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_addr: int = 8,
    max_num_postings: int = 2000
):

    print(
        f"    [Signal C: Address Numbers] "
        f"Indexing physical address numbers on "
        f"{len(df_targets):,} targets..."
    )

    inverted_index = defaultdict(list)

    target_ids = (
        df_targets["entity_id"]
        .astype(str)
        .tolist()
    )

    # -------------------------------------------------------------------------
    # BUILD NUMBER INDEX
    # -------------------------------------------------------------------------

    for idx, row in df_targets.reset_index(drop=True).iterrows():

        numbers = str(
            row.get(
                "address_numbers",
                ""
            )
        ).split()

        numbers = {
            n
            for n in numbers
            if n
        }

        for number in numbers:

            inverted_index[
                number
            ].append(
                idx
            )

    # Remove very common numbers
    for number in list(
        inverted_index.keys()
    ):

        if len(
            inverted_index[number]
        ) > max_num_postings:

            del inverted_index[number]

    # -------------------------------------------------------------------------
    # QUERY
    # -------------------------------------------------------------------------

    print(
        f"    [Signal C: Address Numbers] "
        f"Querying {len(df_s1):,} anchors..."
    )

    candidates = defaultdict(list)

    t0 = time.time()

    for _, row in df_s1.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        numbers = str(
            row.get(
                "address_numbers",
                ""
            )
        ).split()

        numbers = {
            n
            for n in numbers
            if n
        }

        scores = defaultdict(float)

        for number in numbers:

            postings = inverted_index.get(
                number,
                []
            )

            for target_idx in postings:

                scores[target_idx] += 1.0

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n_addr]

        candidates[s1_id] = [
            (
                target_ids[idx],
                float(score)
            )
            for idx, score in ranked
        ]

    elapsed = time.time() - t0

    anchors_with_candidates = sum(
        1
        for v in candidates.values()
        if v
    )

    print(
        f"    [Signal C: Address Numbers] "
        f"Complete in {elapsed:.2f}s "
        f"({len(df_s1) / elapsed:,.0f} queries/sec)"
    )

    print(
        f"    Anchors with candidates: "
        f"{anchors_with_candidates:,}"
    )

    return dict(
        candidates
    )


# =============================================================================
# UNION + RANK + TOP-K
# =============================================================================

def union_and_rank_candidates(
    s1_ids,
    candidates_tfidf,
    candidates_token,
    candidates_addr,
    k: int = 20
):

    final_candidates = {}

    # Signal weights
    WEIGHT_A = 1.0
    WEIGHT_B = 0.35
    WEIGHT_C = 0.25

    for s1_id in s1_ids:

        scores = defaultdict(float)

        # ---------------------------------------------------------------------
        # SIGNAL A
        # ---------------------------------------------------------------------

        for target_id, score in candidates_tfidf.get(
            s1_id,
            []
        ):

            scores[
                target_id
            ] += WEIGHT_A * float(score)

        # ---------------------------------------------------------------------
        # SIGNAL B
        # ---------------------------------------------------------------------

        for target_id, score in candidates_token.get(
            s1_id,
            []
        ):

            scores[
                target_id
            ] += WEIGHT_B * float(score)

        # ---------------------------------------------------------------------
        # SIGNAL C
        # ---------------------------------------------------------------------

        for target_id, score in candidates_addr.get(
            s1_id,
            []
        ):

            scores[
                target_id
            ] += WEIGHT_C * float(score)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_candidates[s1_id] = [
            target_id
            for target_id, _ in ranked[:k]
        ]

    return final_candidates


# =============================================================================
# CANDIDATE RECALL EVALUATION
# =============================================================================

def evaluate_candidate_recall(
    candidate_dict,
    ground_truth_dict,
    total_s1_count,
    total_target_count
):

    total_true_matches = 0
    found_true_matches = 0

    for s1_id, true_targets in ground_truth_dict.items():

        true_targets = set(
            true_targets
        )

        candidate_targets = set(
            candidate_dict.get(
                s1_id,
                []
            )
        )

        total_true_matches += len(
            true_targets
        )

        found_true_matches += len(
            true_targets
            & candidate_targets
        )

    recall = (
        found_true_matches / total_true_matches
        if total_true_matches > 0
        else 0.0
    )

    return {
        "total_s1": total_s1_count,
        "total_targets": total_target_count,
        "total_true_matches": total_true_matches,
        "found_true_matches": found_true_matches,
        "candidate_recall": recall
    }
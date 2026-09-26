"""
Memory-Safe Blocking / Candidate Generation
Amazon ML Challenge - Business Entity Resolution

Signals
-------
A: multilingual-e5-small + FAISS IVF-PQ
B: Brand-token inverted index
C: Address-number inverted index

Design
------
The expensive FAISS index/search is checkpointed to disk.

Candidate generation is performed in batches so that we do NOT keep
millions of Python tuples/dictionaries in RAM.

Existing FAISS/query checkpoints are reused whenever possible.
"""

import os
import json
import time
import gc
from collections import defaultdict
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------

try:
    import faiss
except ImportError:
    faiss = None

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# FAISS IVF-PQ configuration
IVF_NLIST = 2048
PQ_M = 48
PQ_NBITS = 8
NPROBE = 32

INDEX_VERSION = "ivf-pq-v2"

# Checkpoint sizes
DEFAULT_TARGET_CHUNK_SIZE = 100_000
DEFAULT_QUERY_CHUNK_SIZE = 10_000

# Candidate limits
DEFAULT_TOKEN_TOP_K = 20
DEFAULT_ADDRESS_TOP_K = 15

# Final signal weights
WEIGHT_A = 1.0
WEIGHT_B = 0.35
WEIGHT_C = 0.25

# Generic tokens that should not be useful as brand tokens
GENERIC_STOP_TOKENS = {
    "the",
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "at",
    "to",
    "a",
    "an",
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "pvt",
    "private",
    "plc",
    "llp",
    "india",
    "ind",
    "usa",
    "us",
    "france",
    "fr",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def _require_faiss():
    if faiss is None:
        raise ImportError(
            "FAISS is not installed. "
            "Install faiss-cpu or the appropriate FAISS package."
        )


def _require_embedding_dependencies():
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed."
        )

    if torch is None:
        raise ImportError(
            "PyTorch is not installed."
        )


def _cleanup_memory():
    """
    Aggressively release Python / CUDA memory after a batch.
    """
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _safe_str(value) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value)


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

_EMBEDDING_MODEL = None


def get_embedding_model():
    """
    Load the multilingual E5 model once.

    Uses CUDA when available.
    """
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    _require_embedding_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Embedding device: {device}"
    )

    _EMBEDDING_MODEL = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    return _EMBEDDING_MODEL


def _encode_passages(
    texts: List[str],
    batch_size: int = 256
) -> np.ndarray:
    """
    Encode target/business names using E5 passage format.
    """
    model = get_embedding_model()

    prepared = [
        "passage: " + _safe_str(x)
        for x in texts
    ]

    embeddings = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


def _encode_queries(
    texts: List[str],
    batch_size: int = 256
) -> np.ndarray:
    """
    Encode S1 names using E5 query format.
    """
    model = get_embedding_model()

    prepared = [
        "query: " + _safe_str(x)
        for x in texts
    ]

    embeddings = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# =============================================================================
# COUNTRY PARTITIONING
# =============================================================================

def country_partition(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Partition S1 and target records by country.

    Country remains open-set; no hard-coded country list is used.
    """

    if "country" not in df_s1.columns:
        raise ValueError(
            "S1 dataframe must contain 'country'."
        )

    if "country" not in df_targets.columns:
        raise ValueError(
            "Target dataframe must contain 'country'."
        )

    partitions = {}

    countries = sorted(
        set(
            df_s1["country"].astype(str)
        )
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

        s1_part = df_s1.loc[
            s1_mask
        ].copy()

        target_part = df_targets.loc[
            target_mask
        ].copy()

        partitions[str(country)] = {
            "s1": s1_part,
            "targets": target_part
        }

    return partitions


# =============================================================================
# FAISS CHECKPOINT HELPERS
# =============================================================================

def _atomic_write_faiss_index(
    index,
    path: str
):
    """
    Atomically save FAISS index.
    """
    tmp_path = path + ".tmp"

    faiss.write_index(
        index,
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

    print(
        f"Loading FAISS index: {path}"
    )

    return faiss.read_index(path)


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
) -> Optional[dict]:

    if not os.path.isfile(path):
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
# GPU INDEX HELPERS
# =============================================================================

def _create_gpu_index(cpu_index):
    """
    Move FAISS index to GPU when a GPU is available.

    Returns:
        gpu_index, gpu_resources
    """

    if not torch.cuda.is_available():
        return cpu_index, None

    try:

        print(
            "Moving FAISS index to GPU..."
        )

        resources = faiss.StandardGpuResources()

        gpu_index = faiss.index_cpu_to_gpu(
            resources,
            0,
            cpu_index
        )

        return gpu_index, resources

    except Exception as exc:

        print(
            "WARNING: Could not move FAISS index to GPU."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Continuing with CPU FAISS."
        )

        return cpu_index, None


def _move_index_to_cpu(index):
    """
    Safely convert a GPU FAISS index back to CPU.
    """
    try:
        if hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(index)
    except Exception:
        pass

    return index


# =============================================================================
# IVF-PQ TRAINING
# =============================================================================

def _train_ivf_pq(
    df_targets: pd.DataFrame
):
    """
    Train IVF-PQ using up to 100k target names.
    """

    _require_faiss()

    print(
        "\nTraining FAISS IVF-PQ index..."
    )

    train_size = min(
        100_000,
        len(df_targets)
    )

    train_names = (
        df_targets[
            "clean_name"
        ]
        .astype(str)
        .iloc[:train_size]
        .tolist()
    )

    train_embeddings = _encode_passages(
        train_names
    )

    dimension = train_embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"Training vectors: {len(train_embeddings):,}"
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
        f"IVF nlist: {IVF_NLIST}"
    )

    print(
        f"PQ m: {PQ_M}"
    )

    print(
        f"PQ nbits: {PQ_NBITS}"
    )

    start = time.time()

    index.train(
        train_embeddings
    )

    elapsed = time.time() - start

    print(
        f"IVF-PQ training complete in "
        f"{elapsed:.2f}s"
    )

    index.nprobe = NPROBE

    del train_embeddings

    _cleanup_memory()

    return index


# =============================================================================
# QUERY CHECKPOINT VALIDATION
# =============================================================================

def _query_chunk_path(
    checkpoint_dir: str,
    start_idx: int
) -> str:

    return os.path.join(
        checkpoint_dir,
        "query_chunks",
        f"query_{start_idx:06d}.npz"
    )


def _is_valid_query_checkpoint(
    path: str,
    expected_start: int,
    expected_end: int,
    top_k: int
) -> bool:
    """
    Validate a query checkpoint.

    This prevents an old 4,619-row smoke-test checkpoint from
    being incorrectly accepted as a complete 10,000-row chunk.
    """

    if not os.path.isfile(path):
        return False

    try:

        data = np.load(
            path
        )

        scores = data["scores"]
        indices = data["indices"]

        start_idx = int(
            data["start_idx"]
        )

        end_idx = int(
            data["end_idx"]
        )

        expected_rows = (
            expected_end
            - expected_start
        )

        if start_idx != expected_start:
            return False

        if end_idx != expected_end:
            return False

        if scores.shape != (
            expected_rows,
            top_k
        ):
            return False

        if indices.shape != (
            expected_rows,
            top_k
        ):
            return False

        return True

    except Exception:
        return False


# =============================================================================
# SIGNAL A - BUILD / CHECKPOINT FAISS INDEX
# =============================================================================

def build_or_resume_embedding_index(
    df_targets: pd.DataFrame,
    checkpoint_dir: str,
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE
):
    """
    Build or resume the country-specific IVF-PQ target index.

    Existing completed target chunks are reused.
    """

    _require_faiss()

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    query_dir = os.path.join(
        checkpoint_dir,
        "query_chunks"
    )

    os.makedirs(
        query_dir,
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

    total_targets = len(
        df_targets
    )

    # -----------------------------------------------------------------
    # Check existing metadata
    # -----------------------------------------------------------------

    metadata = _load_metadata(
        metadata_path
    )

    compatible = False

    if metadata is not None:

        compatible = (
            metadata.get(
                "index_version"
            ) == INDEX_VERSION
            and
            metadata.get(
                "embedding_model"
            ) == EMBEDDING_MODEL
            and
            int(
                metadata.get(
                    "total_targets",
                    -1
                )
            ) == total_targets
            and
            int(
                metadata.get(
                    "target_chunk_size",
                    -1
                )
            ) == target_chunk_size
            and
            int(
                metadata.get(
                    "ivf_nlist",
                    -1
                )
            ) == IVF_NLIST
            and
            int(
                metadata.get(
                    "pq_m",
                    -1
                )
            ) == PQ_M
            and
            int(
                metadata.get(
                    "pq_nbits",
                    -1
                )
            ) == PQ_NBITS
        )

    # -----------------------------------------------------------------
    # Load compatible index
    # -----------------------------------------------------------------

    if compatible and os.path.isfile(index_path):

        print(
            "\nExisting compatible FAISS checkpoint found."
        )

        index = _load_faiss_index(
            index_path
        )

        if index is None:
            compatible = False

    else:

        if metadata is not None:
            print(
                "\nExisting FAISS checkpoint is incompatible."
            )

        print(
            "Creating a new IVF-PQ index."
        )

        index = _train_ivf_pq(
            df_targets
        )

        metadata = {
            "index_version": INDEX_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "total_targets": total_targets,
            "target_chunk_size": target_chunk_size,
            "ivf_nlist": IVF_NLIST,
            "pq_m": PQ_M,
            "pq_nbits": PQ_NBITS,
            "metric": "inner_product",
            "indexed_targets": 0,
            "last_completed_chunk": -1
        }

        _save_metadata(
            metadata,
            metadata_path
        )

    # -----------------------------------------------------------------
    # Resume indexing
    # -----------------------------------------------------------------

    indexed_targets = int(
        metadata.get(
            "indexed_targets",
            0
        )
    )

    last_completed_chunk = int(
        metadata.get(
            "last_completed_chunk",
            -1
        )
    )

    if indexed_targets > total_targets:
        print(
            "WARNING: indexed_targets exceeds target count."
        )

        indexed_targets = 0
        last_completed_chunk = -1

    if indexed_targets < total_targets:

        print(
            "\nSignal A Phase 1: "
            "Indexing target embeddings"
        )

        print(
            f"Already indexed: "
            f"{indexed_targets:,} / {total_targets:,}"
        )

        # CPU index used for persistent checkpoint
        cpu_index = index

        for start_idx in range(
            indexed_targets,
            total_targets,
            target_chunk_size
        ):

            end_idx = min(
                start_idx + target_chunk_size,
                total_targets
            )

            chunk_number = (
                start_idx // target_chunk_size
            )

            print(
                f"\nTarget chunk "
                f"{chunk_number}: "
                f"{start_idx:,} -> {end_idx:,}"
            )

            names = (
                df_targets[
                    "clean_name"
                ]
                .astype(str)
                .iloc[
                    start_idx:end_idx
                ]
                .tolist()
            )

            embeddings = _encode_passages(
                names
            )

            # GPU acceleration for adding this chunk
            gpu_index, gpu_resources = (
                _create_gpu_index(
                    cpu_index
                )
            )

            gpu_index.add(
                embeddings
            )

            if gpu_resources is not None:
                cpu_index = faiss.index_gpu_to_cpu(
                    gpu_index
                )

            else:
                cpu_index = gpu_index

            del embeddings
            del names

            _cleanup_memory()

            indexed_targets = end_idx
            last_completed_chunk = chunk_number

            metadata.update({
                "indexed_targets": indexed_targets,
                "last_completed_chunk": last_completed_chunk
            })

            _atomic_write_faiss_index(
                cpu_index,
                index_path
            )

            _save_metadata(
                metadata,
                metadata_path
            )

            print(
                f"Checkpoint saved: "
                f"{indexed_targets:,} / "
                f"{total_targets:,}"
            )

        index = cpu_index

    else:

        print(
            "\nSignal A Phase 1 already complete."
        )

        print(
            f"Indexed targets: "
            f"{indexed_targets:,}"
        )

    return index


# =============================================================================
# SIGNAL A - GENERATE / RESUME QUERY CHUNKS
# =============================================================================

def generate_embedding_query_checkpoints(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    index,
    checkpoint_dir: str,
    top_k: int = 20,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE
):
    """
    Generate FAISS S1 query chunks.

    Existing VALID chunks are skipped.

    Invalid/stale chunks are regenerated.

    Important:
    The results are saved to disk rather than kept in RAM.
    """

    _require_faiss()

    query_dir = os.path.join(
        checkpoint_dir,
        "query_chunks"
    )

    os.makedirs(
        query_dir,
        exist_ok=True
    )

    total_queries = len(
        df_s1
    )

    print(
        "\nSignal A Phase 2: "
        "Generating query checkpoints"
    )

    print(
        f"S1 queries: {total_queries:,}"
    )

    print(
        f"Query chunk size: {query_chunk_size:,}"
    )

    print(
        f"Top-K: {top_k}"
    )

    # ---------------------------------------------------------------
    # GPU copy of index
    # ---------------------------------------------------------------

    search_index, gpu_resources = (
        _create_gpu_index(
            index
        )
    )

    try:

        search_index.nprobe = NPROBE

    except Exception:
        pass

    # ---------------------------------------------------------------
    # Process query chunks
    # ---------------------------------------------------------------

    for start_idx in range(
        0,
        total_queries,
        query_chunk_size
    ):

        end_idx = min(
            start_idx + query_chunk_size,
            total_queries
        )

        path = _query_chunk_path(
            checkpoint_dir,
            start_idx
        )

        # -----------------------------------------------------------
        # Validate existing checkpoint
        # -----------------------------------------------------------

        if _is_valid_query_checkpoint(
            path,
            start_idx,
            end_idx,
            top_k
        ):

            print(
                f"Query chunk "
                f"{start_idx:,}:{end_idx:,} "
                f"already valid -> SKIP"
            )

            continue

        # -----------------------------------------------------------
        # Delete stale checkpoint if present
        # -----------------------------------------------------------

        if os.path.isfile(path):

            print(
                f"Invalid/stale checkpoint found: "
                f"{os.path.basename(path)}"
            )

            print(
                "Regenerating..."
            )

            try:
                os.remove(path)
            except Exception:
                pass

        print(
            f"\nQuery chunk: "
            f"{start_idx:,} -> {end_idx:,}"
        )

        names = (
            df_s1[
                "clean_name"
            ]
            .astype(str)
            .iloc[
                start_idx:end_idx
            ]
            .tolist()
        )

        embeddings = _encode_queries(
            names
        )

        # -----------------------------------------------------------
        # FAISS search
        # -----------------------------------------------------------

        scores, indices = search_index.search(
            embeddings,
            top_k
        )

        # -----------------------------------------------------------
        # Atomic NPZ checkpoint
        # -----------------------------------------------------------

        temp_path = path + ".tmp"

        np.savez(
            temp_path,
            scores=np.asarray(
                scores,
                dtype=np.float32
            ),
            indices=np.asarray(
                indices,
                dtype=np.int64
            ),
            start_idx=np.int64(
                start_idx
            ),
            end_idx=np.int64(
                end_idx
            )
        )

        # np.savez adds .npz if not already present
        actual_temp = (
            temp_path
            if os.path.isfile(temp_path)
            else temp_path + ".npz"
        )

        os.replace(
            actual_temp,
            path
        )

        print(
            f"Saved: "
            f"{os.path.basename(path)}"
        )

        del embeddings
        del scores
        del indices
        del names

        _cleanup_memory()

    print(
        "\nSignal A Phase 2 complete."
    )


# =============================================================================
# SIGNAL A - STREAM QUERY RESULTS
# =============================================================================

def stream_embedding_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    checkpoint_dir: str,
    top_k: int = 20,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE
) -> Iterator[
    Tuple[int, int, List[Tuple[str, float]]]
]:
    """
    Stream Signal-A candidate results one query chunk at a time.

    Yields:
        start_idx,
        end_idx,
        candidates

    candidates is a list with one entry per S1 in this chunk.

    IMPORTANT:
    Only one query chunk exists in RAM at a time.
    """

    query_dir = os.path.join(
        checkpoint_dir,
        "query_chunks"
    )

    total_queries = len(
        df_s1
    )

    target_ids = (
        df_targets[
            "entity_id"
        ]
        .astype(str)
        .tolist()
    )

    s1_ids = (
        df_s1[
            "entity_id"
        ]
        .astype(str)
        .tolist()
    )

    for start_idx in range(
        0,
        total_queries,
        query_chunk_size
    ):

        end_idx = min(
            start_idx + query_chunk_size,
            total_queries
        )

        path = _query_chunk_path(
            checkpoint_dir,
            start_idx
        )

        if not _is_valid_query_checkpoint(
            path,
            start_idx,
            end_idx,
            top_k
        ):
            raise RuntimeError(
                f"Missing or invalid Signal A checkpoint: "
                f"{path}"
            )

        data = np.load(
            path
        )

        scores = data[
            "scores"
        ]

        indices = data[
            "indices"
        ]

        chunk_candidates = []

        for local_idx in range(
            len(scores)
        ):

            candidates = []

            row_scores = scores[
                local_idx
            ]

            row_indices = indices[
                local_idx
            ]

            for j in range(
                len(row_scores)
            ):

                target_index = int(
                    row_indices[j]
                )

                if (
                    target_index < 0
                    or
                    target_index >= len(target_ids)
                ):
                    continue

                target_id = target_ids[
                    target_index
                ]

                score = float(
                    row_scores[j]
                )

                candidates.append(
                    (
                        target_id,
                        score
                    )
                )

            chunk_candidates.append(
                candidates
            )

        yield (
            start_idx,
            end_idx,
            chunk_candidates
        )

        del data
        del scores
        del indices
        del chunk_candidates

        _cleanup_memory()


# =============================================================================
# SIGNAL A - CONVENIENCE FUNCTION
# =============================================================================

def generate_embedding_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_k: int = 20,
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    checkpoint_dir: str = None
):
    """
    Backwards-compatible Signal-A function.

    WARNING:
    This function returns a dictionary and therefore should only be used
    for small/smoke-test datasets.

    The full test pipeline should use the streaming functions above.
    """

    if checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir is required."
        )

    print(
        "\n"
        "WARNING: generate_embedding_candidates() "
        "returns all candidates in RAM."
    )

    print(
        "For the full test dataset use "
        "stream_embedding_candidates()."
    )

    index = build_or_resume_embedding_index(
        df_targets,
        checkpoint_dir,
        target_chunk_size
    )

    generate_embedding_query_checkpoints(
        df_s1,
        df_targets,
        index,
        checkpoint_dir,
        top_k,
        query_chunk_size
    )

    candidates = {}

    s1_ids = (
        df_s1[
            "entity_id"
        ]
        .astype(str)
        .tolist()
    )

    for (
        start_idx,
        end_idx,
        chunk_candidates
    ) in stream_embedding_candidates(
        df_s1,
        df_targets,
        checkpoint_dir,
        top_k,
        query_chunk_size
    ):

        for offset, candidate_list in enumerate(
            chunk_candidates
        ):

            s1_id = s1_ids[
                start_idx + offset
            ]

            candidates[
                s1_id
            ] = candidate_list

    return candidates


# =============================================================================
# SIGNAL B - TOKEN HELPERS
# =============================================================================

def _tokenize_name(
    text: str
) -> List[str]:

    text = _safe_str(
        text
    ).lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        text
    )

    result = []

    for token in tokens:

        if len(token) < 2:
            continue

        if token in GENERIC_STOP_TOKENS:
            continue

        result.append(
            token
        )

    return result


# =============================================================================
# SIGNAL B - BUILD INDEX
# =============================================================================

def build_token_inverted_index(
    df_targets: pd.DataFrame,
    max_postings: int = 5000
):
    """
    Build token -> target positional-index mapping.

    Very common tokens are ignored.
    """

    print(
        "\nBuilding Signal B token inverted index..."
    )

    inverted_index = defaultdict(
        list
    )

    target_names = (
        df_targets[
            "clean_name"
        ]
        .astype(str)
        .tolist()
    )

    for target_idx, name in enumerate(
        target_names
    ):

        tokens = set(
            _tokenize_name(
                name
            )
        )

        for token in tokens:

            postings = inverted_index[
                token
            ]

            if len(postings) <= max_postings:
                postings.append(
                    target_idx
                )

    # Remove very common tokens
    filtered_index = {}

    for token, postings in inverted_index.items():

        if (
            len(postings)
            <= max_postings
        ):
            filtered_index[
                token
            ] = postings

    del inverted_index
    del target_names

    _cleanup_memory()

    print(
        f"Signal B index tokens: "
        f"{len(filtered_index):,}"
    )

    return filtered_index


# =============================================================================
# SIGNAL B - BATCH GENERATION
# =============================================================================

def generate_token_candidates_batch(
    df_s1_batch: pd.DataFrame,
    df_targets: pd.DataFrame,
    inverted_index,
    top_n_token: int = DEFAULT_TOKEN_TOP_K
):
    """
    Generate Signal-B candidates for one S1 batch only.

    Returns:
        dict[s1_id] = [(target_id, score), ...]
    """

    target_ids = (
        df_targets[
            "entity_id"
        ]
        .astype(str)
        .tolist()
    )

    candidates = {}

    for _, row in df_s1_batch.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        tokens = set(
            _tokenize_name(
                row["clean_name"]
            )
        )

        if not tokens:

            candidates[
                s1_id
            ] = []

            continue

        scores = defaultdict(
            int
        )

        for token in tokens:

            postings = inverted_index.get(
                token,
                []
            )

            for target_idx in postings:

                scores[
                    target_idx
                ] += 1

        ranked = sorted(
            scores.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        )[
            :top_n_token
        ]

        result = []

        for target_idx, score in ranked:

            if (
                0 <= target_idx
                < len(target_ids)
            ):

                result.append(
                    (
                        target_ids[
                            target_idx
                        ],
                        float(score)
                    )
                )

        candidates[
            s1_id
        ] = result

    return candidates


# =============================================================================
# SIGNAL B - BACKWARDS COMPATIBLE FULL FUNCTION
# =============================================================================

def generate_token_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_token: int = DEFAULT_TOKEN_TOP_K
):
    """
    Backwards-compatible Signal-B function.

    For the full test dataset, use build_token_inverted_index()
    + generate_token_candidates_batch().
    """

    inverted_index = build_token_inverted_index(
        df_targets
    )

    candidates = generate_token_candidates_batch(
        df_s1,
        df_targets,
        inverted_index,
        top_n_token
    )

    del inverted_index

    _cleanup_memory()

    return candidates


# =============================================================================
# SIGNAL C - ADDRESS NUMBER EXTRACTION
# =============================================================================

def _extract_address_numbers(
    text: str
) -> List[str]:

    text = _safe_str(
        text
    )

    return re.findall(
        r"\d+",
        text
    )


# =============================================================================
# SIGNAL C - BUILD INDEX
# =============================================================================

def build_address_number_index(
    df_targets: pd.DataFrame,
    max_postings: int = 2000
):
    """
    Build address-number -> target positional-index mapping.
    """

    print(
        "\nBuilding Signal C address-number index..."
    )

    inverted_index = defaultdict(
        list
    )

    target_addresses = (
        df_targets[
            "clean_address"
        ]
        .astype(str)
        .tolist()
    )

    for target_idx, address in enumerate(
        target_addresses
    ):

        numbers = set(
            _extract_address_numbers(
                address
            )
        )

        for number in numbers:

            postings = inverted_index[
                number
            ]

            if len(postings) < max_postings:

                postings.append(
                    target_idx
                )

    filtered_index = {}

    for number, postings in inverted_index.items():

        if (
            len(postings)
            <= max_postings
        ):

            filtered_index[
                number
            ] = postings

    del inverted_index
    del target_addresses

    _cleanup_memory()

    print(
        f"Signal C index keys: "
        f"{len(filtered_index):,}"
    )

    return filtered_index


# =============================================================================
# SIGNAL C - BATCH GENERATION
# =============================================================================

def generate_address_number_candidates_batch(
    df_s1_batch: pd.DataFrame,
    df_targets: pd.DataFrame,
    inverted_index,
    top_n_addr: int = DEFAULT_ADDRESS_TOP_K
):
    """
    Generate Signal-C candidates for one S1 batch.
    """

    target_ids = (
        df_targets[
            "entity_id"
        ]
        .astype(str)
        .tolist()
    )

    candidates = {}

    for _, row in df_s1_batch.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        numbers = set(
            _extract_address_numbers(
                row["clean_address"]
            )
        )

        if not numbers:

            candidates[
                s1_id
            ] = []

            continue

        scores = defaultdict(
            int
        )

        for number in numbers:

            postings = inverted_index.get(
                number,
                []
            )

            for target_idx in postings:

                scores[
                    target_idx
                ] += 1

        ranked = sorted(
            scores.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        )[
            :top_n_addr
        ]

        result = []

        for target_idx, score in ranked:

            if (
                0 <= target_idx
                < len(target_ids)
            ):

                result.append(
                    (
                        target_ids[
                            target_idx
                        ],
                        float(score)
                    )
                )

        candidates[
            s1_id
        ] = result

    return candidates


# =============================================================================
# SIGNAL C - BACKWARDS COMPATIBLE FUNCTION
# =============================================================================

def generate_address_number_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_addr: int = DEFAULT_ADDRESS_TOP_K
):
    """
    Backwards-compatible Signal-C function.
    """

    inverted_index = build_address_number_index(
        df_targets
    )

    candidates = (
        generate_address_number_candidates_batch(
            df_s1,
            df_targets,
            inverted_index,
            top_n_addr
        )
    )

    del inverted_index

    _cleanup_memory()

    return candidates


# =============================================================================
# UNION + RANK
# =============================================================================

def union_and_rank_candidates(
    s1_ids: List[str],
    candidates_a: Dict[str, List[Tuple[str, float]]],
    candidates_b: Dict[str, List[Tuple[str, float]]],
    candidates_c: Dict[str, List[Tuple[str, float]]],
    k: int = 20
):
    """
    Union candidates from Signals A/B/C and rank them.

    This function operates on ONE BATCH only.

    Therefore the returned dictionary should remain small.
    """

    final_candidates = {}

    for s1_id in s1_ids:

        combined = {}

        # -------------------------------------------------------------
        # Signal A
        # -------------------------------------------------------------

        for target_id, score in candidates_a.get(
            s1_id,
            []
        ):

            combined.setdefault(
                target_id,
                {
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0
                }
            )

            combined[
                target_id
            ]["a"] = float(score)

        # -------------------------------------------------------------
        # Signal B
        # -------------------------------------------------------------

        for target_id, score in candidates_b.get(
            s1_id,
            []
        ):

            combined.setdefault(
                target_id,
                {
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0
                }
            )

            combined[
                target_id
            ]["b"] = float(score)

        # -------------------------------------------------------------
        # Signal C
        # -------------------------------------------------------------

        for target_id, score in candidates_c.get(
            s1_id,
            []
        ):

            combined.setdefault(
                target_id,
                {
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0
                }
            )

            combined[
                target_id
            ]["c"] = float(score)

        # -------------------------------------------------------------
        # Final weighted score
        # -------------------------------------------------------------

        ranked = []

        for target_id, signals in combined.items():

            final_score = (
                WEIGHT_A * signals["a"]
                +
                WEIGHT_B * signals["b"]
                +
                WEIGHT_C * signals["c"]
            )

            ranked.append(
                (
                    target_id,
                    final_score
                )
            )

        ranked.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        final_candidates[
            str(s1_id)
        ] = [
            target_id
            for target_id, _ in ranked[:k]
        ]

    return final_candidates


# =============================================================================
# STREAMING UNION HELPER
# =============================================================================

def combine_candidate_batches(
    s1_ids: List[str],
    candidates_a: Dict[str, List[Tuple[str, float]]],
    candidates_b: Dict[str, List[Tuple[str, float]]],
    candidates_c: Dict[str, List[Tuple[str, float]]],
    k: int = 20
):
    """
    Alias for batch union/ranking.

    Kept separate so the caller can clearly express that this
    operation is batch-scoped.
    """

    return union_and_rank_candidates(
        s1_ids,
        candidates_a,
        candidates_b,
        candidates_c,
        k
    )


# =============================================================================
# CANDIDATE RECALL EVALUATION
# =============================================================================

def evaluate_candidate_recall(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
):
    """
    Evaluate whether true matches appear in generated candidates.

    Recall =
        true matches present in candidate set
        /
        total true matches
    """

    total_true = 0
    found_true = 0

    for s1_id, true_targets in ground_truth.items():

        true_targets = [
            str(x)
            for x in true_targets
            if str(x).strip()
        ]

        candidate_targets = set(
            str(x)
            for x in candidates.get(
                str(s1_id),
                []
            )
        )

        total_true += len(
            true_targets
        )

        for target_id in true_targets:

            if target_id in candidate_targets:
                found_true += 1

    recall = (
        found_true / total_true
        if total_true > 0
        else 0.0
    )

    return {
        "total_true_matches": total_true,
        "found_true_matches": found_true,
        "candidate_recall": recall
    }


# =============================================================================
# DEBUG / INFORMATION
# =============================================================================

def print_configuration():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BLOCKING CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Embedding model : {EMBEDDING_MODEL}"
    )

    print(
        f"IVF nlist       : {IVF_NLIST}"
    )

    print(
        f"PQ m            : {PQ_M}"
    )

    print(
        f"PQ nbits        : {PQ_NBITS}"
    )

    print(
        f"nprobe          : {NPROBE}"
    )

    print(
        f"Index version   : {INDEX_VERSION}"
    )

    if torch is not None:

        print(
            f"CUDA available  : "
            f"{torch.cuda.is_available()}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU             : "
                f"{torch.cuda.get_device_name(0)}"
            )

    print(
        "=" * 70
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print_configuration()
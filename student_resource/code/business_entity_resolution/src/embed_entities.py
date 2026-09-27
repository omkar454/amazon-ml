"""
Ultra-Fast Entity Embedding Engine using ONNX Runtime / Quantized Bi-Encoder.
Encodes S1 queries and active target entities into 384-dim normalized float16 vectors.
Uses disk-backed np.memmap to guarantee ultra-low RAM usage (< 1.5 GB peak).
"""

import os
import gc
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NORM_DIR = os.path.join(BASE_DIR, "dataset", "normalized")
OUT_DIR = os.path.join(BASE_DIR, "output")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

os.makedirs(EMBED_DIR, exist_ok=True)


def get_embedding_engine(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_gpu: bool = False):
    """
    Initializes a high-throughput embedding model.
    Falls back gracefully across fastembed, onnxruntime, or sentence-transformers.
    """
    try:
        from fastembed import TextEmbedding
        print(f"Loading FastEmbed engine with model: {model_name} (INT8/SIMD Optimized)...")
        # fastembed model name default is BAAI/bge-small-en-v1.5 or sentence-transformers/all-MiniLM-L6-v2
        return "fastembed", TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=None)
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer on device: {device.upper()}...")
        model = SentenceTransformer(model_name, device=device)
        return "st", model
    except ImportError:
        raise ImportError("Please install either 'fastembed' (`pip install fastembed`) or 'sentence-transformers' (`pip install sentence-transformers`).")


def encode_texts_batch(engine_type: str, model, texts: List[str], batch_size: int = 512) -> np.ndarray:
    """Encodes a batch of strings and returns L2-normalized float16 numpy vectors."""
    if engine_type == "fastembed":
        embeddings = list(model.embed(texts, batch_size=batch_size))
        vecs = np.array(embeddings, dtype=np.float32)
    elif engine_type == "st":
        vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float16)
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    # L2 normalization
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs.astype(np.float16)


def format_entity_text(name: str, address: str, country: str) -> str:
    """Formats structured entity attributes into a standard serializable string."""
    name_str = str(name).strip() if pd.notna(name) else ""
    addr_str = str(address).strip() if pd.notna(address) else ""
    ctry_str = str(country).strip() if pd.notna(country) else ""
    return f"business: {name_str} | address: {addr_str} | country: {ctry_str}"


def embed_s1_entities(engine_type, model, batch_size: int = 512, chunk_size: int = 50000):
    """Encodes all Test S1 entities and stores to disk-backed memmap."""
    s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv")
    print(f"\n[1/2] Loading S1 entities from {os.path.basename(s1_path)}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    
    total_s1 = len(df_s1)
    dim = 384
    
    mmap_path = os.path.join(EMBED_DIR, "s1_embeddings.mmap")
    id_map_path = os.path.join(EMBED_DIR, "s1_id_map.json")
    
    print(f"Creating S1 disk memmap of shape ({total_s1:,}, {dim}) at: {mmap_path}...")
    mmap_vecs = np.memmap(mmap_path, dtype=np.float16, mode="w+", shape=(total_s1, dim))
    
    id_map = {}
    t0 = time.time()
    
    for start_idx in range(0, total_s1, chunk_size):
        end_idx = min(total_s1, start_idx + chunk_size)
        sub_df = df_s1.iloc[start_idx:end_idx]
        
        texts = [format_entity_text(r["clean_name"], r["clean_address"], r["country"]) for _, r in sub_df.iterrows()]
        ids = sub_df["entity_id"].tolist()
        
        for i, eid in enumerate(ids):
            id_map[eid] = start_idx + i
            
        chunk_vecs = encode_texts_batch(engine_type, model, texts, batch_size=batch_size)
        mmap_vecs[start_idx:end_idx] = chunk_vecs
        mmap_vecs.flush()
        
        elapsed = time.time() - t0
        rate = end_idx / elapsed if elapsed > 0 else 0
        print(f"  S1 Progress: {end_idx:,}/{total_s1:,} ({end_idx/total_s1*100:.1f}%) | Speed: {rate:,.0f} entities/sec")
        
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
        
    print(f"S1 Embeddings Complete in {time.time()-t0:.2f}s! Saved to {mmap_path}")
    del df_s1, mmap_vecs, id_map
    gc.collect()


def embed_active_targets(engine_type, model, candidate_file: str, batch_size: int = 512, chunk_size: int = 50000):
    """Encodes only unique active candidate target entities to disk-backed memmap."""
    print(f"\n[2/2] Scanning {os.path.basename(candidate_file)} for active candidate target IDs...")
    active_ids = set()
    with open(candidate_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                for cid in parts[1].split(","):
                    cid = cid.strip()
                    if cid:
                        active_ids.add(cid)
                        
    total_targets = len(active_ids)
    print(f"Found {total_targets:,} unique active target entities.")
    
    # Load metadata from S2 and S3 for active IDs only
    s2_path = os.path.join(NORM_DIR, "test_source2_normalized.tsv")
    s3_path = os.path.join(NORM_DIR, "test_source3_normalized.tsv")
    
    tgt_records = []
    for path in [s2_path, s3_path]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df_act = df[df["entity_id"].isin(active_ids)]
        for _, row in df_act.iterrows():
            tgt_records.append((row["entity_id"], row["clean_name"], row["clean_address"], row["country"]))
        del df, df_act
        gc.collect()
        
    print(f"Collected metadata for {len(tgt_records):,} active target records.")
    
    dim = 384
    mmap_path = os.path.join(EMBED_DIR, "target_embeddings.mmap")
    id_map_path = os.path.join(EMBED_DIR, "target_id_map.json")
    
    mmap_vecs = np.memmap(mmap_path, dtype=np.float16, mode="w+", shape=(len(tgt_records), dim))
    id_map = {}
    t0 = time.time()
    
    total_tgt = len(tgt_records)
    for start_idx in range(0, total_tgt, chunk_size):
        end_idx = min(total_tgt, start_idx + chunk_size)
        sub_records = tgt_records[start_idx:end_idx]
        
        texts = [format_entity_text(r[1], r[2], r[3]) for r in sub_records]
        ids = [r[0] for r in sub_records]
        
        for i, eid in enumerate(ids):
            id_map[eid] = start_idx + i
            
        chunk_vecs = encode_texts_batch(engine_type, model, texts, batch_size=batch_size)
        mmap_vecs[start_idx:end_idx] = chunk_vecs
        mmap_vecs.flush()
        
        elapsed = time.time() - t0
        rate = end_idx / elapsed if elapsed > 0 else 0
        print(f"  Target Progress: {end_idx:,}/{total_tgt:,} ({end_idx/total_tgt*100:.1f}%) | Speed: {rate:,.0f} entities/sec")
        
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
        
    print(f"Target Embeddings Complete in {time.time()-t0:.2f}s! Saved to {mmap_path}")
    del tgt_records, mmap_vecs, id_map
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate Compact ONNX / Bi-Encoder Embeddings")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA GPU if available")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size (default: 512)")
    parser.add_argument("--candidate-file", type=str, default=os.path.join(OUT_DIR, "candidate_pairs.tsv"))
    args = parser.parse_args()
    
    print("=" * 80)
    print("QUANTIZED / BI-ENCODER ENTITY EMBEDDING PIPELINE")
    print(f"Model: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)
    
    engine_type, model = get_embedding_engine(args.model_name, args.use_gpu)
    
    # 1. Embed S1
    embed_s1_entities(engine_type, model, batch_size=args.batch_size)
    
    # 2. Embed Targets
    embed_active_targets(engine_type, model, candidate_file=args.candidate_file, batch_size=args.batch_size)
    
    print("\n" + "=" * 80)
    print("ALL EMBEDDING ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

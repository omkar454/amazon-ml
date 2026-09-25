"""
High-Recall Scalable Blocking / Candidate Generation Engine
Multi-signal candidate generation across:
1. Dynamic Country Partitioning (US, India, France, etc.)
2. Signal A: Character 3-Gram Sparse TF-IDF Cosine Retrieval
3. Signal B: Distinct Brand Token Inverted Index
4. Signal C: Physical Address Number Co-occurrence
5. Candidate Union, Scoring, and Top-K Selection
"""

import os
import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Generic non-discriminative stop tokens to ignore in Brand Token Index
GENERIC_STOP_TOKENS = {
    'and', 'the', 'of', 'for', 'in', 'at', 'by', 'with', 'from', 'to', 'on',
    'services', 'service', 'enterprises', 'enterprise', 'traders', 'trading',
    'solutions', 'technologies', 'technology', 'consultants', 'consulting',
    'management', 'group', 'india', 'national', 'international', 'global',
    'center', 'centre', 'associates', 'holdings', 'holding', 'industries',
    'industry', 'products', 'product', 'corporation', 'corp', 'company', 'co',
    'limited', 'ltd', 'pvt', 'private', 'llc', 'llp', 'inc', 'sarl', 'sas',
    'sasu', 'eurl', 'et', 'fils'
}

# ==============================================================================
# 1. DYNAMIC COUNTRY PARTITIONING
# ==============================================================================

def country_partition(df_s1: pd.DataFrame, df_targets: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Dynamically partitions S1 anchors and S2/S3 targets by country.
    Open-set: automatically discovers countries present (US, INDIA, FRANCE, etc.).
    """
    partitions = {}
    unique_countries = df_s1['country'].unique()
    
    for c in unique_countries:
        s1_subset = df_s1[df_s1['country'] == c].reset_index(drop=True)
        target_subset = df_targets[df_targets['country'] == c].reset_index(drop=True)
        
        partitions[c] = {
            "s1": s1_subset,
            "targets": target_subset
        }
        
    return partitions


# ==============================================================================
# 2. SIGNAL A: CHARACTER 3-GRAM TF-IDF COSINE RETRIEVAL
# ==============================================================================

def generate_3gram_tfidf_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_tfidf: int = 15,
    min_similarity: float = 0.22,
    batch_size: int = 200
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes Top-N character 3-gram cosine similarity matches between S1 and Targets.
    Returns: {s1_id: [(target_id, score), ...]}
    """
    print(f"    [Signal A: TF-IDF] Vectorizing {len(df_targets):,} target names...")
    t0 = time.time()
    
    # Fit vectorizer on target clean_name
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 3),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        dtype=np.float32
    )
    
    target_names = df_targets['clean_name'].fillna("").values
    X_targets = vectorizer.fit_transform(target_names)
    X_targets_T = X_targets.T.tocsc()
    
    s1_names = df_s1['clean_name'].fillna("").values
    s1_ids = df_s1['entity_id'].values
    target_ids = df_targets['entity_id'].values
    
    candidates_tfidf = defaultdict(list)
    
    print(f"    [Signal A: TF-IDF] Querying {len(df_s1):,} anchors in batches of {batch_size:,}...")
    
    num_batches = (len(df_s1) + batch_size - 1) // batch_size
    for b_idx in range(num_batches):
        start_i = b_idx * batch_size
        end_i = min(len(df_s1), (b_idx + 1) * batch_size)
        
        batch_names = s1_names[start_i:end_i]
        X_s1_batch = vectorizer.transform(batch_names)
        
        # Batch dot product: (batch_size x num_targets)
        sim_matrix = X_s1_batch.dot(X_targets_T)
        
        indptr = sim_matrix.indptr
        indices = sim_matrix.indices
        data = sim_matrix.data
        
        for row_idx in range(end_i - start_i):
            global_s1_idx = start_i + row_idx
            s1_id = s1_ids[global_s1_idx]
            
            row_start = indptr[row_idx]
            row_end = indptr[row_idx + 1]
            if row_start == row_end:
                continue
                
            cols = indices[row_start:row_end]
            row_data = data[row_start:row_end]
            
            # Filter by minimum similarity
            valid_mask = row_data >= min_similarity
            if not np.any(valid_mask):
                continue
                
            valid_cols = cols[valid_mask]
            valid_data = row_data[valid_mask]
            
            # Select top-N
            if len(valid_data) > top_n_tfidf:
                top_indices = np.argpartition(valid_data, -top_n_tfidf)[-top_n_tfidf:]
                top_indices = top_indices[np.argsort(-valid_data[top_indices])]
                sel_cols = valid_cols[top_indices]
                sel_scores = valid_data[top_indices]
            else:
                sort_order = np.argsort(-valid_data)
                sel_cols = valid_cols[sort_order]
                sel_scores = valid_data[sort_order]
                
            for col, score in zip(sel_cols, sel_scores):
                candidates_tfidf[s1_id].append((target_ids[col], float(score)))
                
        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == num_batches:
            processed = end_i
            cur_elapsed = time.time() - t0
            print(f"      [TF-IDF] Processed {processed:,}/{len(df_s1):,} ({processed/len(df_s1)*100:.1f}%) in {cur_elapsed:.1f}s...")
                
    elapsed = time.time() - t0
    print(f"    [Signal A: TF-IDF] Complete in {elapsed:.2f}s ({len(df_s1)/elapsed:,.0f} queries/sec). Anchors with candidates: {len(candidates_tfidf):,}")
    return candidates_tfidf


# ==============================================================================
# 3. SIGNAL B: BRAND TOKEN INVERTED INDEX
# ==============================================================================

def generate_token_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_token: int = 10,
    max_token_postings: int = 5000
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Builds inverted index on informative brand tokens and queries matching records.
    Returns: {s1_id: [(target_id, score), ...]}
    """
    print(f"    [Signal B: Token Index] Building inverted index on {len(df_targets):,} target names...")
    t0 = time.time()
    
    target_ids = df_targets['entity_id'].values
    target_names = df_targets['clean_name'].fillna("").values
    
    token_index = defaultdict(list)
    for idx, name in enumerate(target_names):
        words = set(name.split())
        for w in words:
            if len(w) >= 3 and w not in GENERIC_STOP_TOKENS:
                token_index[w].append(idx)
                
    # Filter out tokens with too many postings to avoid memory explosion on generic tokens
    filtered_index = {k: v for k, v in token_index.items() if len(v) <= max_token_postings}
    
    s1_ids = df_s1['entity_id'].values
    s1_names = df_s1['clean_name'].fillna("").values
    
    candidates_token = defaultdict(list)
    print(f"    [Signal B: Token Index] Querying {len(df_s1):,} anchors...")
    
    for s1_idx, name in enumerate(s1_names):
        s1_id = s1_ids[s1_idx]
        words = [w for w in set(name.split()) if len(w) >= 3 and w in filtered_index]
        if not words:
            continue
            
        # Count hits per target
        target_hits = defaultdict(float)
        for w in words:
            postings = filtered_index[w]
            # IDF-like weight: rarer words have higher weight
            weight = 1.0 / np.log1p(len(postings))
            for t_idx in postings:
                target_hits[t_idx] += weight
                
        if not target_hits:
            continue
            
        # Sort targets by score and take top_n_token
        sorted_hits = sorted(target_hits.items(), key=lambda x: x[1], reverse=True)[:top_n_token]
        for t_idx, score in sorted_hits:
            candidates_token[s1_id].append((target_ids[t_idx], float(score)))
            
    elapsed = time.time() - t0
    print(f"    [Signal B: Token Index] Complete in {elapsed:.2f}s ({len(df_s1)/elapsed:,.0f} queries/sec). Anchors with candidates: {len(candidates_token):,}")
    return candidates_token


# ==============================================================================
# 4. SIGNAL C: PHYSICAL ADDRESS NUMBER CO-OCCURRENCE
# ==============================================================================

def generate_address_number_candidates(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    top_n_addr: int = 8,
    max_num_postings: int = 2000
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Builds inverted index on address numbers and retrieves candidate pairs sharing exact physical premise numbers.
    Returns: {s1_id: [(target_id, score), ...]}
    """
    print(f"    [Signal C: Address Numbers] Indexing physical address numbers on {len(df_targets):,} targets...")
    t0 = time.time()
    
    target_ids = df_targets['entity_id'].values
    target_nums = df_targets['address_numbers'].fillna("").values
    target_names = df_targets['clean_name'].fillna("").values
    
    num_index = defaultdict(list)
    for idx, num_str in enumerate(target_nums):
        if not num_str:
            continue
        nums = [n.strip() for n in num_str.split(",") if n.strip() and len(n.strip()) >= 2]
        for n in nums:
            num_index[n].append(idx)
            
    filtered_num_index = {k: v for k, v in num_index.items() if len(v) <= max_num_postings}
    
    s1_ids = df_s1['entity_id'].values
    s1_nums = df_s1['address_numbers'].fillna("").values
    s1_names = df_s1['clean_name'].fillna("").values
    
    candidates_addr = defaultdict(list)
    print(f"    [Signal C: Address Numbers] Querying {len(df_s1):,} anchors...")
    
    for s1_idx, num_str in enumerate(s1_nums):
        s1_id = s1_ids[s1_idx]
        if not num_str:
            continue
            
        nums = [n.strip() for n in num_str.split(",") if n.strip() and n.strip() in filtered_num_index]
        if not nums:
            continue
            
        s1_name_words = set(s1_names[s1_idx].split())
        
        target_hits = defaultdict(float)
        for n in nums:
            postings = filtered_num_index[n]
            for t_idx in postings:
                # Require at least 1 common character 2-gram or first-letter match to avoid random number collisions
                t_name = target_names[t_idx]
                if s1_name_words and set(t_name.split()) & s1_name_words:
                    target_hits[t_idx] += 1.0
                elif len(s1_names[s1_idx]) > 0 and len(t_name) > 0 and s1_names[s1_idx][0] == t_name[0]:
                    target_hits[t_idx] += 0.5
                else:
                    target_hits[t_idx] += 0.25
                    
        if not target_hits:
            continue
            
        sorted_hits = sorted(target_hits.items(), key=lambda x: x[1], reverse=True)[:top_n_addr]
        for t_idx, score in sorted_hits:
            candidates_addr[s1_id].append((target_ids[t_idx], float(score)))
            
    elapsed = time.time() - t0
    print(f"    [Signal C: Address Numbers] Complete in {elapsed:.2f}s ({len(df_s1)/elapsed:,.0f} queries/sec). Anchors with candidates: {len(candidates_addr):,}")
    return candidates_addr


# ==============================================================================
# 5. CANDIDATE UNION, SCORING & TOP-K SELECTION
# ==============================================================================

def union_and_rank_candidates(
    all_s1_ids: List[str],
    candidates_tfidf: Dict[str, List[Tuple[str, float]]],
    candidates_token: Dict[str, List[Tuple[str, float]]],
    candidates_addr: Dict[str, List[Tuple[str, float]]],
    k: int = 15
) -> Dict[str, List[str]]:
    """
    Unions candidates from all signals, scores them using a composite priority metric, and slices Top-K.
    Returns: {s1_id: [target_id_1, target_id_2, ...]}
    """
    final_candidates = {}
    
    for s1_id in all_s1_ids:
        candidate_scores = defaultdict(float)
        
        # 1. Signal A: TF-IDF Score
        for tid, score in candidates_tfidf.get(s1_id, []):
            candidate_scores[tid] += score * 1.0
            
        # 2. Signal B: Token Match Score
        for tid, score in candidates_token.get(s1_id, []):
            candidate_scores[tid] += score * 0.35
            
        # 3. Signal C: Address Number Match Score
        for tid, score in candidates_addr.get(s1_id, []):
            candidate_scores[tid] += score * 0.25
            
        if not candidate_scores:
            final_candidates[s1_id] = []
            continue
            
        # Sort by composite score descending and take top-K
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        final_candidates[s1_id] = [tid for tid, _ in sorted_candidates]
        
    return final_candidates


# ==============================================================================
# 6. EVALUATION METRICS FOR HYPERPARAMETER TUNING
# ==============================================================================

def evaluate_candidate_recall(
    candidate_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, Set[str]],
    total_s1_count: int,
    total_target_count: int
) -> dict:
    """
    Evaluates candidate recall, total pairs, average candidates per entity, and reduction ratio.
    """
    total_true_matches = sum(len(matches) for matches in ground_truth_dict.values())
    found_true_matches = 0
    total_candidate_pairs = 0
    singletons_correct = 0
    singletons_total = 0
    
    for s1_id, true_matches in ground_truth_dict.items():
        candidates = set(candidate_dict.get(s1_id, []))
        total_candidate_pairs += len(candidates)
        
        if len(true_matches) == 0:
            singletons_total += 1
            if len(candidates) == 0:
                singletons_correct += 1
        else:
            hits = len(true_matches & candidates)
            found_true_matches += hits
            
    recall = found_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_candidates = total_candidate_pairs / total_s1_count if total_s1_count > 0 else 0.0
    
    total_search_space = total_s1_count * total_target_count
    reduction_ratio = 1.0 - (total_candidate_pairs / total_search_space) if total_search_space > 0 else 1.0
    
    return {
        "candidate_recall": recall,
        "found_true_matches": found_true_matches,
        "total_true_matches": total_true_matches,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_entity": avg_candidates,
        "reduction_ratio": reduction_ratio,
        "singletons_total": singletons_total
    }

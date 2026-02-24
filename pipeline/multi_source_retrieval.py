"""
pipeline/multi_source_retrieval.py

This module implements multi-source evidence retrieval for DeepSieve.
It retrieves candidates from multiple sources and aggregates them for evidence selection.
"""

import time
from typing import List, Dict, Tuple, Optional


def retrieve_multi_source(
    actual_query: str,
    sources: List[Tuple[str, any]],  # List of (source_id, rag_instance) pairs
    top_k_per_source: int = 5
) -> List[Dict]:
    """
    Retrieve candidates from multiple sources.
    
    Args:
        actual_query: The subquery text
        sources: List of (source_id, rag_instance) tuples
        top_k_per_source: Number of candidates to retrieve per source
    
    Returns:
        List of candidate dictionaries, each containing:
        - source_id: str
        - text: str (passage text)
        - score: float or None
        - rank_within_source: int (0-based)
    """
    all_candidates = []
    
    for source_id, rag_instance in sources:
        if rag_instance is None:
            print(f"⚠️  Source '{source_id}' is None, skipping")
            continue
            
        try:
            # Call the RAG instance's retrieval method
            retrieved = rag_instance.rag_qa(actual_query, k=top_k_per_source)
            
            docs = retrieved.get("docs", [])
            doc_scores = retrieved.get("doc_scores", [])
            
            # Add each document as a candidate
            for rank, (doc_text, score) in enumerate(zip(docs, doc_scores)):
                all_candidates.append({
                    "source_id": source_id,
                    "text": doc_text,
                    "score": float(score) if score is not None else None,
                    "rank_within_source": rank
                })
            
            print(f"  📚 Source '{source_id}': retrieved {len(docs)} candidates")
            
        except Exception as e:
            print(f"⚠️  Error retrieving from source '{source_id}': {str(e)}")
            # Continue with other sources even if one fails
            continue
    
    print(f"  ✅ Total candidates from all sources: {len(all_candidates)}")
    return all_candidates

"""
pipeline/evidence_selector.py

This module implements evidence selection strategies for multi-source retrieval.
"""

from typing import List, Dict, Optional
import numpy as np


def _apply_per_source_cap(candidates: List[Dict], per_source_cap: int,
                          preferred_source: str = "", preferred_cap: int = 0, other_cap: int = 0) -> List[Dict]:
    """
    Apply per-source cap. Supports two modes:
    - Fixed cap: all sources get `per_source_cap`
    - Adaptive cap: preferred_source gets `preferred_cap`, others get `other_cap`
    """
    adaptive = preferred_source and preferred_cap > 0 and other_cap > 0

    by_source = {}
    for cand in candidates:
        src = cand.get("source_id", "unknown")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(cand)

    capped = []
    for src, cands in by_source.items():
        sorted_cands = sorted(
            cands,
            key=lambda x: (x.get("score") if x.get("score") is not None else -1),
            reverse=True
        )
        if adaptive:
            cap = preferred_cap if src.lower() == preferred_source.lower() else other_cap
        else:
            cap = per_source_cap
        capped.extend(sorted_cands[:cap])

    return capped


def select_evidence(
    query: str,
    candidates: List[Dict],
    keep_k: int,
    selector: str = "score",
    llm_config: Optional[Dict] = None,
    per_source_cap: int = 0,
    preferred_source: str = "",
    boost: float = 1.5,
    preferred_cap: int = 0,
    other_cap: int = 0
) -> List[Dict]:
    """
    Select final evidences from candidates using the specified strategy.
    
    Args:
        query: The subquery text (for LLM-based selection if needed)
        candidates: List of candidate dictionaries with fields:
            - source_id: str, text: str, score: float or None, rank_within_source: int
        keep_k: Number of evidences to keep
        selector: Selection strategy ("score", "norm_score", "routing_weighted", "rrf", "llm")
        llm_config: Optional dict with keys (api_key, model, base_url) for LLM selector
        per_source_cap: Max candidates per source (0 = no cap, ignored if adaptive cap is set)
        preferred_source: Source name preferred by LLM routing
        boost: Boost factor for preferred source (for routing_weighted)
        preferred_cap: Adaptive cap for preferred source (0 = disabled)
        other_cap: Adaptive cap for non-preferred sources (0 = disabled)
    
    Returns:
        List of selected candidate dictionaries (length <= keep_k)
    """
    if len(candidates) == 0:
        return []

    if preferred_cap > 0 and other_cap > 0 and preferred_source:
        candidates = _apply_per_source_cap(candidates, 0,
                                           preferred_source=preferred_source,
                                           preferred_cap=preferred_cap,
                                           other_cap=other_cap)
        print(f"  📋 Adaptive cap (preferred '{preferred_source}'={preferred_cap}, others={other_cap}): {len(candidates)} candidates")
    elif per_source_cap > 0:
        candidates = _apply_per_source_cap(candidates, per_source_cap)
        print(f"  📋 After per-source cap ({per_source_cap}): {len(candidates)} candidates")
    
    if len(candidates) <= keep_k:
        return candidates
    
    if selector == "score":
        return _select_by_score(candidates, keep_k)
    elif selector == "norm_score":
        return _select_by_norm_score(candidates, keep_k)
    elif selector == "routing_weighted":
        return _select_by_routing_weighted(candidates, keep_k, preferred_source, boost)
    elif selector == "rrf":
        return _select_by_rrf(candidates, keep_k)
    elif selector == "llm":
        if llm_config:
            return _select_by_llm(query, candidates, keep_k, llm_config)
        else:
            print("⚠️  LLM selector requires llm_config, falling back to score-based selection")
            return _select_by_score(candidates, keep_k)
    else:
        print(f"⚠️  Unknown selector '{selector}', falling back to score-based selection")
        return _select_by_score(candidates, keep_k)


def _select_by_score(candidates: List[Dict], keep_k: int) -> List[Dict]:
    """
    Select top-k candidates by score, with fallback to rank ordering.
    """
    scored = [c for c in candidates if c.get("score") is not None]
    unscored = [c for c in candidates if c.get("score") is None]
    
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    unscored_sorted = sorted(unscored, key=lambda x: x.get("rank_within_source", 999))
    
    combined = scored_sorted + unscored_sorted
    return combined[:keep_k]


def _select_by_norm_score(candidates: List[Dict], keep_k: int) -> List[Dict]:
    """
    Z-score normalize within each source, then select top-k by normalized score.
    Solves cross-source score incomparability.
    """
    by_source = {}
    for cand in candidates:
        src = cand.get("source_id", "unknown")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(cand)

    normed = []
    for src, cands in by_source.items():
        scores = [c.get("score", 0.0) for c in cands if c.get("score") is not None]
        if len(scores) >= 2:
            mean = np.mean(scores)
            std = np.std(scores)
            for c in cands:
                raw = c.get("score")
                if raw is not None and std > 1e-8:
                    c = {**c, "norm_score": (raw - mean) / std}
                else:
                    c = {**c, "norm_score": 0.0}
                normed.append(c)
        else:
            for c in cands:
                normed.append({**c, "norm_score": 0.0})

    normed.sort(key=lambda x: x["norm_score"], reverse=True)
    return normed[:keep_k]


def _select_by_routing_weighted(
    candidates: List[Dict], keep_k: int,
    preferred_source: str = "", boost: float = 1.5
) -> List[Dict]:
    """
    Boost candidates from the LLM-preferred source, then z-normalize + select.
    Combines routing intelligence with multi-source coverage.
    """
    by_source = {}
    for cand in candidates:
        src = cand.get("source_id", "unknown")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(cand)

    normed = []
    for src, cands in by_source.items():
        scores = [c.get("score", 0.0) for c in cands if c.get("score") is not None]
        if len(scores) >= 2:
            mean = np.mean(scores)
            std = np.std(scores)
        else:
            mean, std = 0.0, 0.0

        for c in cands:
            raw = c.get("score")
            if raw is not None and std > 1e-8:
                ns = (raw - mean) / std
            else:
                ns = 0.0
            if src.lower() == preferred_source.lower():
                ns *= boost
            normed.append({**c, "weighted_score": ns})

    normed.sort(key=lambda x: x["weighted_score"], reverse=True)
    return normed[:keep_k]


def _select_by_rrf(candidates: List[Dict], keep_k: int, c: int = 60) -> List[Dict]:
    """
    Select using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum over sources of 1 / (rank_in_source + c)
    """
    # Group candidates by source
    by_source = {}
    for cand in candidates:
        source_id = cand["source_id"]
        if source_id not in by_source:
            by_source[source_id] = []
        by_source[source_id].append(cand)
    
    # Calculate RRF scores
    rrf_scores = {}
    for source_id, source_candidates in by_source.items():
        # Sort by rank_within_source to get proper ordering
        sorted_cands = sorted(source_candidates, key=lambda x: x.get("rank_within_source", 999))
        
        for rank, cand in enumerate(sorted_cands):
            # Use a unique key: (source_id, text) to handle duplicates
            key = (cand["source_id"], cand["text"])
            if key not in rrf_scores:
                rrf_scores[key] = {"candidate": cand, "rrf_score": 0.0}
            
            # Add RRF contribution: 1 / (rank + c)
            rrf_scores[key]["rrf_score"] += 1.0 / (rank + c)
    
    # Sort by RRF score descending
    scored_list = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    # Return top keep_k
    return [item["candidate"] for item in scored_list[:keep_k]]


def _select_by_llm(
    query: str,
    candidates: List[Dict],
    keep_k: int,
    llm_config: Dict
) -> List[Dict]:
    """
    Select using LLM relevance judgment combined with scores.
    
    This is a placeholder implementation. For production, you might want to:
    - Batch LLM calls for efficiency
    - Use a cheaper/faster model
    - Cache results
    """
    from utils.llm_call import call_openai_chat
    
    api_key = llm_config.get("api_key")
    model = llm_config.get("model", "gpt-4o-mini")
    base_url = llm_config.get("base_url")
    
    if not api_key:
        print("⚠️  LLM selector requires api_key, falling back to score-based selection")
        return _select_by_score(candidates, keep_k)
    
    # Build prompt for relevance judgment
    # For efficiency, we could batch multiple candidates, but for now do simple version
    scored_candidates = []
    
    for cand in candidates:
        text = cand["text"]
        score = cand.get("score", 0.0)
        
        prompt = f"""Is the following passage relevant to answering this question?

Question: {query}

Passage: {text}

Respond with only "yes" or "no"."""
        
        try:
            response = call_openai_chat(prompt, api_key, model, base_url)
            is_relevant = response.strip().lower().startswith("yes")
            
            # Combine relevance (1.0 if yes, 0.0 if no) with original score
            # If no score, use 0.5 as default
            base_score = score if score is not None else 0.5
            combined_score = base_score + (1.0 if is_relevant else 0.0)
            
            scored_candidates.append({
                **cand,
                "llm_relevant": is_relevant,
                "combined_score": combined_score
            })
        except Exception as e:
            print(f"⚠️  LLM relevance check failed for candidate, using score only: {str(e)}")
            # Fallback: use original score or rank
            base_score = cand.get("score", 0.0) if cand.get("score") is not None else 0.0
            scored_candidates.append({
                **cand,
                "llm_relevant": None,
                "combined_score": base_score
            })
    
    # Sort by combined_score descending
    sorted_candidates = sorted(scored_candidates, key=lambda x: x["combined_score"], reverse=True)
    
    return sorted_candidates[:keep_k]

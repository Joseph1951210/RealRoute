"""
DeepSieve RAG-only pipeline

This script implements a modular RAG pipeline with the following features:
- LLM-based query decomposition (optional)
- Per-subquestion routing to local/global knowledge sources
- Light-weight vector RAG with top-k retrieval
- Structured LLM prompting for reasoning + JSON parsing
- Reflection mechanism to reroute or rephrase failed queries
- Final answer fusion based on reasoning trace
- Comprehensive logging of retrieval, token usage, and accuracy metrics

Usage:
    Configure flags like `decompose`, `use_routing`, `use_reflection` to ablate components.
    Example:
        python script.py --decompose True --use_routing True --use_reflection True

Output:
    Each query's full trace is saved in outputs/{dataset_name}/query_{i}_results.jsonl
    Aggregated metrics saved in overall_results.json

Compatible datasets:
    - hotpot_qa
    - 2wikimultihopqa
    - musique
"""


import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import argparse
import requests
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import tiktoken
import networkx as nx
from collections import defaultdict
import re
from rag.initializer import initialize_rag_system, initialize_multi_source_rag
from pipeline.reasoning_pipeline import plan_subqueries_with_llm, route_query_with_llm, get_fused_final_answer, substitute_variables
from pipeline.subquery_executor import execute_subquery
from utils.data_load import load_queries, load_corpus_and_profiles, load_multi_source_corpus, load_multi_source_queries
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens, evaluate_answer, calculate_overall_metrics

def get_save_dir(decompose: bool, use_routing: bool, use_reflection: bool, dataset: str, rag_type: str, multi_source: bool = False, hard_routing_multi: bool = False, keep_k: int = 5, per_source_cap: int = 0, model: str = "", selector: str = "score", preferred_cap: int = 0, other_cap: int = 0):
    save_dir = "outputs/"
    save_dir += rag_type
    save_dir += "_"
    save_dir += dataset
    if model and model != "gpt-4o-mini":
        save_dir += f"_{model.replace('/', '-')}"
    save_dir += "_"
    if hard_routing_multi:
        save_dir += "_hard_routing_multi"
    elif multi_source:
        save_dir += f"_multi_source_k{keep_k}"
        if preferred_cap > 0 and other_cap > 0:
            save_dir += f"_acap{preferred_cap}_{other_cap}"
        elif per_source_cap > 0:
            save_dir += f"_cap{per_source_cap}"
        if selector and selector != "score":
            save_dir += f"_{selector}"
    elif not use_routing:
        save_dir += "_no_routing"
    if not decompose:
        save_dir += "_no_decompose"
    if not use_reflection:
        save_dir += "_no_reflection"
    return save_dir

def save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics):
    if not overall_metrics:
        print("⚠️  No valid results to summarize (all queries may have failed).")
        # Save empty results for debugging
        overall_json_path = os.path.join(save_dir, "overall_results.json")
        with open(overall_json_path, "w") as f:
            json.dump({"error": "No valid results", "queries": queries_and_truth, "all_query_metrics": all_metrics}, f, indent=2, cls=NumpyEncoder)
        return

    overall_txt_path = os.path.join(save_dir, "overall_results.txt")
    with open(overall_txt_path, "w") as f:
        f.write("📊 Overall Performance Summary:\n")
        f.write(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}\n")
        f.write(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}\n")
        f.write(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s\n")
        f.write(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}\n")
        f.write(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}\n")
        f.write(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}\n")
        f.write(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}\n")

    overall_json_path = os.path.join(save_dir, "overall_results.json")
    with open(overall_json_path, "w") as f:
        json.dump({
            "queries": queries_and_truth,
            "overall_metrics": overall_metrics,
            "all_query_metrics": all_metrics
        }, f, indent=2, cls=NumpyEncoder)

    # Update console output
    print("\n📊 Overall Performance Summary:")
    print(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    print(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}")
    print(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s")
    print(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}")
    print(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}")
    print(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}")
    print(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}")

    print("\n✅ Results saved to:")
    print(f"   - {overall_txt_path}")
    print(f"   - {overall_json_path}")
    for i in range(len(queries_and_truth)):
        print(f"   - {os.path.join(save_dir, f'query_{i+1}_results.txt')}")

def save_single_query_results(save_dir, idx, multi_hop_query, ground_truth, final_answer, final_reason, fusion_token_count, fallback_answer, fusion_prompt, eval_results, eval_results_fallback, performance_metrics, results, fused_answer_texts):
    query_results_path = os.path.join(save_dir, f"query_{idx}_results.jsonl")
    _d = lambda obj: json.dumps(obj, cls=NumpyEncoder)
    with open(query_results_path, "w") as f:
        f.write(_d({
            "type": "query_info",
            "query": multi_hop_query,
            "ground_truth": ground_truth
        }) + "\n")

        f.write(_d({
            "type": "final_answer",
            "final_answer": final_answer,
            "final_reason": final_reason,
            "fusion_prompt_tokens": fusion_token_count,
            "fallback_answer": fallback_answer,
            "fusion_equals_fallback": fallback_answer.strip().lower() == final_answer.strip().lower()
        }) + "\n")

        f.write(_d({
            "type": "evaluation_metrics",
            "fusion": {
                "exact_match": eval_results["exact_match"],
                "f1": eval_results["f1"]
            },
            "fallback": {
                "exact_match": eval_results_fallback["exact_match"],
                "f1": eval_results_fallback["f1"]
            }
        }) + "\n")

        f.write(_d({
            "type": "performance_metrics",
            "total_retrieval_time": performance_metrics["total_retrieval_time"],
            "avg_retrieval_time": performance_metrics["avg_retrieval_time"],
            "total_docs_searched": performance_metrics["total_docs_searched"],
            "avg_similarity": performance_metrics["avg_similarity"],
            "max_similarity": performance_metrics["max_similarity"],
            "token_cost": {
                "total_prompt_tokens": performance_metrics["total_prompt_tokens"],
                "avg_prompt_tokens": performance_metrics["avg_prompt_tokens"],
                "max_prompt_tokens": performance_metrics["max_prompt_tokens"],
                "min_prompt_tokens": performance_metrics["min_prompt_tokens"]
            }
        }) + "\n")

        for metrics in performance_metrics["subquery_metrics"]:
            f.write(_d({
                "type": "subquery_metric",
                "subquery_id": metrics["subquery_id"],
                "retrieval_time": metrics["retrieval_time"],
                "docs_searched": metrics["docs_searched"],
                "avg_similarity": metrics["avg_similarity"],
                "max_similarity": metrics["max_similarity"]
            }) + "\n")

        for r in results:
            f.write(_d({
                "type": "execution_result",
                "subquery_id": r["subquery_id"],
                "original_query": r["original_query"],
                "actual_query": r["actual_query"],
                "variables_used": r.get("variables_used", None),
                "routing": r["routing"],
                "answer": r["answer"],
                "reason": r["reason"],
                "docs": [
                    {
                        "text": doc,
                        "score": r["doc_scores"][i]
                    }
                    for i, doc in enumerate(r["docs"])
                ]
            }) + "\n")

        for step in fused_answer_texts:
            f.write(_d({
                "type": "fused_answer_step",
                "text": step
            }) + "\n")        
    # Save fusion prompt
    fusion_prompt_path = os.path.join(save_dir, f"query_{idx}_fusion_prompt.txt")
    with open(fusion_prompt_path, "w") as f_prompt:
        f_prompt.write(fusion_prompt)
    return performance_metrics

def process_subqueries(performance_metrics, query_plan, variable_values, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir, idx, multi_hop_query, ground_truth, results, fused_answer_texts, multi_source=False, top_k_per_source=5, keep_k=5, selector="score", per_source_cap=0, rag_sources=None, source_profiles=None, hard_routing_multi=False, preferred_cap=0, other_cap=0):
    for subquery_info in query_plan["subqueries"]:
        subquery_result = execute_subquery(
            subquery_info,
            variable_values,
            local_rag,
            global_rag,
            merged_rag,
            use_routing,
            use_reflection,
            max_reflexion_times,
            local_profile,
            global_profile,
            openai_api_key,
            openai_model,
            openai_base_url,
            multi_source,
            top_k_per_source,
            keep_k,
            selector,
            per_source_cap=per_source_cap,
            rag_sources=rag_sources,
            source_profiles=source_profiles,
            hard_routing_multi=hard_routing_multi,
            preferred_cap=preferred_cap,
            other_cap=other_cap
        )
        results.append(subquery_result)
        fused_answer_texts.append(f"{subquery_result['subquery_id']}: {subquery_result['actual_query']} → {subquery_result['answer']} (reason: {subquery_result['reason']})")
        performance_metrics["total_retrieval_time"] += subquery_result["retrieval_time"]
        performance_metrics["total_docs_searched"] += subquery_result["docs_searched"]
        performance_metrics["avg_similarity_scores"].append(subquery_result["avg_similarity"])
        performance_metrics["max_similarity_scores"].append(subquery_result["max_similarity"])
        # Accumulate token counts from each subquery
        performance_metrics["total_prompt_tokens"] += subquery_result.get("prompt_token_count", 0)
        performance_metrics["prompt_token_counts"].append(subquery_result.get("prompt_token_count", 0))
    # Compute summary metrics
    performance_metrics["avg_retrieval_time"] = performance_metrics["total_retrieval_time"] / len(query_plan["subqueries"])
    performance_metrics["avg_similarity"] = np.mean(performance_metrics["avg_similarity_scores"])
    performance_metrics["max_similarity"] = np.max(performance_metrics["max_similarity_scores"])
    
    # Compute token statistics
    performance_metrics["avg_prompt_tokens"] = performance_metrics["total_prompt_tokens"] / len(query_plan["subqueries"])
    performance_metrics["max_prompt_tokens"] = max(performance_metrics["prompt_token_counts"], default=0)
    performance_metrics["min_prompt_tokens"] = min(performance_metrics["prompt_token_counts"], default=0)

    # Get fallback (last hop) answer (for comparison)
    fallback_answer = results[-1]["answer"] if results else ""
    performance_metrics["evaluation_metrics"]["fallback_answer"] = fallback_answer


    # Get final answer (fusing all reasoning steps)
    final_answer, final_reason, fusion_token_count, fusion_prompt = get_fused_final_answer(
        multi_hop_query, results,
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url
    )
    performance_metrics["evaluation_metrics"]["final_answer"] = final_answer
    performance_metrics["evaluation_metrics"]["final_reason"] = final_reason
    performance_metrics["fusion_prompt_tokens"] = fusion_token_count

    performance_metrics["total_prompt_tokens"] += fusion_token_count
    performance_metrics["prompt_token_counts"].append(fusion_token_count)

    
    # Compute evaluation metrics
    eval_results = evaluate_answer(final_answer, ground_truth)
    eval_results_fallback = evaluate_answer(fallback_answer, ground_truth)
    performance_metrics["evaluation_metrics"].update(eval_results)
    performance_metrics["evaluation_metrics_fallback"].update(eval_results_fallback)
    
    # Save current query results
    performance_metrics = save_single_query_results(save_dir, idx, multi_hop_query, ground_truth, final_answer, final_reason, fusion_token_count, fallback_answer, fusion_prompt, eval_results, eval_results_fallback, performance_metrics, results, fused_answer_texts)

    return performance_metrics

def single_query_execution(decompose, all_metrics, queries_and_truth, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir, multi_source=False, top_k_per_source=5, keep_k=5, selector="score", per_source_cap=0, rag_sources=None, source_profiles=None, hard_routing_multi=False, preferred_cap=0, other_cap=0):
    # Process each query
    for idx, query_info in enumerate(queries_and_truth, 1):
        multi_hop_query = query_info["query"]
        ground_truth = query_info["ground_truth"]
        
        print(f"\n📝 Processing query {idx}/{len(queries_and_truth)}:")
        print(f"Query: {multi_hop_query}")
        print(f"Ground Truth: {ground_truth}")
        
        # Initialize variable_values dict
        variable_values = {}

        query_plan = plan_subqueries_with_llm(decompose, multi_hop_query, openai_api_key, openai_model, openai_base_url)
        if not query_plan or not query_plan["subqueries"]:
            print("❌ Subquery planning failed, skipping current query.")
            continue

        results, fused_answer_texts = [], []
        performance_metrics = {
            "total_retrieval_time": 0,
            "total_docs_searched": 0,
            "avg_similarity_scores": [],
            "max_similarity_scores": [],
            "subquery_metrics": [],
            "total_prompt_tokens": 0,
            "prompt_token_counts": [],
            "evaluation_metrics": {
                "exact_match": 0,
                "f1": 0,
                "final_answer": "",
                "ground_truth": ground_truth
            },
            "evaluation_metrics_fallback": {
                "exact_match": 0,
                "f1": 0,
                "ground_truth": ground_truth
            }
        }

        # Process subqueries in order, handle dependencies
        performance_metrics = process_subqueries(
            performance_metrics, 
            query_plan, 
            variable_values, 
            local_rag, 
            global_rag, 
            merged_rag, 
            use_routing, 
            use_reflection, 
            max_reflexion_times, 
            local_profile, 
            global_profile, 
            openai_api_key, 
            openai_model, 
            openai_base_url, 
            save_dir, 
            idx, 
            multi_hop_query, 
            ground_truth, 
            results, 
            fused_answer_texts,
            multi_source,
            top_k_per_source,
            keep_k,
            selector,
            per_source_cap=per_source_cap,
            rag_sources=rag_sources,
            source_profiles=source_profiles,
            hard_routing_multi=hard_routing_multi,
            preferred_cap=preferred_cap,
            other_cap=other_cap
        )
        all_metrics.append(performance_metrics)
    
    return all_metrics



# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSieve RAG-only pipeline")
    parser.add_argument("--decompose", action="store_true", help="Enable query decomposition")
    parser.add_argument("--use_routing", action="store_true", help="Enable routing to local/global RAG")
    parser.add_argument("--use_reflection", action="store_true", help="Enable reflection on failed queries")
    parser.add_argument("--max_reflexion_times", type=int, default=2, help="Max retry times for reflection")
    
    parser.add_argument("--dataset", type=str, default="hotpot_qa", help="Dataset name")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to evaluate")

    parser.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--openai_base_url", type=str, default=os.environ.get("OPENAI_API_BASE"))

    parser.add_argument("--rag_type", type=str, choices=["naive", "graph"], default=os.environ.get("RAG_TYPE", "naive"))
    
    # Multi-source evidence-level retrieval arguments
    parser.add_argument("--multi_source", action="store_true", help="Enable multi-source evidence-level retrieval (overrides hard routing)")
    parser.add_argument("--top_k_per_source", type=int, default=5, help="Number of candidates to retrieve per source")
    parser.add_argument("--keep_k", type=int, default=5, help="Number of final evidences to keep after selection")
    parser.add_argument("--selector", type=str, choices=["score", "norm_score", "routing_weighted", "rrf", "llm"], default="score", help="Evidence selection strategy")
    parser.add_argument("--per_source_cap", type=int, default=0, help="Max evidences per source (0 = no cap)")
    parser.add_argument("--preferred_cap", type=int, default=0, help="Adaptive cap: max evidences from LLM-preferred source (0 = disabled)")
    parser.add_argument("--other_cap", type=int, default=0, help="Adaptive cap: max evidences from non-preferred sources (0 = disabled)")
    parser.add_argument("--hard_routing_multi", action="store_true", help="Enable N-source hard routing (DeepSieve-style, route to ONE of N sources)")
    
    return parser.parse_args()


def main(args):
    """
    Main function
    Args:
        decompose: Whether to decompose the query
        use_routing: Whether to use routing
        use_reflection: Whether to use reflection mechanism
        max_reflexion_times: Max reflection times
        dataset: Dataset name
        sample_size: Sample size
        openai_model: OpenAI model name
        openai_api_key: OpenAI API key
        openai_base_url: OpenAI API base URL
        rag_type: RAG type, can be "naive" or "graph"
    """
    # Load data
    save_dir = get_save_dir(args.decompose, args.use_routing, args.use_reflection, args.dataset, args.rag_type, args.multi_source, args.hard_routing_multi, args.keep_k, args.per_source_cap, args.openai_model, args.selector, args.preferred_cap, args.other_cap)
    os.makedirs(save_dir, exist_ok=True)

    openai_model = args.openai_model
    openai_api_key = args.openai_api_key
    openai_base_url = args.openai_base_url
    if not openai_api_key:
        raise ValueError("❌ Please set your OPENAI_API_KEY environment variable.")

    # Detect if this is a multi-source dataset (has _profiles.json instead of _corpus_profiles.json)
    is_multi_source_dataset = os.path.exists(f"data/rag/{args.dataset}_profiles.json")

    # --- Multi-source cross-domain path ---
    if (args.multi_source or args.hard_routing_multi) and is_multi_source_dataset:
        mode_label = "hard-routing" if args.hard_routing_multi else "multi-source"
        print(f"🌐 Cross-domain {mode_label} mode detected for dataset '{args.dataset}'")

        # Load multi-source QA (with source labels)
        queries_and_truth = load_multi_source_queries(args.dataset, args.sample_size)
        print(f"✅ Loaded {len(queries_and_truth)} queries (with source labels)")

        # Load multi-source corpus and profiles
        sources_docs, source_profiles = load_multi_source_corpus(args.dataset)

        # Initialize RAG for each source
        rag_sources = initialize_multi_source_rag(args.rag_type, sources_docs)

        if args.hard_routing_multi:
            print(f"🔀 Hard routing mode: LLM will route each query to ONE of {len(rag_sources)} sources")
        else:
            print(f"🔍 Multi-source evidence-level retrieval enabled")
        print(f"   Sources: {list(rag_sources.keys())}")
        if not args.hard_routing_multi:
            print(f"   selector={args.selector}, top_k_per_source={args.top_k_per_source}, keep_k={args.keep_k}")

        # In cross-domain multi-source mode, we don't use local/global RAG or profiles
        local_rag, global_rag, merged_rag = None, None, None
        local_profile, global_profile = "", ""
        use_routing_effective = False

        all_metrics = []
        all_metrics = single_query_execution(
            args.decompose,
            all_metrics,
            queries_and_truth,
            local_rag,
            global_rag,
            merged_rag,
            use_routing_effective,
            args.use_reflection,
            args.max_reflexion_times,
            local_profile,
            global_profile,
            openai_api_key,
            openai_model,
            openai_base_url,
            save_dir,
            args.multi_source,
            args.top_k_per_source,
            args.keep_k,
            args.selector,
            per_source_cap=args.per_source_cap,
            rag_sources=rag_sources,
            source_profiles=source_profiles,
            hard_routing_multi=args.hard_routing_multi,
            preferred_cap=args.preferred_cap,
            other_cap=args.other_cap
        )

    # --- Original path (local/global split) ---
    else:
        queries_and_truth = load_queries(args.dataset, args.sample_size)

        # Prepare knowledge base documents
        local_docs, global_docs, local_profile, global_profile = load_corpus_and_profiles(args.dataset)
        print(f"✅ Loaded {len(local_docs)} documents into local_docs.")
        print(f"✅ Loaded {len(global_docs)} documents into global_docs.")
        print(f"✅ Loaded local_profile and global_profile.")

        # Handle multi_source mode interaction with use_routing
        use_routing_effective = args.use_routing
        if args.multi_source and args.use_routing:
            print("⚠️  Multi-source mode enabled: hard routing will be bypassed, using evidence-level selection instead")
            use_routing_effective = False

        # Initialize RAG system
        local_rag, global_rag, merged_rag = initialize_rag_system(
            args.rag_type,
            use_routing_effective or args.multi_source,
            local_docs,
            global_docs
        )

        if args.multi_source:
            print(f"🔍 Multi-source evidence-level retrieval enabled (selector={args.selector}, top_k_per_source={args.top_k_per_source}, keep_k={args.keep_k})")

        all_metrics = []
        all_metrics = single_query_execution(
            args.decompose,
            all_metrics,
            queries_and_truth,
            local_rag,
            global_rag,
            merged_rag,
            use_routing_effective,
            args.use_reflection,
            args.max_reflexion_times,
            local_profile,
            global_profile,
            openai_api_key,
            openai_model,
            openai_base_url,
            save_dir,
            args.multi_source,
            args.top_k_per_source,
            args.keep_k,
            args.selector,
            per_source_cap=args.per_source_cap,
            preferred_cap=args.preferred_cap,
            other_cap=args.other_cap
        )
    # Compute and save overall performance metrics
    overall_metrics = calculate_overall_metrics(all_metrics)
    
    # Save overall results
    save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics)

if __name__ == "__main__":
    args = parse_args()
    main(args)
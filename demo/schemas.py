from typing import Dict, Set


RECORD_REQUIRED_KEYS: Dict[str, Set[str]] = {
    "query_info": {"type", "query"},
    "final_answer": {"type", "final_answer", "final_reason"},
    "evaluation_metrics": {"type", "fusion", "fallback"},
    "performance_metrics": {"type", "total_retrieval_time", "total_docs_searched"},
    "subquery_metric": {"type", "subquery_id"},
    "execution_result": {
        "type",
        "subquery_id",
        "original_query",
        "actual_query",
        "routing",
        "answer",
        "reason",
        "docs",
    },
    "fused_answer_step": {"type", "text"},
}


RECORD_OPTIONAL_KEYS: Dict[str, Set[str]] = {
    "query_info": {"ground_truth"},
    "final_answer": {"fusion_prompt_tokens", "fallback_answer", "fusion_equals_fallback"},
    "evaluation_metrics": set(),
    "performance_metrics": {"avg_retrieval_time", "avg_similarity", "max_similarity", "token_cost"},
    "subquery_metric": {"retrieval_time", "docs_searched", "avg_similarity", "max_similarity"},
    # `success` is not present in the current JSONL schema written by runner/main_rag_only.py.
    "execution_result": {"variables_used", "doc_scores", "metrics", "prompt_token_count", "success"},
    "fused_answer_step": set(),
}


RECORD_TYPE_ORDER = [
    "query_info",
    "final_answer",
    "evaluation_metrics",
    "performance_metrics",
    "subquery_metric",
    "execution_result",
    "fused_answer_step",
]


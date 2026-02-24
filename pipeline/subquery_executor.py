"""
pipeline/subquery_executor.py

This module implements the subquery executor for DeepSieve.
"""

import time
import json
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens
from pipeline.reasoning_pipeline import route_query_with_llm, route_query_multi_source, substitute_variables
from pipeline.multi_source_retrieval import retrieve_multi_source
from pipeline.evidence_selector import select_evidence



def execute_subquery(
    subquery_info: dict, 
    variable_values: dict, 
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
    multi_source=False,
    top_k_per_source=5,
    keep_k=5,
    selector="score",
    per_source_cap=0,
    rag_sources=None,
    source_profiles=None,
    hard_routing_multi=False,
    preferred_cap=0,
    other_cap=0
) -> dict:
    """
    Execute a subquery and return the results.
    """

    # Ensure all return variables are initialized
    answer = ""
    reason = ""
    success = 0
    retrieved = {"docs": [], "doc_scores": []}
    subquery_metrics = {
        "subquery_id": subquery_info.get("id", ""),
        "retrieval_time": 0,
        "docs_searched": 0,
        "avg_similarity": 0,
        "max_similarity": 0
    }
    token_count = 0
    current_variables = {}
    results = []
    fused_answer_texts = []

    subquery_start_time = time.time()
    subquery_id = subquery_info["id"]
    original_query = subquery_info["query"]
    
    # Check and wait for all dependencies to complete
    if subquery_info["depends_on"]:
        print(f"\n⏳ Processing dependencies for query {subquery_id}: {subquery_info['depends_on']}")
        
    # Replace variables in the query
    for var in subquery_info.get("variables", []):
        # Some LLM outputs may produce an unexpected structure for `variables`,
        # e.g. a list of strings instead of dicts. Be defensive here to avoid crashes.
        if isinstance(var, str):
            print(f"⚠️ Unexpected variable entry (string) in query {subquery_id}, skipping: {var}")
            continue
        if not isinstance(var, dict):
            print(f"⚠️ Unexpected variable entry type in query {subquery_id}: {type(var)}, skipping")
            continue

        var_name = var.get("name")
        source_query = var.get("source_query")
        if not var_name or not source_query:
            print(f"⚠️ Variable entry missing name/source_query in query {subquery_id}, skipping: {var}")
            continue

        if source_query not in variable_values:
            print(f"❌ Error: Query {subquery_id} depends on an incomplete query {source_query}")
            continue
        current_variables[var_name] = variable_values[source_query]
    
    # Actual query after variable substitution
    actual_query = substitute_variables(original_query, current_variables)
    print(f"\n🔍 Processing query {subquery_id}: {actual_query}")
    print(f"Original query: {original_query}")
    if current_variables:
        print(f"Variable substitution: {current_variables}")

    # Loop for reflection
    fail_history = ""
    left_reflexion_times = max_reflexion_times
    while True and left_reflexion_times > 0:
        left_reflexion_times -= 1
        success = 0  # 初始化success为0
        
        if hard_routing_multi and rag_sources is not None and source_profiles is not None:
            # N-source hard routing: LLM picks ONE source from N, then retrieves from that source only
            route = route_query_multi_source(
                actual_query, source_profiles,
                api_key=openai_api_key, model=openai_model, base_url=openai_base_url,
                fail_history=fail_history
            )
            print(f"🔀 Hard routing → source '{route}'")

            selected_rag = rag_sources.get(route)
            if selected_rag is None:
                print(f"⚠️  Routed source '{route}' not found, falling back to first source")
                route = list(rag_sources.keys())[0]
                selected_rag = rag_sources[route]

            try:
                retrieved = selected_rag.rag_qa(actual_query, k=5)
            except Exception as e:
                print(f"⚠️  Error in hard routing retrieval: {str(e)}")
                retrieved = {"docs": [], "doc_scores": [], "metrics": {"retrieval_time": 0, "avg_similarity": 0, "max_similarity": 0, "total_docs_searched": 0}}

        elif multi_source:
            # Multi-source evidence-level retrieval mode
            route = "multi_source"
            print(f"🔍 Multi-source retrieval mode (selector={selector})")
            
            # Build sources list: prefer rag_sources dict (cross-domain), fallback to local/global
            sources = []
            if rag_sources is not None:
                for name, rag in rag_sources.items():
                    if rag is not None:
                        sources.append((name, rag))
            else:
                if local_rag is not None:
                    sources.append(("local", local_rag))
                if global_rag is not None:
                    sources.append(("global", global_rag))
            
            if len(sources) == 0:
                print("⚠️  No sources available for multi-source retrieval")
                retrieved = {"docs": [], "doc_scores": [], "metrics": {"retrieval_time": 0, "avg_similarity": 0, "max_similarity": 0, "total_docs_searched": 0}}
            else:
                try:
                    retrieval_start = time.time()
                    
                    # Get preferred source via LLM routing when needed
                    need_routing = selector == "routing_weighted" or (preferred_cap > 0 and other_cap > 0)
                    preferred_source = ""
                    if need_routing and source_profiles:
                        preferred_source = route_query_multi_source(
                            actual_query, source_profiles,
                            api_key=openai_api_key, model=openai_model, base_url=openai_base_url,
                            fail_history=fail_history
                        )
                        mode_label = "adaptive cap" if preferred_cap > 0 else "boost signal"
                        print(f"  🎯 Routing preference: '{preferred_source}' (used as {mode_label})")
                    elif need_routing and local_profile and global_profile:
                        preferred_source = route_query_with_llm(
                            actual_query, local_profile, global_profile,
                            api_key=openai_api_key, model=openai_model, base_url=openai_base_url,
                            fail_history=fail_history
                        )
                        mode_label = "adaptive cap" if preferred_cap > 0 else "boost signal"
                        print(f"  🎯 Routing preference: '{preferred_source}' (used as {mode_label})")
                    
                    candidates = retrieve_multi_source(actual_query, sources, top_k_per_source)
                    
                    selected = select_evidence(
                        actual_query,
                        candidates,
                        keep_k,
                        selector,
                        llm_config={"api_key": openai_api_key, "model": openai_model, "base_url": openai_base_url} if selector == "llm" else None,
                        per_source_cap=per_source_cap,
                        preferred_source=preferred_source,
                        preferred_cap=preferred_cap,
                        other_cap=other_cap
                    )
                    
                    retrieval_time = time.time() - retrieval_start
                    
                    # Log source distribution of selected evidence
                    src_dist = {}
                    for e in selected:
                        s = e.get("source_id", "?")
                        src_dist[s] = src_dist.get(s, 0) + 1
                    print(f"  ✅ Selected {len(selected)} evidences from {len(candidates)} candidates | distribution: {src_dist}")
                    
                    retrieved_docs = [e["text"] for e in selected]
                    retrieved_scores = [e.get("score", 0.0) if e.get("score") is not None else 0.0 for e in selected]
                    
                    total_docs_searched = 0
                    all_scores = [s for s in retrieved_scores if s > 0]
                    avg_sim = sum(all_scores) / len(all_scores) if all_scores else 0.0
                    max_sim = max(all_scores) if all_scores else 0.0
                    
                    for source_id, rag in sources:
                        if hasattr(rag, 'docs'):
                            total_docs_searched += len(rag.docs)
                    
                    retrieved = {
                        "docs": retrieved_docs,
                        "doc_scores": retrieved_scores,
                        "metrics": {
                            "retrieval_time": retrieval_time,
                            "avg_similarity": avg_sim,
                            "max_similarity": max_sim,
                            "total_docs_searched": total_docs_searched
                        }
                    }
                except Exception as e:
                    print(f"⚠️  Multi-source retrieval failed: {str(e)}")
                    retrieved = {"docs": [], "doc_scores": [], "metrics": {"retrieval_time": 0, "avg_similarity": 0, "max_similarity": 0, "total_docs_searched": 0}}
        elif use_routing:
            # Original hard routing mode
            route = route_query_with_llm(actual_query, local_profile, global_profile,
                                    api_key=openai_api_key, model=openai_model, base_url=openai_base_url, fail_history=fail_history)
            rag = local_rag if route == "local" else global_rag
            print(f"Routing to {route.upper()} DB")
            
            try:
                retrieved = rag.rag_qa(actual_query, k=5)
            except Exception as e:
                print(f"⚠️  Error in routing retrieval: {str(e)}")
                retrieved = {"docs": [], "doc_scores": [], "metrics": {"retrieval_time": 0, "avg_similarity": 0, "max_similarity": 0, "total_docs_searched": 0}}
        else:
            # Original merged mode
            rag = merged_rag
            route = "merged"
            print(f"Using merged DB")
            
            try:
                retrieved = rag.rag_qa(actual_query, k=5)
            except Exception as e:
                print(f"⚠️  Error in merged retrieval: {str(e)}")
                retrieved = {"docs": [], "doc_scores": [], "metrics": {"retrieval_time": 0, "avg_similarity": 0, "max_similarity": 0, "total_docs_searched": 0}}

        try:
            
            # Collect performance metrics
            metrics = retrieved["metrics"]
            
            subquery_metrics = {
                "subquery_id": subquery_id,
                "retrieval_time": metrics["retrieval_time"],
                "docs_searched": metrics["total_docs_searched"],
                "avg_similarity": metrics["avg_similarity"],
                "max_similarity": metrics["max_similarity"]
            }
            
            prompt = f"""Answer the following question based on the provided documents.

Please respond strictly in JSON format with the following fields:
- answer: the direct, concise answer (just the value/entity/fact, no explanation). Leave it empty ("") if the answer is not found.
- reason: a brief explanation of how you arrived at this answer.
- success: 1 if the answer is confidently found from the documents, 0 otherwise.

Format:
{{
"answer": "...",
"reason": "...",
"success": 1
}}

If the answer is not mentioned or cannot be inferred from the documents, return:
{{
"answer": "",
"reason": "no relevant information found",
"success": 0
}}

Question: {actual_query}

Documents:
"""
            for d in retrieved["docs"]:
                prompt += f"- {d}\n"
            prompt += "\nOnly output valid JSON. Do not add any explanation or markdown code block markers."

            token_count = count_tokens(prompt, openai_model)
            print(f"🧮 Prompt Token Count: {token_count}")
            
            response = call_openai_chat(prompt, openai_api_key, openai_model, openai_base_url)
            try:
                # Clean the response of markdown code block markers
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                parsed_response = json.loads(cleaned_response)
                answer = parsed_response["answer"].strip()
                reason = parsed_response["reason"].strip()
                success = int(parsed_response["success"])
                
                # Store answer for subsequent queries
                if success == 1:
                    variable_values[subquery_id] = answer
                    print(f"Extracted answer: {answer}")
                    print(f"Reasoning: {reason}")
                    print(f"Success: {success}")
                else:
                    variable_values[subquery_id] = ""
                    if hard_routing_multi:
                        fail_history += f"Fail History: Last routing failed because {reason}. Last routing chose source \"{route}\". Please try a different source, don't choose \"{route}\" again."
                    elif multi_source:
                        # For multi-source, we can't easily change routing, so just note the failure
                        fail_history += f"Fail History: Last retrieval failed because {reason}. Using multi-source mode."
                    else:
                        fail_history += f"Fail History: Last routing failed because {reason}. Last routing result is {route}. So please try another routing choice, don't choose {route} again."
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Failed to parse answer: {str(e)}")
                print(f"Raw response: {response}")
                answer = f"Error: {str(e)}"
                reason = ""
                success = 0
            
            # Store the results of this subquery
            results.append({
                "subquery_id": subquery_id,
                "original_query": original_query,
                "actual_query": actual_query,
                "routing": route,
                "answer": answer,
                "reason": reason,
                "docs": retrieved["docs"],
                "doc_scores": retrieved["doc_scores"],
                "variables_used": current_variables,
                "metrics": subquery_metrics,
                "prompt_token_count": token_count
            })
            fused_answer_texts.append(f"{subquery_id}: {actual_query} → {answer} (reason: {reason})")
            
        except Exception as e:
            print(f"⚠️ Error occurred while processing query: {str(e)}")
            answer = f"Error: {str(e)}"
            reason = ""
            success = 0
            retrieved = {"docs": [], "doc_scores": []}
            subquery_metrics = {
                "subquery_id": subquery_id,
                "retrieval_time": 0,
                "docs_searched": 0,
                "avg_similarity": 0,
                "max_similarity": 0
            }
            token_count = 0

        if success == 1 or (not multi_source and use_routing == False) or use_reflection == False or left_reflexion_times <= 0:
            break   

    return {
        "answer": answer,
        "reason": reason,
        "success": success,
        "docs": retrieved["docs"],
        "doc_scores": retrieved["doc_scores"],
        "variables_used": current_variables,
        "metrics": subquery_metrics,
        "prompt_token_count": token_count,
        "subquery_id": subquery_id,
        "original_query": original_query,
        "actual_query": actual_query,
        "routing": route,
        # for main aggregation:
        "retrieval_time": subquery_metrics["retrieval_time"],
        "docs_searched": subquery_metrics["docs_searched"],
        "avg_similarity": subquery_metrics["avg_similarity"],
        "max_similarity": subquery_metrics["max_similarity"]
    }

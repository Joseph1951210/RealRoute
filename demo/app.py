import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.run_pipeline import (  # noqa: E402
    CustomSourceSpec,
    PRESETS,
    DemoConfig,
    load_last_run_meta,
    load_run_sidecar,
    parse_uploaded_queries,
    parse_uploaded_source_corpus,
    run_demo,
)
from demo.trace_parser import (  # noqa: E402
    list_trace_files,
    parse_trace_file,
    summarize_doc_source_distribution,
    summarize_routing_distribution,
)


st.set_page_config(page_title="Multi-Source RAG Demo", layout="wide")


def _render_json_downloads(trace_path: str, parsed_export: Dict[str, Any], key_prefix: str) -> None:
    try:
        raw = Path(trace_path).read_bytes()
        st.download_button(
            "Download trace (JSONL)",
            data=raw,
            file_name=Path(trace_path).name,
            mime="application/json",
            key=f"{key_prefix}_dl_jsonl",
        )
    except FileNotFoundError:
        st.warning("Trace file not found for download.")
    st.download_button(
        "Download trace (JSON)",
        data=json.dumps(parsed_export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(trace_path).stem}.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
    )


def _render_doc_list(docs: List[Dict[str, Any]], key_prefix: str) -> None:
    if not docs:
        st.caption("No docs")
        return
    rows = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, dict):
            text = str(d.get("text", ""))
            rows.append(
                {
                    "rank": i,
                    "score": d.get("score"),
                    "source_id": d.get("source_id"),
                    "text_preview": (text[:220] + "...") if len(text) > 220 else text,
                }
            )
        else:
            rows.append({"rank": i, "score": None, "source_id": None, "text_preview": str(d)[:220]})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Show full docs text"):
        for i, d in enumerate(docs, 1):
            if isinstance(d, dict):
                st.markdown(f"**Doc {i}** (score={d.get('score')}, source={d.get('source_id', 'N/A')})")
                st.code(str(d.get("text", "")))
            else:
                st.markdown(f"**Doc {i}**")
                st.code(str(d))


def _render_trace(trace_path: str, query_meta: Optional[Dict[str, Any]], section_key: str) -> None:
    parsed = parse_trace_file(trace_path)
    if parsed.warnings:
        with st.expander("Schema / parse warnings", expanded=False):
            for w in parsed.warnings:
                st.warning(w)

    qinfo = parsed.query_info
    final_answer = parsed.final_answer
    eval_rec = parsed.evaluation_metrics
    perf = parsed.performance_metrics

    st.subheader("Trace Overview")
    st.write(f"**Trace file:** `{trace_path}`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**query_info**")
        st.write("Query:", qinfo.get("query", ""))
        st.write("Ground truth:", qinfo.get("ground_truth", ""))
    with c2:
        st.markdown("**final_answer**")
        st.write("Final answer:", final_answer.get("final_answer", ""))
        st.write("Fallback answer:", final_answer.get("fallback_answer", ""))
        st.write("Fusion prompt tokens:", final_answer.get("fusion_prompt_tokens", ""))

    st.markdown("**Final reason**")
    st.write(final_answer.get("final_reason", ""))

    has_gt = bool(query_meta and query_meta.get("has_ground_truth"))
    if has_gt and eval_rec:
        st.markdown("**evaluation_metrics**")
        e1, e2 = st.columns(2)
        with e1:
            st.write("Fusion EM:", (eval_rec.get("fusion") or {}).get("exact_match"))
            st.write("Fusion F1:", (eval_rec.get("fusion") or {}).get("f1"))
        with e2:
            st.write("Fallback EM:", (eval_rec.get("fallback") or {}).get("exact_match"))
            st.write("Fallback F1:", (eval_rec.get("fallback") or {}).get("f1"))
    else:
        st.info("Ground truth missing for this query; EM/F1 is hidden in the UI.")

    if perf:
        st.markdown("**performance_metrics**")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Total retrieval time (s)", f"{perf.get('total_retrieval_time', 0):.3f}" if isinstance(perf.get("total_retrieval_time"), (int, float)) else perf.get("total_retrieval_time"))
            st.metric("Avg retrieval time (s)", f"{perf.get('avg_retrieval_time', 0):.3f}" if isinstance(perf.get("avg_retrieval_time"), (int, float)) else perf.get("avg_retrieval_time"))
        with p2:
            st.metric("Docs searched", perf.get("total_docs_searched"))
            st.metric("Avg similarity", f"{perf.get('avg_similarity', 0):.4f}" if isinstance(perf.get("avg_similarity"), (int, float)) else perf.get("avg_similarity"))
        with p3:
            st.metric("Max similarity", f"{perf.get('max_similarity', 0):.4f}" if isinstance(perf.get("max_similarity"), (int, float)) else perf.get("max_similarity"))
            st.write("Token cost:", perf.get("token_cost", {}))

    st.subheader("execution_result (subqueries)")
    for idx, rec in enumerate(parsed.execution_results, 1):
        label = f"{rec.get('subquery_id', f'subquery_{idx}')}: {rec.get('actual_query', '')[:80]}"
        with st.expander(label, expanded=(idx == 1)):
            left, right = st.columns([3, 2])
            with left:
                st.write("original_query:", rec.get("original_query", ""))
                st.write("actual_query:", rec.get("actual_query", ""))
                st.write("variables_used:", rec.get("variables_used", {}))
                st.write("routing:", rec.get("routing", ""))
                st.write("success:", rec.get("success", "N/A (not stored in current JSONL schema)"))
            with right:
                st.write("answer:", rec.get("answer", ""))
                st.write("reason:", rec.get("reason", ""))
            st.markdown("**docs (text + score)**")
            _render_doc_list(rec.get("docs", []) or [], key_prefix=f"{section_key}_subq_{idx}")

    fused_steps = parsed.fused_steps
    if fused_steps:
        with st.expander("fused_answer_step (trace chain)", expanded=False):
            for i, step in enumerate(fused_steps, 1):
                st.write(f"{i}. {step.get('text', '')}")

    _render_json_downloads(trace_path, parsed.to_export_json(), key_prefix=section_key)


def _find_query_meta(run_block: Dict[str, Any], selected_index_1based: int) -> Optional[Dict[str, Any]]:
    items = (run_block or {}).get("query_items") or []
    idx = selected_index_1based - 1
    if 0 <= idx < len(items):
        return items[idx]
    return None


def _load_trace_path_for_index(output_dir: str, selected_index_1based: int) -> Optional[str]:
    files = list_trace_files(output_dir)
    idx = selected_index_1based - 1
    if 0 <= idx < len(files):
        return files[idx]
    return None


def _compare_panel(primary_block: Dict[str, Any], baseline_block: Dict[str, Any], selected_index_1based: int) -> None:
    p_path = _load_trace_path_for_index(primary_block["output_dir"], selected_index_1based)
    b_path = _load_trace_path_for_index(baseline_block["output_dir"], selected_index_1based)
    if not p_path or not b_path:
        st.warning("Compare traces not found for selected query.")
        return

    p_trace = parse_trace_file(p_path)
    b_trace = parse_trace_file(b_path)

    # Explicit baseline/ours assignment for metrics
    left_block, right_block = baseline_block, primary_block
    left_trace, right_trace = b_trace, p_trace

    st.subheader("Compare: Final Answer / Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Baseline ({left_block.get('mode')})**")
        st.write("Final answer:", left_trace.final_answer.get("final_answer", ""))
        st.write("Fallback answer:", left_trace.final_answer.get("fallback_answer", ""))
        st.write("Fusion prompt tokens:", left_trace.final_answer.get("fusion_prompt_tokens", ""))
    with c2:
        st.markdown(f"**Ours ({right_block.get('mode')})**")
        st.write("Final answer:", right_trace.final_answer.get("final_answer", ""))
        st.write("Fallback answer:", right_trace.final_answer.get("fallback_answer", ""))
        st.write("Fusion prompt tokens:", right_trace.final_answer.get("fusion_prompt_tokens", ""))

    qmeta = _find_query_meta(primary_block, selected_index_1based)
    has_gt = bool(qmeta and qmeta.get("has_ground_truth"))
    if has_gt:
        st.markdown("**EM / F1**")
        m1, m2, m3, m4 = st.columns(4)
        left_eval = left_trace.evaluation_metrics or {}
        right_eval = right_trace.evaluation_metrics or {}
        m1.metric("Baseline EM", (left_eval.get("fusion") or {}).get("exact_match"))
        m2.metric("Ours EM", (right_eval.get("fusion") or {}).get("exact_match"))
        m3.metric("Baseline F1", (left_eval.get("fusion") or {}).get("f1"))
        m4.metric("Ours F1", (right_eval.get("fusion") or {}).get("f1"))

    st.markdown("**Retrieval / Token / Routing**")
    left_perf = left_trace.performance_metrics or {}
    right_perf = right_trace.performance_metrics or {}
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Baseline retrieval time", left_perf.get("total_retrieval_time"))
    r2.metric("Ours retrieval time", right_perf.get("total_retrieval_time"))
    r3.metric(
        "Baseline prompt tokens",
        (left_perf.get("token_cost") or {}).get("total_prompt_tokens"),
    )
    r4.metric(
        "Ours prompt tokens",
        (right_perf.get("token_cost") or {}).get("total_prompt_tokens"),
    )

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Baseline routing distribution (subquery-level)**")
        st.json(summarize_routing_distribution(left_trace))
        doc_dist = summarize_doc_source_distribution(left_trace)
        if doc_dist:
            st.markdown("**Baseline evidence source distribution (from docs.source_id)**")
            st.json(doc_dist)
    with d2:
        st.markdown("**Ours routing distribution (subquery-level)**")
        st.json(summarize_routing_distribution(right_trace))
        doc_dist = summarize_doc_source_distribution(right_trace)
        if doc_dist:
            st.markdown("**Ours evidence source distribution (from docs.source_id)**")
            st.json(doc_dist)
        else:
            st.info("Docs in current JSONL schema do not include source_id for multi-source selections.")


def _render_overall_summary(block: Dict[str, Any], title: str) -> None:
    st.markdown(f"### {title}")
    st.write("Output dir:", block.get("output_dir"))
    st.write("Preset:", f"{block.get('preset_label')} (`{block.get('dataset')}`)")
    st.write("Mode:", block.get("mode"))
    st.write("Custom queries mode:", block.get("custom_mode"))
    st.write("Actual run count:", block.get("actual_run_count"))
    st.write("Params:", block.get("params", {}))
    if block.get("overall_metrics"):
        st.write("Overall metrics (runner output):", block["overall_metrics"])


def _coerce_run_blocks(result_obj: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if result_obj is None:
        return None, None
    if isinstance(result_obj, dict):
        return result_obj.get("primary"), result_obj.get("baseline")
    # dataclass from current process
    primary = getattr(result_obj, "primary", None)
    baseline = getattr(result_obj, "baseline", None)
    return (
        primary.__dict__ if primary is not None else None,
        baseline.__dict__ if baseline is not None else None,
    )


def main() -> None:
    st.title("Multi-Source RAG Demo")
    st.caption("Thin UI wrapper over the existing DeepSieve runner/pipeline. Displays trace JSONL, evidence selection outputs, and Adaptive Cap parameters.")

    st.markdown("## Configuration")
    with st.expander("Open / Edit Configuration", expanded=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1.2, 1.0, 1.0])

        with cfg_col1:
            st.markdown("### Dataset / Mode")
            preset_label = st.selectbox("Dataset preset", list(PRESETS.keys()), index=0)
            preset_spec = PRESETS[preset_label]
            st.caption(f"Repo dataset: `{preset_spec.dataset}` ({preset_spec.source_count} sources)")

            mode_ui = st.radio(
                "Mode",
                ["Hard Routing (single source)", "Adaptive Cap (preferred_cap + other_cap)"],
                index=1,
            )
            mode = "adaptive_cap" if mode_ui.startswith("Adaptive") else "hard_routing"

            compare_with_baseline = st.checkbox(
                "Compare with baseline",
                value=False,
                help="Run both Hard Routing and Adaptive Cap on the same query set.",
            )

        with cfg_col2:
            st.markdown("### Retrieval / Selection")
            top_k_per_source = st.number_input("top_k_per_source", min_value=1, max_value=50, value=8, step=1)
            keep_k = st.number_input("keep_k", min_value=1, max_value=50, value=8, step=1)
            selector = st.selectbox("selector", ["score", "norm_score", "routing_weighted", "rrf", "llm"], index=0)

            preferred_cap = 5
            other_cap = 2
            if mode == "adaptive_cap" or compare_with_baseline:
                st.markdown("### Adaptive Cap")
                preferred_cap = st.number_input("preferred_cap", min_value=1, max_value=50, value=5, step=1)
                other_cap = st.number_input("other_cap", min_value=1, max_value=50, value=2, step=1)

        with cfg_col3:
            st.markdown("### Pipeline Toggles")
            decompose = st.checkbox("decompose", value=True)
            use_reflection = st.checkbox("use_reflection", value=True)
            sample_size = st.number_input("sample_size", min_value=1, max_value=1000, value=3, step=1)
            quick_mode = st.checkbox("Quick mode: run only one query_index", value=True)
            query_index = None
            if quick_mode:
                query_index = st.number_input("query_index (0-based)", min_value=0, max_value=100000, value=0, step=1)

        st.markdown("### Custom Queries")
        st.caption("Upload JSON or CSV. If uploaded, custom queries replace preset query loader while still using preset corpora.")
        upload = st.file_uploader("Upload JSON or CSV", type=["json", "csv"])
        custom_queries = None
        custom_parse_error = None
        if upload is not None:
            try:
                custom_queries = parse_uploaded_queries(upload.name, upload.getvalue())
                st.success(f"Loaded {len(custom_queries)} custom queries. Preset corpora ({preset_spec.dataset}) will still be used.")
                preview = [
                    {
                        "query": q.query[:120] + ("..." if len(q.query) > 120 else ""),
                        "ground_truth": q.ground_truth if q.has_ground_truth else None,
                        "has_ground_truth": q.has_ground_truth,
                    }
                    for q in custom_queries[:5]
                ]
                st.dataframe(preview, use_container_width=True, hide_index=True)
            except Exception as e:
                custom_parse_error = str(e)
                st.error(custom_parse_error)

        st.markdown("### Custom Source (optional)")
        st.caption("Augment the selected preset with one uploaded source corpus (used as an additional retrieval source).")
        enable_custom_source = st.checkbox("Enable custom source", value=False)
        custom_source_spec = None
        custom_source_parse_error = None
        if enable_custom_source:
            src_col1, src_col2 = st.columns([1, 2])
            with src_col1:
                custom_source_name = st.text_input("source_name", value="custom_source")
            with src_col2:
                custom_source_profile = st.text_area(
                    "source_profile (used by router)",
                    value="A user-uploaded corpus containing domain-specific documents relevant to the custom task.",
                    height=90,
                )
            source_upload = st.file_uploader(
                "Upload custom source corpus (JSON/CSV)",
                type=["json", "csv"],
                key="custom_source_uploader",
            )

            if source_upload is not None:
                try:
                    custom_docs = parse_uploaded_source_corpus(source_upload.name, source_upload.getvalue())
                    custom_source_spec = CustomSourceSpec(
                        source_name=str(custom_source_name).strip(),
                        source_profile=str(custom_source_profile).strip(),
                        docs=custom_docs,
                        input_doc_count=len(custom_docs),
                        file_name=source_upload.name,
                    )
                    st.success(f"Loaded custom source '{custom_source_spec.source_name}' with {len(custom_docs)} docs.")
                    st.caption(
                        f"Effective source count (if run): {preset_spec.source_count + 1} "
                        f"(preset {preset_spec.source_count} + custom 1)"
                    )
                    with st.expander("Custom source preview", expanded=False):
                        preview_rows = [{"rank": i + 1, "doc_preview": d[:200] + ("..." if len(d) > 200 else "")} for i, d in enumerate(custom_docs[:5])]
                        st.dataframe(preview_rows, use_container_width=True, hide_index=True)
                except Exception as e:
                    custom_source_parse_error = str(e)
                    st.error(custom_source_parse_error)
            else:
                st.info("Upload a JSON/CSV corpus file to activate the custom source.")

        run_col1, run_col2 = st.columns([1, 4])
        with run_col1:
            disabled = (
                (upload is not None and custom_parse_error is not None)
                or (enable_custom_source and (custom_source_parse_error is not None or custom_source_spec is None))
            )
            run_clicked = st.button("Run", type="primary", use_container_width=True, disabled=disabled)
        with run_col2:
            if custom_queries is not None:
                st.info(f"Custom queries mode enabled. sample_size={int(sample_size)}; actual run count will be min(sample_size, uploaded_rows) unless query_index is set.")
            elif enable_custom_source:
                st.info("Custom source mode enabled. The uploaded source will be added to the selected preset sources.")

    if run_clicked:
        cfg = DemoConfig(
            preset_label=preset_label,
            mode=mode,
            sample_size=int(sample_size),
            query_index=int(query_index) if quick_mode else None,
            rag_type="naive",
            openai_model="gpt-4o",
            openai_api_key=None,
            openai_base_url=None,
            decompose=decompose,
            use_reflection=use_reflection,
            top_k_per_source=int(top_k_per_source),
            keep_k=int(keep_k),
            selector=str(selector),
            preferred_cap=int(preferred_cap),
            other_cap=int(other_cap),
            compare_with_baseline=compare_with_baseline,
            custom_queries=custom_queries,
            custom_source=custom_source_spec,
        )
        with st.spinner("Running pipeline... this may take a while depending on model latency and retrieval size."):
            try:
                result = run_demo(cfg)
                st.session_state["demo_run_result"] = result
                st.success("Run completed.")
            except Exception as e:
                st.exception(e)

    primary_block, baseline_block = _coerce_run_blocks(st.session_state.get("demo_run_result"))
    if primary_block is None:
        last_meta = load_last_run_meta()
        if last_meta:
            st.info(f"Found previous demo run saved at {last_meta.get('saved_at')}.")
            if st.button("Load latest run metadata"):
                st.session_state["demo_run_result"] = last_meta
                primary_block, baseline_block = _coerce_run_blocks(last_meta)

    primary_block, baseline_block = _coerce_run_blocks(st.session_state.get("demo_run_result"))
    if primary_block is None:
        st.stop()

    st.divider()
    left, right = st.columns(2)
    with left:
        _render_overall_summary(primary_block, "Primary Run")
    with right:
        if baseline_block:
            _render_overall_summary(baseline_block, "Comparison Run")
        else:
            st.info("Comparison run not enabled.")

    # Keep UI robust across app restarts by reloading sidecar if query_items missing.
    if not primary_block.get("query_items"):
        sidecar = load_run_sidecar(primary_block["output_dir"])
        if sidecar:
            primary_block["query_items"] = sidecar.get("query_items", [])

    trace_files = list_trace_files(primary_block["output_dir"])
    if not trace_files:
        st.warning("No trace JSONL files found in the primary output directory.")
        st.stop()

    max_queries = len(trace_files)
    if baseline_block:
        max_queries = min(max_queries, len(list_trace_files(baseline_block["output_dir"])))
    selected_idx = st.number_input("Select query index (1-based, trace file index)", min_value=1, max_value=max_queries, value=1, step=1)
    selected_idx = int(selected_idx)
    qmeta = _find_query_meta(primary_block, selected_idx)
    if qmeta:
        st.caption(
            f"Selected query input_index={qmeta.get('input_index')} | has_ground_truth={qmeta.get('has_ground_truth')} | custom_mode={primary_block.get('custom_mode')}"
        )

    primary_trace_path = _load_trace_path_for_index(primary_block["output_dir"], selected_idx)
    if primary_trace_path is None:
        st.warning("Selected trace index is not available.")
        st.stop()

    tabs = ["Trace (Primary)"]
    if baseline_block:
        tabs.append("Compare")
        tabs.append("Trace (Comparison)")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        _render_trace(primary_trace_path, qmeta, "primary")

    if baseline_block:
        with tab_objs[1]:
            # Normalize so baseline is hard routing and ours is adaptive cap in the compare panel.
            blocks = [primary_block, baseline_block]
            hard = next((b for b in blocks if b and b.get("mode") == "hard_routing"), None)
            ours = next((b for b in blocks if b and b.get("mode") == "adaptive_cap"), None)
            if hard and ours:
                _compare_panel(primary_block=ours, baseline_block=hard, selected_index_1based=selected_idx)
            else:
                st.info("Compare panel expects one hard_routing run and one adaptive_cap run.")

        with tab_objs[2]:
            cmp_trace_path = _load_trace_path_for_index(baseline_block["output_dir"], selected_idx)
            qmeta_cmp = _find_query_meta(baseline_block, selected_idx)
            if cmp_trace_path:
                _render_trace(cmp_trace_path, qmeta_cmp, "comparison")
            else:
                st.warning("Comparison trace file not found.")


if __name__ == "__main__":
    main()

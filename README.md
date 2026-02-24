# Multi-Source RAG Demo (RealRoute)

This repository is an experimental extension of the DeepSieve-style pipeline, with a current focus on:

- Multi-source retrieval
- Evidence-level selection
- Adaptive Cap (routing-conditioned per-source quotas)
- A Streamlit demo UI for system workflow, evidence, and trace visualization


## 1. System Workflow (Current Version)

The current system workflow (as implemented in code) is:

1. Query input (from preset datasets or uploaded custom queries)
2. Optional `decompose`: split a complex question into subqueries
3. Execute subqueries in order with variable binding (placeholder substitution)
4. Retrieval stage:
   - Hard Routing: select one source first, then retrieve
   - Multi-Source / Adaptive Cap: retrieve from all sources first, then select evidence
5. Evidence selection stage:
   - `selector=score/norm_score/routing_weighted/rrf/llm`
   - Optional fixed cap or Adaptive Cap
6. Subquery answer generation (with optional reflection retries)
7. Final fusion: combine subquery answers into the final answer
8. Save trace and metrics to `outputs/`

## 2. Main Features (Current Repository)

- Original 2-source local/global pipeline (DeepSieve-style)
- 3-source / 4-source multi-source retrieval
- N-source hard routing (`--hard_routing_multi`)
- Adaptive Cap (`--preferred_cap`, `--other_cap`)
- JSONL trace outputs for case analysis
- Streamlit demo UI (supports Custom Queries / Custom Source)

## 3. Environment Setup

Use `python3` (some environments do not provide the `pip` alias, only `pip3`).

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install streamlit
```

### Configure API key

```bash
export OPENAI_API_KEY=your_api_key
```

Optional:

```bash
export OPENAI_MODEL=gpt-4o
export OPENAI_API_BASE=https://api.openai.com/v1
```

## 4. CLI Usage (Core Experiments)

Main entrypoint:

- `runner/main_rag_only.py`

### 4.1 Hard Routing (baseline)

2-source (local/global) example:

```bash
python3 runner/main_rag_only.py \
  --dataset hotpot_qa \
  --rag_type naive \
  --sample_size 100 \
  --decompose \
  --use_routing \
  --use_reflection
```

3/4-source N-way hard routing example:

```bash
python3 runner/main_rag_only.py \
  --dataset multi_source \
  --rag_type naive \
  --sample_size 100 \
  --decompose \
  --use_reflection \
  --multi_source \
  --hard_routing_multi
```

### 4.2 Adaptive Cap (ours)

Example (`top_k_per_source=8, keep_k=8, preferred_cap=5, other_cap=2`):

```bash
python3 runner/main_rag_only.py \
  --dataset mixed_4source \
  --rag_type naive \
  --sample_size 100 \
  --openai_model gpt-4o \
  --decompose \
  --use_reflection \
  --multi_source \
  --top_k_per_source 8 \
  --keep_k 8 \
  --preferred_cap 5 \
  --other_cap 2 \
  --selector score
```

Parameter meanings (current implementation):

- `top_k_per_source`: number of candidates retrieved from each source first
- `keep_k`: number of final evidences kept for generation
- `preferred_cap / other_cap`: Adaptive Cap quotas
- `selector`: evidence selection strategy (default: `score`)

Important note:
- Current Adaptive Cap is `preferred_source + fixed quotas`, not a confidence-calibrated quota policy.

## 5. Streamlit Web UI (Demo View)

Entrypoint:

- `demo/app.py`

Run:

```bash
python3 -m streamlit run demo/app.py
```

The UI supports:

- Preset datasets (2-source / 3-source / 4-source)
- Mode selection:
  - Hard Routing
  - Adaptive Cap
- Parameter controls (`top_k_per_source`, `keep_k`, `selector`, `preferred_cap`, `other_cap`)
- `decompose` / `use_reflection` toggles
- Compare with baseline (run the same queries twice)
- Trace rendering from `query_i_results.jsonl`
- Trace download (JSONL / JSON)

## 6. Custom Queries (Upload)

The Web UI supports uploading custom queries in JSON or CSV:

- Uploaded queries override the preset query loader
- The preset sources/corpora are still used for retrieval

### JSON format

```json
[
  {"query": "Who wrote ...?", "ground_truth": "..."},
  {"query": "What is ...?"}
]
```

### CSV format

Required column:

- `query`

Optional column:

- `ground_truth`

Behavior:

- If `ground_truth` is provided, the UI shows EM/F1.
- If `ground_truth` is missing, the UI shows answer + evidence trace only.

## 7. Custom Source (Upload)

The Web UI can add one extra source on top of the selected preset sources (for demoing extension to a new source):

- `source_name`
- `source_profile` (used by the router)
- Upload corpus (JSON/CSV)

### Custom Source JSON format (recommended)

```json
[
  {"title": "Doc 1", "text": "Document content..."},
  {"title": "Doc 2", "text": "Another content..."}
]
```

Also supported:

```json
[
  "plain document text 1",
  "plain document text 2"
]
```

### Custom Source CSV format

Required column:

- `text`

Optional column:

- `title`

Notes:

- The uploaded source is added to the preset sources (it does not replace them).
- Example: 3-source preset + custom source => effective 4-source run.

## 8. Outputs and Trace

Each run creates a directory under `outputs/`. Directory names encode key settings (mode, k, cap, selector, etc.).

Common outputs:

- `query_{i}_results.jsonl`: per-query trace
- `query_{i}_fusion_prompt.txt`: fusion-stage prompt
- `overall_results.json`: aggregated metrics
- `overall_results.txt`: readable summary

### Common JSONL record types

- `query_info`
- `final_answer`
- `evaluation_metrics`
- `performance_metrics`
- `execution_result` (one per subquery)
- `fused_answer_step`

## 9. Batch Experiment Scripts

Located in `scripts/`:

- `run_gpt4o_adaptive_cap.sh`
- `run_gpt4o_k8_cap3.sh`
- `run_all_gpt4o.sh`

Check script parameters before running.

## 10. Relation to the Original DeepSieve Baseline

This repository keeps DeepSieve-style components:

- decomposition
- routing
- reflection
- final fusion

and extends them with:

- multi-source retrieval
- evidence selector
- Adaptive Cap
- demo UI / trace visualization

For the original DeepSieve work, please refer to the original repository and paper.


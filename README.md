# Multi-Source RAG Demo (RealRoute)

This repository provides an installable demo package for a DeepSieve-derived multi-source RAG system with:

- Multi-source retrieval
- Evidence-level selection
- Adaptive Cap (`preferred_source` + fixed quotas)
- Streamlit UI for workflow, evidence, and trace inspection

## Installable Package Link

- Source repository: `https://github.com/Joseph1951210/RealRoute`
- Installable package (ACL demo artifact): `https://github.com/Joseph1951210/RealRoute/archive/refs/tags/v1.0-acl-demo.zip`

## Environment Requirements

- Python 3.10+ (recommended: Python 3.10 or 3.11)
- macOS/Linux shell
- OpenAI-compatible API key (`OPENAI_API_KEY`)

## Quick Start (Demo UI)

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
python3 -m streamlit run demo/app.py
```

Open the local Streamlit URL shown in the terminal (typically `http://localhost:8501`).

## Dataset Availability in Release `v1.0-acl-demo`

The `v1.0-acl-demo` package includes tracked datasets required for the original 2-source preset (e.g., `hotpot_qa` local/global files).

For 3-source / 4-source presets (`multi_source`, `mixed_4source`), make sure the corresponding files exist under `data/rag/` before running those presets:

- `{dataset}.json`
- `{dataset}_profiles.json`
- `{dataset}_corpus_*.json`

If these files are missing, those presets will fail at data-loading time.

## What the Demo Shows

1. Configure dataset preset and mode (`Hard Routing` or `Adaptive Cap`)
2. Run the pipeline with configurable retrieval/selection parameters
3. Inspect run-level summary (output directory, config snapshot, overall metrics)
4. Inspect query-level traces (subqueries, routing, evidence, final answer, metrics)
5. Compare baseline vs Adaptive Cap in the same UI

## System Workflow (Implemented Behavior)

1. Query input (preset dataset or uploaded custom queries)
2. Optional decomposition into subqueries (`decompose`)
3. Ordered subquery execution with variable binding
4. Retrieval:
   - Hard Routing: route to one source then retrieve
   - Adaptive Cap mode: retrieve from all sources, then select evidence
5. Evidence selection using `selector` with optional Adaptive Cap
6. Subquery answer generation (optional reflection retries)
7. Final answer fusion
8. Save traces and run summaries under `outputs/`

## Main Entrypoints

- CLI pipeline: `runner/main_rag_only.py`
- Web UI: `demo/app.py`

## CLI Examples

### Hard Routing (2-source baseline)

```bash
python3 runner/main_rag_only.py \
  --dataset hotpot_qa \
  --rag_type naive \
  --sample_size 100 \
  --decompose \
  --use_routing \
  --use_reflection
```

### Hard Routing (N-source: 3/4-source datasets)

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

### Adaptive Cap (example configuration)

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

## Parameter Notes

- `top_k_per_source`: candidates retrieved per source before selection
- `keep_k`: final evidence budget for answer generation
- `preferred_cap` / `other_cap`: source quota in Adaptive Cap mode
- `selector`: evidence selector (`score`, `norm_score`, `routing_weighted`, `rrf`, `llm`)

Important: current Adaptive Cap is not confidence-calibrated. It uses routing-preferred source + fixed quotas.

## Web UI Features

- Dataset presets: 2-source / 3-source / 4-source
- Mode toggle: Hard Routing vs Adaptive Cap
- Pipeline toggles: `decompose`, `use_reflection`, `sample_size`, optional `query_index`
- Compare mode: run baseline and Adaptive Cap on the same query set
- Trace view tabs: primary trace, side-by-side compare, comparison trace
- Trace download: JSONL and JSON

## Custom Queries Upload

Custom queries override preset query loading but still use preset corpora.

Supported formats:

- JSON:

```json
[
  {"query": "Who wrote ...?", "ground_truth": "..."},
  {"query": "What is ...?"}
]
```

- CSV:
  - required column: `query`
  - optional column: `ground_truth`

If `ground_truth` is provided, EM/F1 is shown; otherwise answer/trace only.

## Custom Source Upload (Optional)

The UI can add one uploaded source corpus to the selected preset sources.

- Required fields: `source_name`, `source_profile`
- File format: JSON or CSV

JSON examples:

```json
[
  {"title": "Doc 1", "text": "Document content..."},
  {"title": "Doc 2", "text": "Another content..."}
]
```

or

```json
[
  "plain document text 1",
  "plain document text 2"
]
```

CSV:

- required: `text`
- optional: `title`

## Output Artifacts

Each run writes to a directory in `outputs/` (directory name encodes key settings).

Common files:

- `query_{i}_results.jsonl`
- `query_{i}_fusion_prompt.txt`
- `overall_results.json`
- `overall_results.txt`
- `demo_run_meta.json` (UI run metadata)

Typical JSONL record types:

- `query_info`
- `final_answer`
- `evaluation_metrics`
- `performance_metrics`
- `execution_result`
- `fused_answer_step`

## Common Runtime Issues

- `OPENAI_API_KEY is required`:
  - Ensure `export OPENAI_API_KEY=...` is executed in the same shell session before launching Streamlit.
- `pip: command not found`:
  - Use `python3 -m pip ...` instead of `pip ...`.

## Release Checklist (Installable Package for ACL Demo)

1. Push the latest code to GitHub.
2. Create a version tag (e.g., `v1.0-acl-demo`).
3. Create a GitHub Release from that tag.
4. Upload a downloadable source archive (`.zip` or `.tar.gz`) as release asset.
5. Replace `[[TODO: add GitHub Release asset URL]]` above with the release asset link.

## Relation to the Original DeepSieve Baseline

This repository keeps DeepSieve-style components (decomposition, routing, reflection, fusion) and extends them with multi-source retrieval, evidence selection, Adaptive Cap, and a trace-oriented demo UI.

#!/bin/bash
# ============================================================
# Run 5 experiments with gpt-4o (skip A1, paper already has it)
# ============================================================
# A2: 2-source multi-source (ours) — 300 samples
# D1: 4-source hard routing         — 300 samples
# D2: 4-source multi-source k5 cap2 — 300 samples
# B:  3-source hard routing          — 300 samples
# C:  3-source multi-source k5 cap2  — 300 samples
# ============================================================

set -e

if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Python not found"
    exit 1
fi
echo "Using Python: $PYTHON"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set. Please run: export OPENAI_API_KEY=your_key"
    exit 1
fi

MODEL="gpt-4o"
RAG_TYPE="naive"
SAMPLES=300

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo "Model: $MODEL"
echo "Samples per experiment: $SAMPLES"
echo ""

TOTAL_START=$(date +%s)

# ============================================================
# A2: 2-source Multi-source (ours)
# ============================================================
echo "============================================================"
echo "  [1/5] A2: 2-source Multi-source ($SAMPLES samples)"
echo "============================================================"
echo "Start: $(date)"
T_START=$(date +%s)

$PYTHON runner/main_rag_only.py \
    --dataset hotpot_qa \
    --rag_type $RAG_TYPE \
    --sample_size $SAMPLES \
    --openai_model $MODEL \
    --decompose \
    --use_reflection \
    --multi_source \
    --top_k_per_source 5 \
    --keep_k 5 \
    --per_source_cap 2 \
    --selector score

T_END=$(date +%s)
echo "A2 done in $((T_END - T_START))s"
echo ""

# ============================================================
# D1: 4-source Hard Routing
# ============================================================
echo "============================================================"
echo "  [2/5] D1: 4-source Hard Routing ($SAMPLES samples)"
echo "============================================================"
echo "Start: $(date)"
T_START=$(date +%s)

$PYTHON runner/main_rag_only.py \
    --dataset mixed_4source \
    --rag_type $RAG_TYPE \
    --sample_size $SAMPLES \
    --openai_model $MODEL \
    --decompose \
    --use_reflection \
    --hard_routing_multi \
    --multi_source

T_END=$(date +%s)
echo "D1 done in $((T_END - T_START))s"
echo ""

# ============================================================
# D2: 4-source Multi-source k5 cap2
# ============================================================
echo "============================================================"
echo "  [3/5] D2: 4-source Multi-source k5 cap2 ($SAMPLES samples)"
echo "============================================================"
echo "Start: $(date)"
T_START=$(date +%s)

$PYTHON runner/main_rag_only.py \
    --dataset mixed_4source \
    --rag_type $RAG_TYPE \
    --sample_size $SAMPLES \
    --openai_model $MODEL \
    --decompose \
    --use_reflection \
    --multi_source \
    --top_k_per_source 5 \
    --keep_k 5 \
    --per_source_cap 2 \
    --selector score

T_END=$(date +%s)
echo "D2 done in $((T_END - T_START))s"
echo ""

# ============================================================
# B: 3-source Hard Routing
# ============================================================
echo "============================================================"
echo "  [4/5] B: 3-source Hard Routing ($SAMPLES samples)"
echo "============================================================"
echo "Start: $(date)"
T_START=$(date +%s)

$PYTHON runner/main_rag_only.py \
    --dataset multi_source \
    --rag_type $RAG_TYPE \
    --sample_size $SAMPLES \
    --openai_model $MODEL \
    --decompose \
    --use_reflection \
    --hard_routing_multi \
    --multi_source

T_END=$(date +%s)
echo "B done in $((T_END - T_START))s"
echo ""

# ============================================================
# C: 3-source Multi-source k5 cap2
# ============================================================
echo "============================================================"
echo "  [5/5] C: 3-source Multi-source k5 cap2 ($SAMPLES samples)"
echo "============================================================"
echo "Start: $(date)"
T_START=$(date +%s)

$PYTHON runner/main_rag_only.py \
    --dataset multi_source \
    --rag_type $RAG_TYPE \
    --sample_size $SAMPLES \
    --openai_model $MODEL \
    --decompose \
    --use_reflection \
    --multi_source \
    --top_k_per_source 5 \
    --keep_k 5 \
    --per_source_cap 2 \
    --selector score

T_END=$(date +%s)
echo "C done in $((T_END - T_START))s"
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo "============================================================"
echo "  All 5 experiments complete! Total: ${TOTAL}s"
echo "  Model: $MODEL"
echo "============================================================"
echo ""
echo "Results in outputs/ directories (with gpt-4o in name):"
ls -d outputs/*gpt-4o* 2>/dev/null || echo "  (check outputs/ manually)"
echo ""
echo "Done!"

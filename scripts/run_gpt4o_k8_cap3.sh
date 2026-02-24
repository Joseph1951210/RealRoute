#!/bin/bash
# ============================================================
# Run only OUR multi-source experiments with k=8 cap=3 (gpt-4o)
# ============================================================
# A2: 2-source multi-source  — 300 samples
# D2: 4-source multi-source  — 300 samples
# C:  3-source multi-source  — 300 samples
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
KEEP_K=8
CAP=3

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo "Model: $MODEL  |  keep_k=$KEEP_K  |  per_source_cap=$CAP"
echo ""

TOTAL_START=$(date +%s)

# ============================================================
# A2: 2-source Multi-source
# ============================================================
echo "============================================================"
echo "  [1/3] A2: 2-source Multi-source ($SAMPLES samples, k=$KEEP_K cap=$CAP)"
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
    --keep_k $KEEP_K \
    --per_source_cap $CAP \
    --selector score

T_END=$(date +%s)
echo "A2 done in $((T_END - T_START))s"
echo ""

# ============================================================
# D2: 4-source Multi-source
# ============================================================
echo "============================================================"
echo "  [2/3] D2: 4-source Multi-source ($SAMPLES samples, k=$KEEP_K cap=$CAP)"
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
    --keep_k $KEEP_K \
    --per_source_cap $CAP \
    --selector score

T_END=$(date +%s)
echo "D2 done in $((T_END - T_START))s"
echo ""

# ============================================================
# C: 3-source Multi-source
# ============================================================
echo "============================================================"
echo "  [3/3] C: 3-source Multi-source ($SAMPLES samples, k=$KEEP_K cap=$CAP)"
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
    --keep_k $KEEP_K \
    --per_source_cap $CAP \
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
echo "  All 3 experiments complete! Total: ${TOTAL}s"
echo "  Model: $MODEL  |  keep_k=$KEEP_K  |  per_source_cap=$CAP"
echo "============================================================"
echo ""
echo "Results:"
ls -d outputs/*gpt-4o*k${KEEP_K}*cap${CAP}* 2>/dev/null || echo "  (check outputs/ manually)"
echo ""
echo "Done!"

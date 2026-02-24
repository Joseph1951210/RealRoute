#!/bin/bash
# ============================================================
# Test adaptive cap: preferred=5, others=2 (gpt-4o)
# top_k_per_source=8, keep_k=8, no fixed cap
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
TOP_K=8
PREFERRED_CAP=5
OTHER_CAP=2

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo "Model: $MODEL | top_k=$TOP_K | keep_k=$KEEP_K | adaptive cap: preferred=$PREFERRED_CAP others=$OTHER_CAP"
echo ""

TOTAL_START=$(date +%s)

# ============================================================
# 1. 4-source adaptive cap
# ============================================================
echo "============================================================"
echo "  [1/3] D2: 4-source adaptive cap ($SAMPLES samples)"
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
    --top_k_per_source $TOP_K \
    --keep_k $KEEP_K \
    --preferred_cap $PREFERRED_CAP \
    --other_cap $OTHER_CAP \
    --selector score

T_END=$(date +%s)
echo "Done in $((T_END - T_START))s"
echo ""

# ============================================================
# 2. 3-source adaptive cap
# ============================================================
echo "============================================================"
echo "  [2/3] C: 3-source adaptive cap ($SAMPLES samples)"
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
    --top_k_per_source $TOP_K \
    --keep_k $KEEP_K \
    --preferred_cap $PREFERRED_CAP \
    --other_cap $OTHER_CAP \
    --selector score

T_END=$(date +%s)
echo "Done in $((T_END - T_START))s"
echo ""

# ============================================================
# 3. 2-source adaptive cap (hotpot_qa)
# ============================================================
echo "============================================================"
echo "  [3/3] A2: 2-source adaptive cap ($SAMPLES samples)"
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
    --top_k_per_source $TOP_K \
    --keep_k $KEEP_K \
    --preferred_cap $PREFERRED_CAP \
    --other_cap $OTHER_CAP \
    --selector score

T_END=$(date +%s)
echo "Done in $((T_END - T_START))s"
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo "============================================================"
echo "  All 3 experiments complete! Total: ${TOTAL}s"
echo "  Config: top_k=$TOP_K keep_k=$KEEP_K preferred=$PREFERRED_CAP others=$OTHER_CAP"
echo "============================================================"
echo ""
echo "Results:"
ls -d outputs/*acap* 2>/dev/null || echo "  (check outputs/ manually)"
echo ""
echo "Done!"

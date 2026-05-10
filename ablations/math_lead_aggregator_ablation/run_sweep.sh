#!/bin/bash
# =============================================================================
# Driver for Ablation 4: how to aggregate correct lengths into L*_q
# =============================================================================
# We already have AGG=mean_correct from the main LEAD run; this driver
# fills in the three missing variants needed for the ablation table.
#
# Usage:
#   bash run_sweep.sh
#   AGGS="min_correct" bash run_sweep.sh           # subset
#   CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 bash run_sweep.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_lead_aggregator.sh"

# Default: the three missing variants (mean_correct is the main LEAD run)
AGGS="${AGGS:-min_correct median_correct mean_all}"

echo "==========================================================="
echo "  Ablation 4: L*_q aggregator sweep"
echo "  Aggregators: $AGGS"
echo "  Base: DeepSeek-R1-Distill-Qwen-1.5B (LEAD, 4K)"
echo "==========================================================="

for AGG in $AGGS; do
    CKPT_DIR="${OUTPUT_ROOT:-./results}/math_lead_agg_${AGG}_deepseek-r1-1.5b"
    if [ -d "$CKPT_DIR/actor/global_step_460" ] || [ -d "$CKPT_DIR/actor/global_step_461" ]; then
        echo "[skip] aggregator=$AGG already completed at $CKPT_DIR"
        continue
    fi
    echo ""
    echo " Training with aggregator=$AGG"
    echo " -> $CKPT_DIR"
    AGG="$AGG" bash "$TRAIN_SCRIPT"
done

echo ""
echo "==========================================================="
echo " Ablation 4 sweep complete."
echo "==========================================================="

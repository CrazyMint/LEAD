#!/bin/bash
# =============================================================================
# Driver: ratio sweep for Ablation 1
#   GRPO  (combine-then-normalize, sweep lambda_2/lambda_1)
#   LEAD  (decoupled normalize-then-combine, sweep w_eff/w_corr; STATIC weights)
# =============================================================================
# Both methods share the same RATIO grid so results are directly comparable.
# Resume-friendly: any (method, ratio) whose final actor checkpoint is at
# global_step >= DONE_STEP (default 450 — your eval convention) is skipped.
#
# Usage:
#   bash run_sweep.sh                              # both methods, 6 ratios each
#   METHODS="grpo"        bash run_sweep.sh        # only GRPO
#   METHODS="lead"        bash run_sweep.sh        # only LEAD
#   RATIOS="0.5 1.0 4.0"  bash run_sweep.sh        # subset of ratios
#   DONE_STEP=460         bash run_sweep.sh        # stricter "done" threshold
#   CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 bash run_sweep.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_TRAIN="$SCRIPT_DIR/train_math_grpo_lambda_1.5b.sh"
LEAD_TRAIN="$SCRIPT_DIR/../math_lead_lambda_sweep/train_math_lead_static_1.5b.sh"

METHODS="${METHODS:-grpo lead}"
RATIOS="${RATIOS:-0.0 0.25 0.5 1.0 2.0 4.0}"
DONE_STEP="${DONE_STEP:-450}"

echo "==========================================================="
echo "  Ablation 1: ratio sweep"
echo "  Methods : $METHODS"
echo "  Ratios  : $RATIOS"
echo "  Done @  : global_step >= $DONE_STEP"
echo "==========================================================="

# ---- helper: returns "done" / "partial" / "none" given a checkpoint dir ----
ckpt_status() {
    local dir="$1"
    [ -d "$dir/actor" ] || { echo "none"; return; }
    local latest
    latest=$(ls -d "$dir/actor/global_step_"* 2>/dev/null \
             | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$latest" ]; then
        echo "none"
    elif [ "$latest" -ge "$DONE_STEP" ]; then
        echo "done($latest)"
    else
        echo "partial($latest)"
    fi
}

run_one() {
    local method="$1"
    local ratio="$2"
    local tag
    tag=$(printf '%s' "$ratio" | tr -d '.')
    local ckpt_dir="${OUTPUT_ROOT:-./results}/math_${method}_ratio${tag}_deepseek-r1-1.5b"
    local status
    status=$(ckpt_status "$ckpt_dir")

    case "$status" in
        done*)
            printf "  [skip] %-5s ratio=%-5s : already %s at %s\n" "$method" "$ratio" "$status" "$ckpt_dir"
            return
            ;;
        partial*)
            printf "  [restart] %-5s ratio=%-5s : %s found, deleting and re-running\n" "$method" "$ratio" "$status"
            rm -rf "$ckpt_dir"
            ;;
        none)
            printf "  [start] %-5s ratio=%-5s : training fresh\n" "$method" "$ratio"
            ;;
    esac

    if [ "$method" = "grpo" ]; then
        RATIO="$ratio" CKPT_DIR="$ckpt_dir" bash "$GRPO_TRAIN"
    else
        RATIO="$ratio" CKPT_DIR="$ckpt_dir" bash "$LEAD_TRAIN"
    fi
}

for METHOD in $METHODS; do
    echo ""
    echo "========================== $METHOD =========================="
    for R in $RATIOS; do
        run_one "$METHOD" "$R"
    done
done

echo ""
echo "==========================================================="
echo " Sweep complete. Status:"
echo "==========================================================="
for METHOD in $METHODS; do
    for R in $RATIOS; do
        TAG=$(printf '%s' "$R" | tr -d '.')
        DIR="${OUTPUT_ROOT:-./results}/math_${METHOD}_ratio${TAG}_deepseek-r1-1.5b"
        printf "  %-5s ratio=%-5s : %s\n" "$METHOD" "$R" "$(ckpt_status $DIR)"
    done
done

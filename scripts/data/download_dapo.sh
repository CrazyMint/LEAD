#!/bin/bash
# Download and prepare DAPO-Math-17k dataset for GDPO/DGDO2 training
#
# Uses the same parquet format and prompt template as DeepScaler, so the
# existing deepscale.py reward function works unchanged.
#
# Usage:
#   bash scripts/data/download_dapo.sh
#
# Or with custom settings:
#   DATA_DIR=/path/to/data NUM_SAMPLES=1000 bash scripts/data/download_dapo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/dapo}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SEED="${SEED:-42}"

echo "================================================"
echo "DAPO-Math-17k Dataset Download Script"
echo "================================================"
echo "Output directory: $DATA_DIR"
echo "Num samples: $NUM_SAMPLES (0 = all unique)"
echo ""

# Create output directory
mkdir -p "$DATA_DIR"

# Generate datasets
echo "Downloading and processing DAPO-Math-17k dataset..."
cd "$REPO_ROOT"
python scripts/data/prepare_dapo.py \
    --local_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED"

echo ""
echo "================================================"
echo "Download complete!"
echo "================================================"
echo ""
echo "Dataset files created in: $DATA_DIR"
ls -lh "$DATA_DIR"/*.parquet 2>/dev/null || echo "  (no parquet files found)"
echo ""
echo "To use in training:"
echo "  data.train_files=$DATA_DIR/train.parquet"
echo "  data.val_files=$DATA_DIR/test.parquet"
echo ""

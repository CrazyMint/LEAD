#!/bin/bash
# Download and prepare MATH (Hendrycks) Level 3-5 dataset for LEAD training
#
# Uses the same parquet format and prompt template as DeepScaler, so the
# existing deepscale.py reward function works unchanged.
#
# Usage:
#   bash scripts/data/download_math.sh
#
# Or with custom settings:
#   DATA_DIR=/path/to/data MIN_LEVEL=2 MAX_LEVEL=5 bash scripts/data/download_math.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/math}"
MIN_LEVEL="${MIN_LEVEL:-3}"
MAX_LEVEL="${MAX_LEVEL:-5}"

echo "================================================"
echo "MATH Dataset (Level ${MIN_LEVEL}-${MAX_LEVEL}) Download Script"
echo "================================================"
echo "Output directory: $DATA_DIR"
echo ""

# Create output directory
mkdir -p "$DATA_DIR"

# Generate datasets
echo "Downloading and processing MATH dataset..."
cd "$REPO_ROOT"
python scripts/data/prepare_math.py \
    --local_dir "$DATA_DIR" \
    --min_level "$MIN_LEVEL" \
    --max_level "$MAX_LEVEL"

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

#!/bin/bash
# Download and prepare DeepScaler dataset for GDPO/DGDO training
#
# This script clones the DeepScaleR repository and uses its data utilities
# to generate the training and evaluation datasets.
#
# Usage:
#   bash scripts/data/download_deepscaler.sh
#
# Or with custom output directory:
#   DATA_DIR=/path/to/data bash scripts/data/download_deepscaler.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Output directory for datasets
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/deepscaler}"

# Temporary directory for cloning
TMP_DIR="${TMP_DIR:-/tmp/deepscaler_download_$$}"

echo "================================================"
echo "DeepScaler Dataset Download Script"
echo "================================================"
echo "Output directory: $DATA_DIR"
echo "Temporary directory: $TMP_DIR"
echo ""

# Create output directory
mkdir -p "$DATA_DIR"

# Clone DeepScaleR repository
echo "[1/3] Cloning DeepScaleR repository..."
if [ -d "$TMP_DIR/deepscaler" ]; then
    echo "  Using existing clone at $TMP_DIR/deepscaler"
else
    git clone --depth 1 https://github.com/agentica-project/deepscaler.git "$TMP_DIR/deepscaler"
fi

# Install deepscaler package (for data utilities)
echo ""
echo "[2/3] Installing deepscaler package..."
cd "$TMP_DIR/deepscaler"
pip install -e . --quiet

# Generate datasets using prepare_deepscaler.py
echo ""
echo "[3/3] Generating datasets..."
cd "$REPO_ROOT"
python scripts/data/prepare_deepscaler.py --local_dir "$DATA_DIR"

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
echo "  data.val_files=$DATA_DIR/aime.parquet"
echo ""

# Cleanup prompt
read -p "Remove temporary directory $TMP_DIR? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TMP_DIR"
    echo "Cleaned up temporary directory."
fi

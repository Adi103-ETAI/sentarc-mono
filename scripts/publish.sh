#!/usr/bin/env bash
# Publish sentarc packages to PyPI
# Usage: ./scripts/publish.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "=== Sentarc PyPI Publisher ==="
echo ""

# Step 1: Get API token
read -sp "Enter your PyPI API token: " TWINE_PASSWORD
echo ""
if [[ -z "$TWINE_PASSWORD" ]]; then
    echo "Error: API token cannot be empty"
    exit 1
fi

export TWINE_USERNAME="__token__"
export TWINE_PASSWORD

# Step 2: Check if build and twine are installed
echo "Checking dependencies..."
pip install -q build twine || {
    echo "Error: Failed to install build tools"
    exit 1
}
echo "✓ Dependencies installed"
echo ""

# Step 3: Clean previous builds
echo "Cleaning previous builds..."
rm -rf packages/*/dist packages/*/build packages/**/*.egg-info 2>/dev/null || true
echo "✓ Cleaned"
echo ""

# Step 4: Build packages
echo "Building packages..."
echo ""

packages=("ai" "agent" "tui" "coding-agent")
for pkg in "${packages[@]}"; do
    echo "Building sentarc-$pkg..."
    cd "$ROOT_DIR/packages/$pkg"
    python -m build -q
    echo "✓ sentarc-$pkg built"
done

cd "$ROOT_DIR"
echo ""

# Step 5: Upload packages
echo "Uploading to PyPI..."
echo ""

# Upload in dependency order
upload_order=("ai" "agent" "tui" "coding-agent")

for pkg in "${upload_order[@]}"; do
    echo "Uploading sentarc-$pkg..."
    twine upload --non-interactive packages/$pkg/dist/* 2>&1 | grep -v "Skipping" || true
    echo "✓ sentarc-$pkg uploaded"
    sleep 2  # Small delay between uploads
done

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Packages are now on PyPI!"
echo ""
echo "Test with:"
echo "  pip install --upgrade sentarc-ai"
echo "  pip install --upgrade sentarc-agent"
echo "  pip install --upgrade sentarc-tui"
echo "  pip install --upgrade sentarc-coding-agent"
echo "  arc --help"

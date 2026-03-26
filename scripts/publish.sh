#!/usr/bin/env bash
# Publish sentarc packages to PyPI
# Usage: ./scripts/publish.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "=== Sentarc PyPI Publisher ==="
echo ""

# Step 1: Get API token
if [[ -n "${TWINE_PASSWORD:-}" ]]; then
    echo "Using TWINE_PASSWORD from environment"
elif [[ -t 0 ]]; then
    read -rsp "Enter your PyPI API token: " TWINE_PASSWORD
    echo ""
else
    echo "Error: No interactive input available and TWINE_PASSWORD is not set"
    echo "Set token via environment, for example:"
    echo "  export TWINE_USERNAME=__token__"
    echo "  export TWINE_PASSWORD='pypi-...'"
    exit 1
fi

if [[ -z "${TWINE_PASSWORD:-}" ]]; then
    echo "Error: API token cannot be empty"
    exit 1
fi

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
export TWINE_PASSWORD

# Step 2: Check if build and twine are installed
echo "Checking dependencies..."
python -m pip install -q build twine || {
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
    python -m twine check dist/* >/dev/null
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
    set +e
    upload_output=$(twine upload --non-interactive packages/$pkg/dist/* 2>&1)
    upload_rc=$?
    set -e

    echo "$upload_output"

    if [[ $upload_rc -ne 0 ]]; then
        echo "✗ sentarc-$pkg upload failed"
        if echo "$upload_output" | grep -qi "file already exists\|400 Bad Request"; then
            echo "Hint: this version may already be published on PyPI."
            echo "Bump versions first, then rebuild and upload:"
            echo "  python scripts/sync-versions.py --bump patch"
            echo "  ./scripts/publish.sh"
        fi
        exit $upload_rc
    fi

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

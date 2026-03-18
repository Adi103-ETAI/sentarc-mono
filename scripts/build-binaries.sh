#!/usr/bin/env bash
#
# Build arc binaries for all platforms.
#
# Usage:
#   ./scripts/build-binaries.sh [--platform <platform>]
#
# Options:
#   --platform <name>   Build only for specified platform (darwin-arm64, darwin-x64, linux-x64, linux-arm64, windows-x64)
#
# Output:
#   packages/coding-agent/binaries/
#     arc-darwin-arm64.tar.gz
#     arc-darwin-x64.tar.gz
#     arc-linux-x64.tar.gz
#     arc-linux-arm64.tar.gz
#     arc-windows-x64.zip
#
# Requirements:
#   pip install pyinstaller
#
# Note: Cross-compilation requires running on the target platform or using Docker/CI.
#       This script builds for the current platform by default.

set -euo pipefail

cd "$(dirname "$0")/.."

PLATFORM=""
CURRENT_PLATFORM=""

# Detect current platform
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) CURRENT_PLATFORM="darwin-arm64" ;;
    Darwin-x86_64) CURRENT_PLATFORM="darwin-x64" ;;
    Linux-x86_64) CURRENT_PLATFORM="linux-x64" ;;
    Linux-aarch64) CURRENT_PLATFORM="linux-arm64" ;;
    MINGW*|MSYS*|CYGWIN*) CURRENT_PLATFORM="windows-x64" ;;
    *) echo "Unknown platform: $(uname -s)-$(uname -m)"; exit 1 ;;
esac

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate platform if specified
if [[ -n "$PLATFORM" ]]; then
    case "$PLATFORM" in
        darwin-arm64|darwin-x64|linux-x64|linux-arm64|windows-x64)
            ;;
        *)
            echo "Invalid platform: $PLATFORM"
            echo "Valid platforms: darwin-arm64, darwin-x64, linux-x64, linux-arm64, windows-x64"
            exit 1
            ;;
    esac
else
    PLATFORM="$CURRENT_PLATFORM"
fi

echo "==> Building for platform: $PLATFORM"
echo "    (Cross-compilation requires Docker or CI - this builds for current platform only)"
echo ""

# Check if we can build for the requested platform
if [[ "$PLATFORM" != "$CURRENT_PLATFORM" ]]; then
    echo "Warning: Cross-compilation not supported natively."
    echo "         Current platform: $CURRENT_PLATFORM"
    echo "         Requested platform: $PLATFORM"
    echo ""
    echo "For cross-platform builds, use GitHub Actions or Docker."
    exit 1
fi

echo "==> Installing dependencies..."
pip install -e packages/ai -e packages/agent -e packages/tui -e packages/coding-agent
pip install pyinstaller

echo "==> Building binary with PyInstaller..."
cd packages/coding-agent

# Clean previous builds
rm -rf build dist binaries
mkdir -p binaries/$PLATFORM

# Get version from pyproject.toml
VERSION=$(grep 'version = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "    Version: $VERSION"

# Create PyInstaller spec for arc
pyinstaller \
    --name arc \
    --onefile \
    --clean \
    --noconfirm \
    --add-data "../../packages/ai/src:sentarc_ai" \
    --add-data "../../packages/agent/src:sentarc_agent" \
    --add-data "../../packages/tui/src:sentarc_tui" \
    --hidden-import sentarc_ai \
    --hidden-import sentarc_agent \
    --hidden-import sentarc_tui \
    --hidden-import sentarc_coding_agent \
    --hidden-import anthropic \
    --hidden-import openai \
    --hidden-import google.generativeai \
    --hidden-import boto3 \
    --hidden-import httpx \
    --hidden-import textual \
    --hidden-import rich \
    src/cli/__init__.py

# Move binary to platform directory
if [[ "$PLATFORM" == "windows-x64" ]]; then
    mv dist/arc.exe binaries/$PLATFORM/
else
    mv dist/arc binaries/$PLATFORM/
fi

# Copy additional files
cp pyproject.toml binaries/$PLATFORM/
[[ -f README.md ]] && cp README.md binaries/$PLATFORM/ || true
[[ -f CHANGELOG.md ]] && cp CHANGELOG.md binaries/$PLATFORM/ || true

echo "==> Creating release archive..."
cd binaries

if [[ "$PLATFORM" == "windows-x64" ]]; then
    echo "Creating arc-$PLATFORM.zip..."
    (cd $PLATFORM && zip -r ../arc-$PLATFORM.zip .)
else
    echo "Creating arc-$PLATFORM.tar.gz..."
    mv $PLATFORM arc && tar -czf arc-$PLATFORM.tar.gz arc && mv arc $PLATFORM
fi

# Clean up build artifacts
cd ..
rm -rf build dist *.spec

echo ""
echo "==> Build complete!"
echo "Archive available at: packages/coding-agent/binaries/arc-$PLATFORM.tar.gz"
ls -lh binaries/*.tar.gz binaries/*.zip 2>/dev/null || true

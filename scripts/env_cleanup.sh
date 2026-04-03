#!/bin/bash
# scripts/env_cleanup.sh — Cleanup corrupted venv and artifacts

echo "🧹 Cleaning up Aura environment..."

WORKSPACE_DIR="/Users/bryan/Desktop/aura"

if [ -d "$WORKSPACE_DIR/venv" ]; then
    echo "⚠️ Found corrupted venv directory (Python 3.14). Removing..."
    rm -rf "$WORKSPACE_DIR/venv"
    echo "✅ Corrupted venv removed."
else
    echo "ℹ️ No corrupted venv directory found."
fi

# Clean up other potential build artifacts that cause shadowing
echo "🧹 Cleaning up build artifacts..."
find "$WORKSPACE_DIR" -type d -name "build" -exec rm -rf {} +
find "$WORKSPACE_DIR" -type d -name "dist" -exec rm -rf {} +
find "$WORKSPACE_DIR" -type d -name "*.egg-info" -exec rm -rf {} +

echo "✨ Environment cleanup complete. Please ensure you are using '.venv' (Python 3.12)."

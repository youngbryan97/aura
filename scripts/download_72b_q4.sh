#!/bin/bash
# Downloads Qwen2.5-72B-Instruct Q4_K_M (12 shards, ~44GB total)
# Retries each shard until complete. Run from aura root.
set -e
cd "$(dirname "$0")/../models_gguf"

BASE_URL="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main"
SHARDS=12

for i in $(seq -w 1 $SHARDS); do
    SHARD="qwen2.5-72b-instruct-q4_k_m-000${i}-of-000${SHARDS}.gguf"
    
    # Expected sizes (approximate, in bytes)
    EXPECTED=3700000000
    if [ "$i" = "11" ]; then EXPECTED=3300000000; fi
    if [ "$i" = "12" ]; then EXPECTED=1000000000; fi
    
    # Check if already complete
    if [ -f "$SHARD" ]; then
        SIZE=$(stat -f%z "$SHARD" 2>/dev/null || stat -c%s "$SHARD" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt "$EXPECTED" ]; then
            echo "✅ $SHARD already complete ($(echo "scale=1; $SIZE/1073741824" | bc)GB)"
            continue
        fi
    fi
    
    echo "⬇️  Downloading $SHARD..."
    while true; do
        curl -L -C - --retry 5 --retry-delay 10 --connect-timeout 30 \
            -o "$SHARD" "${BASE_URL}/${SHARD}" 2>&1 | tail -1
        
        SIZE=$(stat -f%z "$SHARD" 2>/dev/null || stat -c%s "$SHARD" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt "$EXPECTED" ]; then
            echo "✅ $SHARD complete ($(echo "scale=1; $SIZE/1073741824" | bc)GB)"
            break
        fi
        echo "⚠️  $SHARD incomplete ($SIZE bytes). Retrying in 10s..."
        sleep 10
    done
done

echo ""
echo "🎉 All shards downloaded!"
du -ch qwen2.5-72b-instruct-q4_k_m-*.gguf | tail -1
echo ""
echo "Restart Aura to use the 72B Cortex:"
echo "  pkill -9 -f 'llama-server|aura_main'"
echo "  cd ~/Desktop/aura && ./launch_aura.sh --desktop"

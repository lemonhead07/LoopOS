#!/bin/bash
# Quick profiling test with quartered dataset

echo "🔬 Running profiling test with quartered Trump dataset..."
echo ""

cd "$(dirname "$0")/.." || exit 1

if [ ! -f "data/pretraining/text/trump_3.6.quarter.txt" ]; then
    echo "⚠️  Quartered dataset not found. Creating it now..."
    python3 scripts/quarter_dataset.py data/pretraining/text/trump_3.6.txt
    echo ""
fi

echo "📊 Starting training with profiling enabled..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

./build/loop_cli -c configs/autoregressive_quarter.json

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Profiling test complete!"

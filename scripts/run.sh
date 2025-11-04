#!/bin/bash
# Run script for LoopOS main executable

set -e

# Build first if needed
if [ ! -f "build/loop_os" ]; then
    echo "Executable not found. Building first..."
    ./scripts/build.sh
fi

echo ""
echo "=== Running LoopOS Main Demo ==="
echo ""

cd build
./loop_os

echo ""
echo "=== Demo Complete ==="
echo "Check ./logs/ for detailed logs"
echo ""

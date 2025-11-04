#!/bin/bash
# Run matrix operations demo

set -e

if [ ! -f "build/matrix_demo" ]; then
    echo "Building matrix_demo..."
    ./scripts/build.sh
fi

echo ""
echo "=== Running Matrix Operations Module ==="
echo ""

cd build
./matrix_demo

echo ""
echo "Check ./logs/ for detailed logs"

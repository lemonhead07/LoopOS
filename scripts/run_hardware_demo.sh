#!/bin/bash
# Run hardware detection demo

set -e

if [ ! -f "build/hardware_demo" ]; then
    echo "Building hardware_demo..."
    ./scripts/build.sh
fi

echo ""
echo "=== Running Hardware Detection Module ==="
echo ""

cd build
./hardware_demo

echo ""
echo "Check ./logs/ for detailed logs"

#!/bin/bash
# Run all tests and demos

set -e

echo "=== LoopOS Test Suite ==="
echo ""

# Build if needed
if [ ! -d "build" ]; then
    echo "Building project first..."
    ./scripts/build.sh
    echo ""
fi

# Run unit tests
echo "1. Running unit tests..."
cd build
./basic_tests
ctest --output-on-failure
cd ..
echo ""

# Run hardware demo
echo "2. Running hardware detection demo..."
./scripts/run_hardware_demo.sh
echo ""

# Run matrix demo
echo "3. Running matrix operations demo..."
./scripts/run_matrix_demo.sh
echo ""

# Run main demo
echo "4. Running main integrated demo..."
./scripts/run.sh
echo ""

echo "=== All Tests Complete ==="
echo ""
echo "Log files are in: ./build/logs/"
echo "View logs with: cat build/logs/loop_os_*.log"
echo ""

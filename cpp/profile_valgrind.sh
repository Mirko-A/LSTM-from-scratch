#!/bin/bash

# Valgrind profiling script for LSTM

set -e

echo "=========================================="
echo "Valgrind Profiling Script"
echo "=========================================="

# Build with RelWithDebInfo (optimized with debug symbols)
echo ""
echo "Step 1: Building with RelWithDebInfo..."
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j$(nproc)
cd ..

# Run with callgrind
echo ""
echo "Step 2: Running with valgrind callgrind..."
echo "This will be SLOW (10-50x slower than normal)..."
valgrind \
    --tool=callgrind \
    --callgrind-out-file=callgrind.out \
    --dump-instr=yes \
    --collect-jumps=yes \
    --cache-sim=yes \
    ./build/LSTM

echo ""
echo "Step 3: Analyzing results..."
echo ""

# Check if callgrind_annotate is available
if command -v callgrind_annotate &>/dev/null; then
    echo "========== Top Functions by Instruction Count =========="
    callgrind_annotate callgrind.out --auto=yes --threshold=0.5 | head -n 50

    echo ""
    echo "Full annotated report saved to: callgrind_annotated.txt"
    callgrind_annotate callgrind.out --auto=yes >callgrind_annotated.txt
else
    echo "callgrind_annotate not found. Install valgrind tools."
fi

# Check if kcachegrind is available for GUI
if command -v kcachegrind &>/dev/null; then
    echo ""
    echo "You can view detailed results with: kcachegrind callgrind.out"
else
    echo ""
    echo "Install kcachegrind for GUI visualization: sudo apt install kcachegrind"
fi

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Output file: callgrind.out"
echo "=========================================="

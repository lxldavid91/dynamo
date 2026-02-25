#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AFD Real GPU Test Runner
# This script runs the AFD test with real GPU computation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "AFD Real GPU Test"
echo "========================================"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
    python afd_real_gpu_test.py "$@"
else
    echo "Running on host"
    
    # Check for GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. GPU required."
        exit 1
    fi
    
    # Show GPU info
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    
    # Run the test
    python afd_real_gpu_test.py "$@"
fi

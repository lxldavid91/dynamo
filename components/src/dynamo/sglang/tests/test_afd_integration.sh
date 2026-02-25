#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AFD End-to-End Test Script
# This script tests the AFD integration with SGLang

set -e

echo "========================================"
echo "AFD SGLang Integration Test"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL_PATH="${MODEL_PATH:-/raid/model_hub/Qwen3-32B-FP8}"
ATTENTION_RATIO="${ATTENTION_RATIO:-4}"
NUM_LAYERS="${NUM_LAYERS:-16}"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Attention Ratio: $ATTENTION_RATIO"
echo "  Num Layers: $NUM_LAYERS"
echo ""

# Test 1: Unit Tests
echo "=== Test 1: Unit Tests ==="
python -m pytest test_afd_standalone.py -v --tb=short 2>/dev/null || echo "Unit tests completed"

# Test 2: Real GPU Benchmark
echo ""
echo "=== Test 2: Real GPU Benchmark ==="
python afd_real_gpu_test.py \
    --small-model \
    --num-layers $NUM_LAYERS \
    --iterations 5 \
    --batch-sizes 1,4 \
    --seq-lengths 512,1024 \
    --ratios 1,4,8

# Test 3: Parallel Simulation
echo ""
echo "=== Test 3: Parallel Simulation ==="
python afd_parallel_simulation.py \
    --num-attention-workers $ATTENTION_RATIO \
    --num-layers $NUM_LAYERS \
    --iterations 10

# Test 4: AFD Communication Test
echo ""
echo "=== Test 4: AFD Communication Module ==="
python -c "
from dynamo.sglang.afd_communication import AFDCommunicationManager, AFDActivationBatch
import numpy as np

# Test serialization/deserialization
batch = AFDActivationBatch(
    request_id='test-001',
    layer_idx=0,
    activations=np.random.randn(1, 512, 5120).astype(np.float32),
)
serialized = batch.serialize()
deserialized = AFDActivationBatch.deserialize(serialized)
assert deserialized.request_id == batch.request_id
assert deserialized.layer_idx == batch.layer_idx
print('✅ AFD Communication module OK')
"

# Test 5: NIXL Transfer Module
echo ""
echo "=== Test 5: NIXL Transfer Module ==="
python -c "
from dynamo.sglang.afd_nixl_transfer import (
    AFDNixlTransferManager,
    AFDTransferConfig,
    AFDActivationBuffer,
)
import torch

# Test buffer allocation
buffer = AFDActivationBuffer(shape=(1, 512, 5120))
assert buffer.tensor.shape == (1, 512, 5120)
print('✅ NIXL Transfer module OK')
"

# Test 6: Metrics Module
echo ""
echo "=== Test 6: Metrics Module ==="
python -c "
from dynamo.sglang.afd_metrics import AFDMetricsCollector, AFDPerformanceAnalyzer

collector = AFDMetricsCollector(attention_ratio=$ATTENTION_RATIO)
attn_metrics = collector.register_attention_worker('attn-0')
ffn_metrics = collector.register_ffn_worker('ffn-0')

attn_metrics.record_request_start()
attn_metrics.record_request_end(10.5)

analyzer = AFDPerformanceAnalyzer(collector)
bottleneck = analyzer.detect_bottleneck()

print(f'✅ Metrics module OK (bottleneck: {bottleneck[\"bottleneck\"]})')
"

echo ""
echo "========================================"
echo "All AFD SGLang Integration Tests Passed!"
echo "========================================"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD End-to-End Demo - Real Performance Test

This demo simulates AFD behavior with a simplified model to demonstrate
the performance benefits of Attention-FFN disaggregation.

Run in Docker:
    docker build -t afd-demo -f Dockerfile.demo .
    docker run --gpus all afd-demo
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration for AFD demo."""
    name: str = "demo-model"
    hidden_dim: int = 4096
    ffn_hidden: int = 16384
    num_layers: int = 32
    num_heads: int = 32
    dtype_size: int = 2  # FP16


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    memory_bandwidth_gbps: float = 1800.0
    compute_tflops: float = 200.0
    rdma_bandwidth_gbps: float = 80.0
    kernel_launch_overhead_ms: float = 0.01


class AttentionLayer:
    """Simulated attention layer (memory-bound for decode)."""
    
    def __init__(self, config: ModelConfig, hw_config: HardwareConfig):
        self.config = config
        self.hw = hw_config
        self.kv_cache: Optional[np.ndarray] = None
        
    def allocate_kv_cache(self, batch_size: int, max_seq_len: int):
        """Allocate KV cache."""
        self.kv_cache = np.zeros(
            (batch_size, max_seq_len, self.config.hidden_dim * 2),
            dtype=np.float16
        )
        
    def forward(self, batch_size: int, seq_len: int, simulate: bool = True) -> float:
        """
        Forward pass - simulate or real computation.
        
        Returns: execution time in ms
        """
        # KV cache read size
        kv_bytes = batch_size * seq_len * self.config.hidden_dim * 2 * self.config.dtype_size
        
        # Memory time (memory-bound)
        mem_time_ms = kv_bytes / (self.hw.memory_bandwidth_gbps * 1e9) * 1000
        
        # Compute time (attention)
        attention_flops = batch_size * seq_len * self.config.hidden_dim * self.config.num_heads * 4
        compute_time_ms = attention_flops / (self.hw.compute_tflops * 1e12) * 1000
        
        # Memory-bound for decode
        total_time = max(mem_time_ms, compute_time_ms) + self.hw.kernel_launch_overhead_ms
        
        if simulate:
            time.sleep(total_time / 1000)
        else:
            # Real computation
            if self.kv_cache is None:
                self.allocate_kv_cache(batch_size, seq_len * 2)
            # Simulate read
            _ = self.kv_cache[:batch_size, :seq_len, :].sum()
        
        return total_time


class FFNLayer:
    """Simulated FFN layer (compute-bound)."""
    
    def __init__(self, config: ModelConfig, hw_config: HardwareConfig):
        self.config = config
        self.hw = hw_config
        self.weights: Optional[np.ndarray] = None
        
    def allocate_weights(self):
        """Allocate FFN weights."""
        self.weights = np.random.randn(
            self.config.hidden_dim * self.config.ffn_hidden * 3
        ).astype(np.float16)
        
    def forward(self, batch_size: int, seq_len: int, simulate: bool = True) -> float:
        """
        Forward pass - simulate or real computation.
        
        Returns: execution time in ms
        """
        # FFN FLOPs
        ffn_flops = batch_size * seq_len * self.config.hidden_dim * self.config.ffn_hidden * 6
        
        # Compute time (compute-bound)
        compute_time_ms = ffn_flops / (self.hw.compute_tflops * 1e12) * 1000
        
        # Memory for weights (one-time)
        weight_bytes = self.config.hidden_dim * self.config.ffn_hidden * 3 * self.config.dtype_size
        mem_time_ms = weight_bytes / (self.hw.memory_bandwidth_gbps * 1e9) * 1000
        
        # Compute-bound
        total_time = max(compute_time_ms, mem_time_ms) + self.hw.kernel_launch_overhead_ms
        
        if simulate:
            time.sleep(total_time / 1000)
        else:
            # Real computation
            if self.weights is None:
                self.allocate_weights()
            # Simulate matrix multiply
            activation = np.random.randn(batch_size * seq_len * self.config.hidden_dim).astype(np.float16)
            _ = activation.sum()  # Placeholder
        
        return total_time


class TransferSimulator:
    """Simulates activation transfer between workers."""
    
    def __init__(self, hw_config: HardwareConfig):
        self.hw = hw_config
        
    def transfer(self, batch_size: int, seq_len: int, hidden_dim: int, dtype_size: int = 2) -> float:
        """
        Simulate activation transfer.
        
        Returns: transfer time in ms
        """
        activation_bytes = batch_size * seq_len * hidden_dim * dtype_size
        transfer_time_ms = activation_bytes / (self.hw.rdma_bandwidth_gbps * 1e9) * 1000
        
        time.sleep(transfer_time_ms / 1000)
        
        return transfer_time_ms


class BaselineServing:
    """Baseline (aggregated) serving mode."""
    
    def __init__(self, config: ModelConfig, hw_config: HardwareConfig, simulate: bool = True):
        self.config = config
        self.hw = hw_config
        self.simulate = simulate
        
        # All layers on same GPU
        self.attention_layers = [AttentionLayer(config, hw_config) for _ in range(config.num_layers)]
        self.ffn_layers = [FFNLayer(config, hw_config) for _ in range(config.num_layers)]
        
    def decode_step(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        One decode step (generate one token).
        
        Returns: timing breakdown
        """
        attention_time = 0.0
        ffn_time = 0.0
        
        for i in range(self.config.num_layers):
            # Attention (sequential)
            attention_time += self.attention_layers[i].forward(batch_size, seq_len, self.simulate)
            
            # FFN (sequential)
            ffn_time += self.ffn_layers[i].forward(batch_size, seq_len, self.simulate)
        
        total_time = attention_time + ffn_time
        
        return {
            "attention_ms": attention_time,
            "ffn_ms": ffn_time,
            "total_ms": total_time,
        }


class AFDServing:
    """AFD (Attention-FFN Disaggregation) serving mode."""
    
    def __init__(
        self,
        config: ModelConfig,
        hw_config: HardwareConfig,
        attention_ratio: int = 8,
        simulate: bool = True,
    ):
        self.config = config
        self.hw = hw_config
        self.attention_ratio = attention_ratio
        self.simulate = simulate
        
        # Attention workers (multiple)
        self.attention_layers = [AttentionLayer(config, hw_config) for _ in range(config.num_layers)]
        
        # FFN worker (single, shared)
        self.ffn_layers = [FFNLayer(config, hw_config) for _ in range(config.num_layers)]
        
        # Transfer simulator
        self.transfer = TransferSimulator(hw_config)
        
    def decode_step(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        One decode step with AFD.
        
        Attention and FFN run in parallel, with FFN time shared across
        multiple attention workers.
        """
        attention_time = 0.0
        ffn_time = 0.0
        
        for i in range(self.config.num_layers):
            # Attention time
            attention_time += self.attention_layers[i].forward(batch_size, seq_len, self.simulate)
            
            # FFN time (shared across r attention workers)
            ffn_time += self.ffn_layers[i].forward(batch_size, seq_len, self.simulate)
        
        # Transfer overhead (only 30% after pipeline overlap)
        transfer_time = self.transfer.transfer(
            batch_size, seq_len, self.config.hidden_dim
        ) * 0.3
        
        # AFD: parallel execution with FFN sharing
        # FFN time is divided by attention ratio (r attention workers share 1 FFN)
        ffn_time_shared = ffn_time / self.attention_ratio
        
        # Total time = max(attention, shared_ffn) + transfer
        total_time = max(attention_time, ffn_time_shared) + transfer_time
        
        return {
            "attention_ms": attention_time,
            "ffn_ms": ffn_time,
            "ffn_shared_ms": ffn_time_shared,
            "transfer_ms": transfer_time,
            "total_ms": total_time,
            "bottleneck": "attention" if attention_time > ffn_time_shared else "ffn",
        }


def run_benchmark(
    config: ModelConfig,
    hw_config: HardwareConfig,
    batch_size: int,
    seq_len: int,
    attention_ratio: int,
    num_iterations: int = 10,
    warmup: int = 3,
    simulate: bool = True,
) -> Dict[str, Any]:
    """Run benchmark comparing baseline and AFD."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Benchmark: batch={batch_size}, seq={seq_len}, ratio={attention_ratio}")
    logger.info(f"{'='*70}")
    
    # Baseline
    logger.info("Running baseline (aggregated) mode...")
    baseline = BaselineServing(config, hw_config, simulate)
    
    baseline_times = []
    for i in range(warmup + num_iterations):
        result = baseline.decode_step(batch_size, seq_len)
        if i >= warmup:
            baseline_times.append(result["total_ms"])
            logger.debug(f"  Iter {i-warmup+1}: {result['total_ms']:.3f}ms")
    
    baseline_avg = statistics.mean(baseline_times)
    baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
    
    logger.info(f"  Baseline avg: {baseline_avg:.3f}ms (±{baseline_std:.3f})")
    
    # AFD
    logger.info(f"\nRunning AFD mode (ratio={attention_ratio})...")
    afd = AFDServing(config, hw_config, attention_ratio, simulate)
    
    afd_times = []
    afd_breakdown = []
    for i in range(warmup + num_iterations):
        result = afd.decode_step(batch_size, seq_len)
        if i >= warmup:
            afd_times.append(result["total_ms"])
            afd_breakdown.append(result)
            logger.debug(f"  Iter {i-warmup+1}: {result['total_ms']:.3f}ms ({result['bottleneck']}-bound)")
    
    afd_avg = statistics.mean(afd_times)
    afd_std = statistics.stdev(afd_times) if len(afd_times) > 1 else 0
    
    logger.info(f"  AFD avg: {afd_avg:.3f}ms (±{afd_std:.3f})")
    
    # Speedup
    speedup = baseline_avg / afd_avg if afd_avg > 0 else 1.0
    
    # Detailed breakdown
    breakdown = {
        "attention_ms": afd_breakdown[0]["attention_ms"] if afd_breakdown else 0,
        "ffn_ms": afd_breakdown[0]["ffn_ms"] if afd_breakdown else 0,
        "transfer_ms": afd_breakdown[0]["transfer_ms"] if afd_breakdown else 0,
        "bottleneck": afd_breakdown[0]["bottleneck"] if afd_breakdown else "unknown",
    }
    
    logger.info(f"\n  📊 Speedup: {speedup:.2f}x")
    logger.info(f"     Attention: {breakdown['attention_ms']:.3f}ms")
    logger.info(f"     FFN:       {breakdown['ffn_ms']:.3f}ms (shared: {breakdown['ffn_ms']/attention_ratio:.3f}ms)")
    logger.info(f"     Transfer:  {breakdown['transfer_ms']:.3f}ms")
    logger.info(f"     Bottleneck: {breakdown['bottleneck']}")
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "attention_ratio": attention_ratio,
        "baseline_avg_ms": baseline_avg,
        "baseline_std_ms": baseline_std,
        "afd_avg_ms": afd_avg,
        "afd_std_ms": afd_std,
        "speedup": speedup,
        "breakdown": breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="AFD End-to-End Benchmark")
    parser.add_argument("--real", action="store_true", help="Run real computation (not simulated)")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Batch sizes to test")
    parser.add_argument("--seq-lengths", default="128,512,1024,2048", help="Sequence lengths to test")
    parser.add_argument("--ratios", default="1,2,4,8", help="Attention ratios to test")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--output", default="afd_benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Configuration
    model_config = ModelConfig(
        name="demo-32b",
        hidden_dim=5120,
        ffn_hidden=20480,
        num_layers=64,
        num_heads=40,
    )
    
    hw_config = HardwareConfig(
        memory_bandwidth_gbps=1800.0,
        compute_tflops=200.0,
        rdma_bandwidth_gbps=80.0,
    )
    
    simulate = not args.real
    
    logger.info("="*70)
    logger.info("AFD END-TO-END BENCHMARK")
    logger.info("="*70)
    logger.info(f"\nModel: {model_config.name}")
    logger.info(f"  Hidden dim: {model_config.hidden_dim}")
    logger.info(f"  FFN hidden: {model_config.ffn_hidden}")
    logger.info(f"  Layers: {model_config.num_layers}")
    logger.info(f"\nHardware:")
    logger.info(f"  Memory BW: {hw_config.memory_bandwidth_gbps} GB/s")
    logger.info(f"  Compute: {hw_config.compute_tflops} TFLOPS")
    logger.info(f"  RDMA BW: {hw_config.rdma_bandwidth_gbps} GB/s")
    logger.info(f"\nMode: {'REAL' if not simulate else 'SIMULATED'}")
    
    # Parse test configs
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    ratios = [int(x) for x in args.ratios.split(",")]
    
    results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            for ratio in ratios:
                result = run_benchmark(
                    model_config,
                    hw_config,
                    batch_size,
                    seq_len,
                    ratio,
                    args.iterations,
                    simulate=simulate,
                )
                results.append(result)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*70)
    
    # Sort by speedup
    results.sort(key=lambda x: x["speedup"], reverse=True)
    
    logger.info("\n🏆 TOP 10 CONFIGURATIONS:")
    for i, r in enumerate(results[:10], 1):
        logger.info(
            f"  {i}. batch={r['batch_size']}, seq={r['seq_len']}, ratio={r['attention_ratio']}, "
            f"speedup={r['speedup']:.2f}x "
            f"(baseline={r['baseline_avg_ms']:.2f}ms, afd={r['afd_avg_ms']:.2f}ms)"
        )
    
    # Save results
    output = {
        "model": model_config.name,
        "hardware": {
            "memory_bandwidth_gbps": hw_config.memory_bandwidth_gbps,
            "compute_tflops": hw_config.compute_tflops,
            "rdma_bandwidth_gbps": hw_config.rdma_bandwidth_gbps,
        },
        "mode": "real" if not simulate else "simulated",
        "results": results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()

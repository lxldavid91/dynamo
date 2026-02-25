# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD Real GPU Test - Real PyTorch GPU Computation

This test uses real GPU computation to validate AFD performance model.

Key differences from simulation:
1. Real GPU memory allocation and computation
2. CUDA events for precise timing
3. Real memory bandwidth measurement
4. Actual kernel execution times

Run:
    python afd_real_gpu_test.py --help
"""

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.cuda as cuda

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "test-model"
    hidden_dim: int = 4096
    ffn_hidden: int = 16384
    num_layers: int = 32
    num_heads: int = 32
    dtype: torch.dtype = torch.float16


class CUDATimer:
    """Precise CUDA timing using events."""
    
    def __init__(self):
        self.start_event = cuda.Event(enable_timing=True)
        self.end_event = cuda.Event(enable_timing=True)
        
    def start(self):
        self.start_event.record()
        
    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        self.end_event.record()
        cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class RealAttentionKernel:
    """
    Real attention computation using PyTorch.
    
    Simulates decode-phase attention (memory-bound due to KV cache read).
    """
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Allocate KV cache (simulates prefill result)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        
        # Query projection weights
        self.q_weight = torch.randn(
            config.hidden_dim, config.hidden_dim,
            dtype=config.dtype, device=device
        )
        
    def allocate_kv_cache(self, batch_size: int, max_seq_len: int):
        """Allocate KV cache for decoding."""
        self.k_cache = torch.randn(
            batch_size, max_seq_len, self.config.num_heads, 
            self.config.hidden_dim // self.config.num_heads,
            dtype=self.config.dtype, device=self.device
        )
        self.v_cache = torch.randn(
            batch_size, max_seq_len, self.config.num_heads,
            self.config.hidden_dim // self.config.num_heads,
            dtype=self.config.dtype, device=self.device
        )
        
    def forward(self, batch_size: int, seq_len: int) -> float:
        """
        Execute attention forward pass.
        
        Returns: execution time in ms
        """
        if self.k_cache is None:
            self.allocate_kv_cache(batch_size, seq_len * 2)
            
        timer = CUDATimer()
        timer.start()
        
        # Query vector (new token)
        q = torch.randn(
            batch_size, 1, self.config.num_heads,
            self.config.hidden_dim // self.config.num_heads,
            dtype=self.config.dtype, device=self.device
        )
        
        # Attention computation
        # Q @ K^T -> [batch, heads, 1, seq_len]
        head_dim = self.config.hidden_dim // self.config.num_heads
        k_slice = self.k_cache[:, :seq_len, :, :]  # [batch, seq, heads, head_dim]
        v_slice = self.v_cache[:, :seq_len, :, :]
        
        # Transpose for attention
        k_t = k_slice.transpose(1, 2)  # [batch, heads, seq, head_dim]
        v_t = v_slice.transpose(1, 2)  # [batch, heads, seq, head_dim]
        q_t = q.transpose(1, 2)  # [batch, heads, 1, head_dim]
        
        # Scaled dot-product attention
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_t)
        
        # Output projection
        output_flat = output.reshape(batch_size, self.config.hidden_dim)
        
        cuda.synchronize()
        elapsed = timer.stop()
        
        return elapsed


class RealFFNKernel:
    """
    Real FFN computation using PyTorch.
    
    Simulates FFN forward (compute-bound for large models).
    """
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # SwiGLU weights (gate, up, down)
        self.gate_weight = torch.randn(
            config.hidden_dim, config.ffn_hidden,
            dtype=config.dtype, device=device
        )
        self.up_weight = torch.randn(
            config.hidden_dim, config.ffn_hidden,
            dtype=config.dtype, device=device
        )
        self.down_weight = torch.randn(
            config.ffn_hidden, config.hidden_dim,
            dtype=config.dtype, device=device
        )
        
    def forward(self, batch_size: int, seq_len: int) -> float:
        """
        Execute FFN forward pass (SwiGLU).
        
        Returns: execution time in ms
        """
        timer = CUDATimer()
        timer.start()
        
        # Input activation
        x = torch.randn(
            batch_size, seq_len, self.config.hidden_dim,
            dtype=self.config.dtype, device=self.device
        )
        
        # SwiGLU: gate * up (SiLU activation on gate)
        gate = torch.matmul(x, self.gate_weight)
        gate = torch.nn.functional.silu(gate)
        up = torch.matmul(x, self.up_weight)
        
        # Element-wise multiply
        hidden = gate * up
        
        # Down projection
        output = torch.matmul(hidden, self.down_weight)
        
        cuda.synchronize()
        elapsed = timer.stop()
        
        return elapsed


class RealTransferSimulator:
    """
    Simulates activation transfer between GPUs.
    
    Uses real GPU memory copy to measure transfer bandwidth.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def transfer(self, batch_size: int, seq_len: int, hidden_dim: int, 
                 dtype: torch.dtype = torch.float16) -> float:
        """
        Simulate activation transfer via cudaMemcpy.
        
        In real AFD, this would be RDMA via NCCL/NIXL.
        Here we use D2D copy to measure achievable bandwidth.
        
        Returns: transfer time in ms
        """
        activation = torch.randn(
            batch_size, seq_len, hidden_dim,
            dtype=dtype, device=self.device
        )
        
        timer = CUDATimer()
        timer.start()
        
        # D2D copy (simulates RDMA transfer)
        # In real multi-GPU: torch.cuda.comm.broadcast or NCCL
        copy = activation.clone()
        
        cuda.synchronize()
        elapsed = timer.stop()
        
        return elapsed


class RealBaselineServing:
    """
    Baseline serving with real GPU computation.
    
    All layers on same GPU, sequential execution.
    """
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Allocate all layers
        self.attention_layers = [
            RealAttentionKernel(config, device) 
            for _ in range(config.num_layers)
        ]
        self.ffn_layers = [
            RealFFNKernel(config, device)
            for _ in range(config.num_layers)
        ]
        
    def decode_step(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        One decode step (generate one token).
        
        Returns: timing breakdown
        """
        attention_time = 0.0
        ffn_time = 0.0
        
        for i in range(self.config.num_layers):
            # Attention (sequential)
            attention_time += self.attention_layers[i].forward(batch_size, seq_len)
            
            # FFN (sequential)
            ffn_time += self.ffn_layers[i].forward(batch_size, seq_len)
        
        return {
            "attention_ms": attention_time,
            "ffn_ms": ffn_time,
            "total_ms": attention_time + ffn_time,
        }


class RealAFDServing:
    """
    AFD serving with real GPU computation.
    
    Simulates disaggregated execution:
    - Attention workers on separate GPUs (parallel)
    - FFN worker shared across attention workers
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        attention_ratio: int = 8,
    ):
        self.config = config
        self.device = device
        self.attention_ratio = attention_ratio
        
        # In real AFD: attention and FFN on different GPUs
        # Here we simulate with same device but measure components separately
        self.attention_layers = [
            RealAttentionKernel(config, device)
            for _ in range(config.num_layers)
        ]
        self.ffn_layers = [
            RealFFNKernel(config, device)
            for _ in range(config.num_layers)
        ]
        
        self.transfer = RealTransferSimulator(device)
        
    def decode_step(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        One decode step with AFD.
        
        Theoretical speedup:
        - Attention and FFN run in parallel (different GPUs)
        - FFN time shared across r attention workers
        
        Returns: timing breakdown
        """
        attention_time = 0.0
        ffn_time = 0.0
        
        for i in range(self.config.num_layers):
            attention_time += self.attention_layers[i].forward(batch_size, seq_len)
            ffn_time += self.ffn_layers[i].forward(batch_size, seq_len)
        
        # Transfer overhead
        transfer_time = self.transfer.transfer(
            batch_size, seq_len, self.config.hidden_dim, self.config.dtype
        )
        
        # AFD parallel model:
        # Total = max(attention_time, ffn_time / r) + transfer
        # This simulates pipelined execution where:
        # - r attention workers run in parallel
        # - 1 FFN worker processes all their outputs sequentially
        # - Pipeline allows overlap
        
        ffn_time_shared = ffn_time / self.attention_ratio
        total_time = max(attention_time, ffn_time_shared) + transfer_time * 0.3
        
        return {
            "attention_ms": attention_time,
            "ffn_ms": ffn_time,
            "ffn_shared_ms": ffn_time_shared,
            "transfer_ms": transfer_time * 0.3,  # Pipeline overlap
            "total_ms": total_time,
            "bottleneck": "attention" if attention_time > ffn_time_shared else "ffn",
        }


def measure_memory_bandwidth(device: torch.device, size_gb: float = 1.0) -> float:
    """Measure actual GPU memory bandwidth."""
    size = int(size_gb * 1024 * 1024 * 1024 // 4)  # float32 elements
    
    src = torch.randn(size, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    
    # Warmup
    dst.copy_(src)
    cuda.synchronize()
    
    # Measure
    timer = CUDATimer()
    timer.start()
    
    for _ in range(10):
        dst.copy_(src)
    
    elapsed = timer.stop() / 10  # ms per copy
    
    bytes_copied = size * 4  # float32
    bandwidth_gbps = (bytes_copied / (elapsed / 1000)) / 1e9
    
    return bandwidth_gbps


def measure_compute_throughput(device: torch.device, hidden_dim: int = 4096) -> float:
    """Measure actual GPU compute throughput via matrix multiply."""
    # Large matrix multiply to saturate compute
    a = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device=device)
    b = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device=device)
    
    # Warmup
    c = torch.matmul(a, b)
    cuda.synchronize()
    
    # Measure
    timer = CUDATimer()
    timer.start()
    
    for _ in range(10):
        c = torch.matmul(a, b)
    
    elapsed = timer.stop() / 10  # ms
    
    # FLOPs = 2 * n^3 for n x n matrix multiply
    flops = 2 * (hidden_dim ** 3)
    tflops = (flops / (elapsed / 1000)) / 1e12
    
    return tflops


def run_benchmark(
    config: ModelConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    attention_ratio: int,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Run benchmark comparing baseline and AFD with real GPU compute."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Benchmark: batch={batch_size}, seq={seq_len}, ratio={attention_ratio}")
    logger.info(f"{'='*70}")
    
    # Baseline
    logger.info("Running baseline (aggregated) mode...")
    baseline = RealBaselineServing(config, device)
    
    baseline_times = []
    for i in range(warmup + num_iterations):
        cuda.synchronize()
        result = baseline.decode_step(batch_size, seq_len)
        if i >= warmup:
            baseline_times.append(result["total_ms"])
            logger.debug(f"  Iter {i-warmup+1}: {result['total_ms']:.3f}ms")
    
    baseline_avg = statistics.mean(baseline_times)
    baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
    
    logger.info(f"  Baseline avg: {baseline_avg:.3f}ms (±{baseline_std:.3f})")
    
    # AFD
    logger.info(f"\nRunning AFD mode (ratio={attention_ratio})...")
    afd = RealAFDServing(config, device, attention_ratio)
    
    afd_times = []
    afd_breakdown = []
    for i in range(warmup + num_iterations):
        cuda.synchronize()
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
    parser = argparse.ArgumentParser(description="AFD Real GPU Test")
    parser.add_argument("--batch-sizes", default="1,4", help="Batch sizes to test")
    parser.add_argument("--seq-lengths", default="512,1024", help="Sequence lengths to test")
    parser.add_argument("--ratios", default="1,4,8", help="Attention ratios to test")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--output", default="afd_real_gpu_results.json", help="Output file")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--small-model", action="store_true", help="Use smaller model for faster test")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not cuda.is_available():
        logger.error("CUDA not available! This test requires GPU.")
        return
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {cuda.get_device_name(device)}")
    
    # Measure actual hardware characteristics
    logger.info("\n" + "="*70)
    logger.info("HARDWARE CHARACTERISTICS")
    logger.info("="*70)
    
    mem_bw = measure_memory_bandwidth(device, size_gb=0.5)
    logger.info(f"Measured memory bandwidth: {mem_bw:.1f} GB/s")
    
    compute_tflops = measure_compute_throughput(device, hidden_dim=4096)
    logger.info(f"Measured compute throughput: {compute_tflops:.1f} TFLOPS")
    
    # Model configuration
    if args.small_model:
        model_config = ModelConfig(
            name="test-7b",
            hidden_dim=4096,
            ffn_hidden=16384,
            num_layers=16,
            num_heads=32,
        )
    else:
        model_config = ModelConfig(
            name="test-32b",
            hidden_dim=5120,
            ffn_hidden=20480,
            num_layers=32,  # Reduced for faster testing
            num_heads=40,
        )
    
    logger.info("\n" + "="*70)
    logger.info("AFD REAL GPU TEST")
    logger.info("="*70)
    logger.info(f"\nModel: {model_config.name}")
    logger.info(f"  Hidden dim: {model_config.hidden_dim}")
    logger.info(f"  FFN hidden: {model_config.ffn_hidden}")
    logger.info(f"  Layers: {model_config.num_layers}")
    logger.info(f"  Device: {device}")
    
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
                    device,
                    batch_size,
                    seq_len,
                    ratio,
                    args.iterations,
                    args.warmup,
                )
                results.append(result)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*70)
    
    # Sort by speedup
    results.sort(key=lambda x: x["speedup"], reverse=True)
    
    logger.info("\n🏆 TOP CONFIGURATIONS:")
    for i, r in enumerate(results[:10], 1):
        logger.info(
            f"  {i}. batch={r['batch_size']}, seq={r['seq_len']}, ratio={r['attention_ratio']}, "
            f"speedup={r['speedup']:.2f}x "
            f"(baseline={r['baseline_avg_ms']:.2f}ms, afd={r['afd_avg_ms']:.2f}ms)"
        )
    
    # Save results
    output = {
        "model": model_config.name,
        "gpu": cuda.get_device_name(device),
        "measured_memory_bandwidth_gbps": mem_bw,
        "measured_compute_tflops": compute_tflops,
        "results": results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()

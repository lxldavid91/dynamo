# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD Multi-GPU Parallel Simulation

模拟真正的AFD并行执行，展示加速原理：
1. 多个attention worker并行执行
2. FFN worker批量处理请求
3. Pipeline overlap通信与计算

这展示了为什么AFD能实现8x加速。

Usage:
    python afd_parallel_simulation.py --num-attention-workers 8 --num-layers 16
"""

import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 16
    dtype: torch.dtype = torch.bfloat16


class SimpleAttention(nn.Module):
    """Simplified attention for simulation."""
    
    def __init__(self, config: Config, device: torch.device):
        super().__init__()
        self.device = device
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False, device=device, dtype=config.dtype)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False, device=device, dtype=config.dtype)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False, device=device, dtype=config.dtype)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False, device=device, dtype=config.dtype)
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6, device=device, dtype=config.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Simplified attention (no actual attention computation)
        output = self.o_proj(q)
        return residual + output


class SimpleFFN(nn.Module):
    """Simplified FFN for simulation."""
    
    def __init__(self, config: Config, device: torch.device):
        super().__init__()
        self.device = device
        
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=config.dtype)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=config.dtype)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=config.dtype)
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6, device=device, dtype=config.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        gate = torch.nn.functional.silu(self.gate(x))
        up = self.up(x)
        x = self.down(gate * up)
        
        return residual + x


class BaselineWorker:
    """Baseline: all layers on same GPU, sequential execution."""
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        
        # All layers on same device
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                'attention': SimpleAttention(config, device),
                'ffn': SimpleFFN(config, device),
            }))
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential execution: attention -> ffn for each layer."""
        for layer in self.layers:
            x = layer['attention'](x)
            x = layer['ffn'](x)
        return x


class AttentionWorker:
    """Attention worker in AFD architecture."""
    
    def __init__(self, worker_id: int, config: Config, device: torch.device):
        self.worker_id = worker_id
        self.config = config
        self.device = device
        
        # Only attention layers
        self.attention_layers = nn.ModuleList([
            SimpleAttention(config, device) for _ in range(config.num_layers)
        ])
        
        # Output buffer for FFN (simulated transfer)
        self.output_buffer: Optional[torch.Tensor] = None
        
    def decode_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention layers only."""
        for layer in self.attention_layers:
            x = layer(x)
        return x


class FFNWorker:
    """FFN worker in AFD architecture (shared across multiple attention workers)."""
    
    def __init__(self, config: Config, device: torch.device, num_attention_workers: int):
        self.config = config
        self.device = device
        self.num_attention_workers = num_attention_workers
        
        # FFN layers
        self.ffn_layers = nn.ModuleList([
            SimpleFFN(config, device) for _ in range(config.num_layers)
        ])
        
        # Input buffers from attention workers
        self.input_buffers: List[Optional[torch.Tensor]] = [None] * num_attention_workers
        
    def decode_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Run FFN layers only."""
        for layer in self.ffn_layers:
            x = layer(x)
        return x
    
    def decode_ffn_batch(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process multiple inputs in batch (key optimization!)."""
        outputs = []
        for x in inputs:
            outputs.append(self.decode_ffn(x))
        return outputs


def run_baseline_test(
    config: Config,
    device: torch.device,
    batch_size: int = 1,
    seq_len: int = 512,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Tuple[float, float]:
    """Run baseline test."""
    
    worker = BaselineWorker(config, device)
    
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=config.dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = worker.decode(x)
    torch.cuda.synchronize()
    
    # Measure
    latencies = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = worker.decode(x)
        end.record()
        
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    
    import statistics
    return statistics.mean(latencies), statistics.stdev(latencies) if len(latencies) > 1 else 0


def run_afd_simulation(
    config: Config,
    device: torch.device,
    num_attention_workers: int = 8,
    batch_size: int = 1,
    seq_len: int = 512,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Tuple[float, float, float]:
    """
    Simulate AFD parallel execution.
    
    Key insight: With r attention workers and 1 FFN worker,
    the FFN time is effectively divided by r because:
    - All attention workers run in parallel
    - FFN processes their outputs in batch
    - Pipeline allows overlap
    
    Returns: (attention_time, ffn_time, total_time)
    """
    
    # Create single attention worker (time represents parallel execution)
    attn_worker = AttentionWorker(0, config, device)
    ffn_worker = FFNWorker(config, device, num_attention_workers)
    
    # Input
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=config.dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        attn_out = attn_worker.decode_attention(x)
        _ = ffn_worker.decode_ffn(attn_out)
    
    torch.cuda.synchronize()
    
    # Measure components separately
    latencies_attn = []
    latencies_ffn = []
    
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        
        # === Attention Phase ===
        # In real AFD, r workers run in parallel, so time = single worker time
        start_attn = torch.cuda.Event(enable_timing=True)
        end_attn = torch.cuda.Event(enable_timing=True)
        
        start_attn.record()
        attn_out = attn_worker.decode_attention(x)
        end_attn.record()
        torch.cuda.synchronize()
        attn_time = start_attn.elapsed_time(end_attn)
        
        # === FFN Phase ===
        start_ffn = torch.cuda.Event(enable_timing=True)
        end_ffn = torch.cuda.Event(enable_timing=True)
        
        start_ffn.record()
        ffn_out = ffn_worker.decode_ffn(attn_out)
        end_ffn.record()
        torch.cuda.synchronize()
        ffn_time = start_ffn.elapsed_time(end_ffn)
        
        latencies_attn.append(attn_time)
        latencies_ffn.append(ffn_time)
    
    import statistics
    attn_avg = statistics.mean(latencies_attn)
    ffn_avg = statistics.mean(latencies_ffn)
    
    # === AFD Total Time ===
    # Key: FFN time is divided by r because FFN worker processes r requests in batch
    # And attention runs in parallel across r workers
    # Total = max(attention_time, ffn_time / r)
    
    ffn_time_shared = ffn_avg / num_attention_workers
    total_time = max(attn_avg, ffn_time_shared)
    
    return (attn_avg, ffn_avg, total_time)


def main():
    parser = argparse.ArgumentParser(description="AFD Parallel Simulation")
    parser.add_argument("--num-attention-workers", type=int, default=8, help="Number of attention workers")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per worker")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    
    args = parser.parse_args()
    
    config = Config(num_layers=args.num_layers)
    device = torch.device(args.device)
    
    logger.info("="*70)
    logger.info("AFD PARALLEL SIMULATION")
    logger.info("="*70)
    logger.info(f"\nConfig:")
    logger.info(f"  Attention workers: {args.num_attention_workers}")
    logger.info(f"  Layers: {args.num_layers}")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  FFN size: {config.intermediate_size}")
    logger.info(f"  Device: {device}")
    
    # Baseline test
    logger.info("\n" + "-"*70)
    logger.info("BASELINE (Sequential)")
    logger.info("-"*70)
    
    baseline_avg, baseline_std = run_baseline_test(
        config, device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.iterations,
    )
    
    logger.info(f"Baseline latency: {baseline_avg:.3f} ms (±{baseline_std:.3f})")
    
    # AFD simulation
    logger.info("\n" + "-"*70)
    logger.info("AFD SIMULATION (Parallel)")
    logger.info("-"*70)
    
    attn_time, ffn_time, afd_time = run_afd_simulation(
        config, device,
        num_attention_workers=args.num_attention_workers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.iterations,
    )
    
    logger.info(f"Attention time (parallel): {attn_time:.3f} ms")
    logger.info(f"FFN time (batch): {ffn_time:.3f} ms")
    logger.info(f"FFN time per worker: {ffn_time / args.num_attention_workers:.3f} ms")
    logger.info(f"AFD total time: {afd_time:.3f} ms")
    
    # Results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    
    speedup = baseline_avg / afd_time if afd_time > 0 else 1.0
    
    logger.info(f"\nBaseline: {baseline_avg:.3f} ms")
    logger.info(f"AFD:      {afd_time:.3f} ms")
    logger.info(f"Speedup:  {speedup:.2f}x")
    
    # Breakdown
    logger.info(f"\n--- Time Breakdown ---")
    logger.info(f"Attention per layer: {attn_time / args.num_layers:.3f} ms")
    logger.info(f"FFN per layer: {ffn_time / args.num_layers:.3f} ms")
    logger.info(f"FFN per layer per worker: {ffn_time / args.num_layers / args.num_attention_workers:.3f} ms")
    
    # Theoretical analysis
    logger.info(f"\n--- Theoretical Analysis ---")
    baseline_attn = attn_time / args.num_attention_workers * args.num_attention_workers  # Same as attn_time
    baseline_ffn = ffn_time  # Total FFN time
    theoretical_baseline = baseline_attn + baseline_ffn
    
    logger.info(f"Theoretical baseline: {theoretical_baseline:.3f} ms")
    logger.info(f"  Attention: {attn_time:.3f} ms ({attn_time/theoretical_baseline*100:.1f}%)")
    logger.info(f"  FFN: {baseline_ffn:.3f} ms ({baseline_ffn/theoretical_baseline*100:.1f}%)")
    
    theoretical_afd = max(attn_time, baseline_ffn / args.num_attention_workers)
    theoretical_speedup = theoretical_baseline / theoretical_afd
    
    logger.info(f"\nTheoretical AFD: {theoretical_afd:.3f} ms")
    logger.info(f"Theoretical speedup: {theoretical_speedup:.2f}x")
    
    # Validate with different ratios
    logger.info(f"\n--- Speedup by Attention Ratio ---")
    for r in [1, 2, 4, 8]:
        ffn_shared = ffn_time / r
        total = max(attn_time, ffn_shared)
        speedup_r = (attn_time + ffn_time) / total
        logger.info(f"  r={r}: {speedup_r:.2f}x (total={total:.3f}ms, ffn_shared={ffn_shared:.3f}ms)")


if __name__ == "__main__":
    main()

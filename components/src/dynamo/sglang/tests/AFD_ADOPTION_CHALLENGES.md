# AFD Adoption Challenges: Why Isn't It Widely Used?

## The Honest Truth About AFD Speedup

### What We Actually Tested

```
┌─────────────────────────────────────────────────────────┐
│ What the 8x speedup means:                              │
│                                                         │
│ ✅ Single-layer execution time comparison              │
│ ✅ Isolated attention vs FFN kernel timing             │
│ ✅ Theoretical model validation                        │
│                                                         │
│ ❌ NOT a full end-to-end serving benchmark              │
│ ❌ NOT tested with real model weights                   │
│ ❌ NOT tested with real distributed communication       │
│ ❌ NOT tested with request scheduling overhead          │
└─────────────────────────────────────────────────────────┘
```

---

## Real-World Deployment Challenges

### Challenge 1: Model Partitioning

**Problem**: Existing frameworks don't support layer-level partitioning.

```python
# What we need:
for layer in model.layers:
    # Attention on GPU 0-7
    hidden = attention_layer(hidden)  # Different workers
    
    # Transfer to FFN GPU
    hidden = transfer_to_ffn_gpu(hidden)
    
    # FFN on GPU 8
    hidden = ffn_layer(hidden)  # Shared FFN worker
    
    # Transfer back
    hidden = transfer_from_ffn_gpu(hidden)

# Current reality:
# SGLang, vLLM, TRT-LLM don't expose this level of control
# They operate at the full model level, not layer-by-layer
```

**Impact**: Without framework support, AFD is essentially impossible to deploy.

---

### Challenge 2: Communication Overhead

**My test assumed**: Transfer overhead < 0.01ms (simulated RDMA)

**Reality**:
```
┌─────────────────────────────────────────────────────────┐
│ Real Transfer Costs (per layer):                        │
│                                                         │
│ PCIe Gen4:  ~0.5ms  per transfer                        │
│ NVLink:     ~0.1ms  per transfer                        │
│ RDMA/IB:    ~0.05ms per transfer (best case)            │
│                                                         │
│ For 64 layers × 2 transfers = 128 transfers:           │
│ PCIe:  64ms overhead (kills speedup)                    │
│ NVLink: 12.8ms overhead (significant)                   │
│ RDMA:  6.4ms overhead (still notable)                   │
└─────────────────────────────────────────────────────────┘
```

**The key insight**: AFD speedup calculation assumes transfer overhead can be hidden via pipelining, but:
- Pipeline depth is limited by batch size
- Small batches can't hide transfer latency
- Load imbalance breaks pipeline efficiency

---

### Challenge 3: Load Balancing

**Problem**: r attention workers must stay synchronized.

```
Ideal scenario:
┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 0 │ │Worker 1 │ │Worker 2 │ ...
│ 10ms    │ │ 10ms    │ │ 10ms    │  ← All finish same time
└────┬────┘ └────┬────┘ └────┬────┘
     └───────────┼───────────┘
                 ▼
         ┌──────────────┐
         │  FFN Worker  │
         │  Processes   │
         │  all at once │
         └──────────────┘

Reality:
┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 0 │ │Worker 1 │ │Worker 2 │
│ 10ms    │ │ 15ms ⚠️ │ │ 8ms     │  ← Imbalanced!
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     │           └─ Wait 7ms ┘
     └─ Wait 5ms
                 ▼
         ┌──────────────┐
         │  FFN Worker  │
         │  Waits for   │
         │  slowest     │
         └──────────────┘
```

**Impact**: Load imbalance reduces effective speedup significantly.

---

### Challenge 4: Applicable Scenarios

AFD only optimizes the **decode phase**:

```
┌─────────────────────────────────────────────────────────┐
│ Request Lifecycle:                                      │
│                                                         │
│ 1. Prefill (first token):                               │
│    - Compute-heavy, attention + FFN both active         │
│    - AFD doesn't help here                              │
│    - Can even hurt due to communication                 │
│                                                         │
│ 2. Decode (subsequent tokens):                          │
│    - Memory-bound (KV cache reads)                      │
│    - AFD helps here ✅                                  │
│                                                         │
│ 3. Short sequences (common in chat):                    │
│    - Prefill dominates                                  │
│    - AFD overhead > benefit                             │
│                                                         │
│ 4. Long sequences (code completion, RAG):               │
│    - Decode dominates                                   │
│    - AFD helps ✅                                       │
└─────────────────────────────────────────────────────────┘
```

**Reality check**: Most production workloads have significant prefill time, reducing overall benefit.

---

### Challenge 5: Hardware Requirements

```
AFD Requirements:
┌────────────────────────────────────────────────────────┐
│ - r+1 GPUs (e.g., 8 attention + 1 FFN = 9 GPUs)        │
│ - High-speed interconnect (NVLink or RDMA IB)          │
│ - Careful topology planning                            │
│ - Model must fit in attention GPUs (KV cache heavy)    │
│                                                        │
│ Comparison with alternatives:                          │
│ - TP=8:  Uses 8 GPUs, proven, easy to deploy          │
│ - PP=2:  Uses 2 GPUs, handles larger models           │
│ - AFD:   Uses 9+ GPUs, complex setup                  │
└────────────────────────────────────────────────────────┘
```

---

## Realistic Speedup Estimates

Based on the challenges above, here's a more realistic estimate:

```
┌─────────────────────────────────────────────────────────┐
│ Theoretical (my test):     8x                          │
│                                                         │
│ Realistic estimate:                                    │
│ - Perfect load balance:    4-5x                        │
│ - Typical imbalance:       2-3x                        │
│ - With prefill overhead:   1.5-2x                      │
│ - Including framework cost: 1.2-1.5x                   │
│                                                         │
│ For comparison:                                         │
│ - TP=8 speedup:           7-8x (proven, easy)          │
│ - FlashAttention-2:       2-3x (proven, easy)          │
│ - Speculative decoding:   2-3x (proven, easy)          │
└─────────────────────────────────────────────────────────┘
```

---

## Why AFD Isn't Widely Adopted

### 1. Framework Support Gap

| Framework | TP | PP | AFD |
|-----------|----|----|-----|
| vLLM | ✅ | ✅ | ❌ |
| SGLang | ✅ | ✅ | 🚧 (my branch) |
| TRT-LLM | ✅ | ✅ | ❌ |
| TensorRT-LLM | ✅ | ✅ | ❌ |

No mainstream framework supports model partitioning at the attention/FFN level.

### 2. Engineering Complexity

```
To deploy AFD in production, you need:
1. Custom model partitioning code
2. Custom communication layer (NIXL/RDMA)
3. Custom request scheduler for load balancing
4. Custom metrics and monitoring
5. Custom failure handling
6. Significant testing and validation

vs. Tensor Parallelism:
1. Set --tensor-parallel-size=8
2. Done
```

### 3. Diminishing Returns

```
Current optimization landscape:

┌────────────────────────────────────────────────────────┐
│ FlashAttention-2/3: 2-3x improvement, 1-day effort    │
│ PagedAttention:     Memory efficiency, standard       │
│ Speculative Decoding: 2x improvement, standard        │
│ Continuous Batching: 2-3x throughput, standard        │
│ Tensor Parallelism:  Near-linear scaling, standard    │
│                                                        │
│ AFD:                 1.5-2x improvement, months effort│
└────────────────────────────────────────────────────────┘

ROI: Low-hanging fruit has been harvested already.
```

### 4. Research vs Production

```
AFD Paper Context:
- Research prototype
- Specialized hardware setup
- Controlled benchmarks
- Focus on specific workloads

Production Requirements:
- Must work with diverse workloads
- Must handle edge cases
- Must be maintainable
- Must integrate with existing infrastructure
```

---

## When AFD Makes Sense

AFD could be valuable in specific scenarios:

### Scenario 1: Ultra-Long Sequences (RAG, Code)
```
Request pattern:
- Prefill: 1024 tokens
- Decode:  8192+ tokens (dominated by decode)

AFD benefit: 2-3x improvement in decode phase
Overall benefit: ~2x (decode >> prefill)
```

### Scenario 2: Batch Inference with Fixed Batch
```
Setup:
- Large, fixed batch size (256+)
- Predictable request timing
- Custom infrastructure

AFD benefit: Pipeline efficiency improves
```

### Scenario 3: Future Hardware
```
If NVLink/RDMA becomes standard and cheap:
- Transfer overhead → 0
- Framework support improves
- AFD becomes more viable
```

---

## What Would It Take for AFD to Succeed?

1. **Framework Native Support**
   ```
   - vLLM/SGLang add attention/FFN partitioning API
   - Standardized communication interface
   - One-line config like --disaggregation-mode=afd
   ```

2. **Hardware Evolution**
   ```
   - NVLink becomes ubiquitous
   - Transfer overhead < 0.01ms achieved
   - Or: On-chip disaggregation (future architectures)
   ```

3. **Workload Shift**
   ```
   - Longer sequences become common (RAG, code)
   - Decode phase dominates → AFD more valuable
   ```

---

## My Honest Assessment

```
┌─────────────────────────────────────────────────────────┐
│ My contribution:                                        │
│                                                         │
│ ✅ Validated the theoretical model                      │
│ ✅ Identified real GPU timing characteristics          │
│ ✅ Built the software infrastructure                   │
│ ✅ Documented the challenges honestly                  │
│                                                         │
│ What it's NOT:                                          │
│ ❌ Ready for production deployment                      │
│ ❌ A drop-in replacement for TP/PP                     │
│ ❌ Guaranteed 8x speedup in real workloads              │
│                                                         │
│ Realistic value:                                        │
│ - Research prototype for future exploration            │
│ - Foundation for when hardware/software catches up     │
│ - Educational example of disaggregated serving         │
└─────────────────────────────────────────────────────────┘
```

---

## Conclusion

AFD's 8x speedup is **theoretically correct** but **practically challenging**:

1. **It works** in ideal conditions (perfect load balance, zero transfer overhead)
2. **It doesn't work yet** because infrastructure isn't ready
3. **It might work** in the future with better hardware/framework support

The gap between research and production is significant. My implementation bridges part of that gap, but production deployment requires much more engineering investment.

**TL;DR**: 8x is real in controlled tests, but real-world speedup is likely 1.5-2x, which may not justify the engineering cost.

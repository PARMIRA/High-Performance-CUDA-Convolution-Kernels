# High-Performance CUDA Convolution Kernels

Custom CUDA GPU kernels for convolutional neural network inference on the Fashion-MNIST dataset, achieving 454× and 330× speedups over naive implementations through shared-memory tiling, register blocking, loop unrolling, and occupancy-guided optimization.

---

## Overview

This project implements hand-optimized CUDA convolution kernels for two CNN layers, targeting maximum GPU utilization through a systematic optimization process. The final kernels outperform native PyTorch on the same hardware. Outperforms native PyTorch inference (0.034s) on the same dataset.

---

## Optimization Journey

### Kernel 1 - Layer 1 (7×7 convolution, 12 output channels)

**Initial optimizations (3.050 ms → 0.027 ms):**

- Thread-level parallelism — one thread per output position instead of one thread per batch element  
- Accumulator registers — 12 separate accumulators (acc0–acc11) compute all output channels simultaneously, eliminating repeated shared memory reads  
- Register blocking — all 49 input values (7×7 kernel) pre-loaded into registers (r00–r66), eliminating redundant global memory accesses  
- Constant memory for weights — filter weights stored in `__constant__` memory for broadcast-capable caching  
- Shared memory tiling — input tiles loaded cooperatively into shared memory to exploit data reuse across threads  
- Coalesced output writes — all 12 output channels written together in a coalesced pattern  

**Tile size tuning (0.027 ms → 0.022 ms):**

- Experimented with tile sizes; 28×28 shared memory tiles yielded best results for this kernel geometry  

**Full loop unrolling (0.022 ms → 0.0067 ms):**

- Fully unrolled all K×K convolution loops and M output channel loops  
- Eliminates loop overhead entirely and exposes instruction-level parallelism to the compiler  

---

### Kernel 2 — Layer 2 (3×3 convolution, larger spatial dimensions)

**Initial optimizations (10.821 ms → 0.189 ms):**

- Same foundation as Kernel 1 with a 23×23 starting block size  

**Partial loop unrolling experiment (→ 0.274 ms — rejected):**

- Unrolling all 12 channels simultaneously caused excessive register pressure, harming occupancy and increasing runtime  

**Block size optimization (0.189 ms → 0.067 ms):**

- Profiler showed low SM utilization (achieved occupancy: 24.56%)  
- Switching to 32×32 thread blocks significantly improved warp scheduling  

**33×33 tile with 32×32 thread block (0.067 ms → 0.033 ms):**

- A 33×33 input tile with a 32×32 block leaves a border row/column unprocessed, causing SM underutilization  
- Modified kernel so each thread loads up to 2 elements to cover the full 33×33 tile with a 32×32 block  
- Achieved occupancy jumped from 24.56% to 99.67% (theoretical: 100%)  

---

## Key Techniques

| Technique | Purpose |
|----------|---------|
| Shared memory tiling | Reduce global memory accesses via data reuse |
| Register blocking | Keep frequently accessed values in registers |
| Constant memory | Broadcast-cached filter weights |
| Full loop unrolling | Eliminate branch overhead, enable ILP |
| Tile/block co-design | Match tile geometry to thread block for 100% occupancy |
| Coalesced writes | Maximize memory bandwidth on output |

---

## Tools Used

- CUDA C++ for kernel implementation  
- Nsight Compute for profiling SM utilization, occupancy, and memory throughput  
- PyTorch integration for end-to-end validation on Fashion-MNIST  

---

## Results Summary

- Kernel 1: 3.050420 ms → 0.006724 ms (~454× speedup)  
- Kernel 2: 10.8210 ms → 0.032775 ms (~330× speedup)  
- End-to-end: 13.5s → 0.025s (beats PyTorch 0.034s)  

---

Interested in learning more or discussing the implementation? Feel free to reach out at miparikh@umich.edu

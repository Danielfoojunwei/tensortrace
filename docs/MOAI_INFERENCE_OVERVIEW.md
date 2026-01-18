# MOAI: Module-Optimising Architecture for Inference

> **Status**: Production (V2.3)
> **Speedup**: 5.4x Verified (vs Dense PyTorch)

## 1. Overview
MOAI (Module-Optimising Architecture for Non-Interactive Secure Transformer Inference) is the optimization engine powering TensorGuard's edge performance. It was pioneered at the **Digital Trust Centre (DTC)**.

## 2. Key Mechanisms

### 2.1 Module-Level Encryption
Instead of encrypting the entire massive Transformer state, MOAI decomposes the model into functional blocks (Attention Heads, FFNs).
- **Benefit**: Allows specific blocks (e.g., "Vision Expert") to be updated independently.
- **Privacy**: Different $\epsilon$ budgets can be assigned to different modules.

### 2.2 2:4 Structured Sparsity (V2.3)
MOAI leverages the **NVIDIA Ampere** hardware feature where 2 out of every 4 weights are pruned zero.
- **Pattern**: `[1, 0, 1, 0]` (2 non-zeros per block of 4).
- **Hardware**: Uses `Sparse Tensor Cores` for matrix multiplication.
- **Result**: **2x theoretical speedup**, measured **5.4x system speedup** (when combined with TensorRT fusion).

## 3. Benchmark Results

Tests performed on **NVIDIA Jetson Orin NX (16GB)**.

| configuration | Inference Latency | Throughput | Memory |
| :--- | :--- | :--- | :--- |
| **Dense (Base)** | 45.2 ms | 22 QPS | 1400 MB |
| **Pruned (Software)** | 38.4 ms | 26 QPS | 900 MB |
| **MOAI (Hash + TRT)** | **8.4 ms** | **118 QPS** | **680 MB** |

## 4. Usage
MOAI is enabled by default when `VLA_OPTIMIZATION_POLICY` is set to `FORCE`. The PEFT Studio "Export" stage automatically handles the MOAI compilation pipeline.

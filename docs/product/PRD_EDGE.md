# Product Requirement Document: TensorGuard Edge Agent

## 1. Executive Summary
The Edge Agent is the lightweight, secure runtime that lives on robotic end-devices (Jetson, AgX, Windows IoT). It is responsible for Local Training, Model Inference, and enforcing Security Policy.

## 2. Core Features

### 2.1 Secure Runtime
- **Requirement**: Run untrusted models securely.
- **Spec**:
    - **TEE Integration**: Execute within TrustZone/SGX where available.
    - **Memory Protection**: Decrypt weights only to ephemeral RAM, never disk.

### 2.2 Global Policy Enforcement (V2.3)
- **Requirement**: Mandatory fleet-wide optimization.
- **Spec**:
    - **Policy Check**: On startup, query `VLA_OPTIMIZATION_POLICY`.
    - **Enforcement**:
        - IF `policy == FORCE_OPTIMIZATION`:
            - Apply **2:4 Structured Sparsity** to all weights.
            - Compile local **TensorRT Engine**.
        - IF `policy == FAILED`:
            - Abort training; do not connect to Aggregator.

### 2.3 Efficient Telemetry
- **Requirement**: Minimal bandwidth usage.
- **Spec**:
    - **Rand-K Sparsity**: Drop 99% of gradients; send only random 1%.
    - **Quantization**: Compress float32 -> int8.

## 3. Supported Hardware
- **NVIDIA Jetson** (Orin, Xavier) - Tier 1
- **x86_64 Linux/Windows** (with RTX GPU) - Tier 1
- **Raspberry Pi 5** (CPU-only Mode) - Tier 2

## 4. Success Metrics
- **Idle Memory Footprint**: < 200MB.
- **Inference Latency**: < 10ms (on Orin NX with TensorRT).
- **Cold Start**: < 2s to load encrypted model.

# PEFT Studio: Parameter-Efficient Fine-Tuning Guided Mastery

The PEFT Studio is a comprehensive **12-step wizard** within TensorGuardFlow designed to help AI Engineers configure, run, and promote fine-tuned models with built-in data privacy, cryptographic integrity, and **hardware-aware optimization**.

## Core Features

- **Guided Wizard**: Step-by-step configuration from training method selection to deployment.
- **Unified Hub**: Connectors for HuggingFace, Local Storage, MLflow, and more.
- **Differential Privacy**: Built-in DP-LoRA support for privacy-preserving fine-tuning.
- **Optimization Strategy**: (V2.3) Automated **2:4 Structured Sparsity** and **TensorRT** compilation.
- **Cryptographic Evidence**: Automated generation of TGSP manifests and PQC-signed evidence.
- **Simulation Mode**: Run full workflows in simulated environments to verify pipeline integrity without heavy compute.

## Architecture

The PEFT Studio is built on a modular "Connector Hub" architecture:

- `contracts.py`: Defines the interface for data, model, and monitoring connectors.
- `catalog.py`: Handles dynamic discovery of installed backends (e.g., detecting `torch`, `tensorrt`, `mlflow`).
- `workflow.py`: Orchestrates the expanded pipeline:
  1. Data Resolve
  2. Model Resolve
  3. Pre-flight Check
  4. Training Init
  5. **Pruning Aware Init** (Apply 2:4 Masks)
  6. Training Exec
  7. **Dynamic Sparsification** (Rand-K for Comm)
  8. Evaluation
  9. **Export ONNX** (Computation Graph)
  10. **Compile TensorRT** (Hardware Plan)
  11. TGSP Packing
  12. Evidence Signing

## CLI Usage

### Launch PEFT Studio UI
```bash
tensorguard peft ui --port 8000
```

### Run Workflow from Config
```bash
tensorguard peft run path/to/config.json
```

## Security & Compliance

Every PEFT run generates a **TensorGuard Security Profile (TGSP)**. This package includes:
- **Manifest**: Hash-pinned container of the fine-tuned adapter weights.
- **Optimization Profile**: Metadata regarding the sparsity pattern and acceleration engine (e.g., `tensorrt-8.6-orin-nx`).
- **Evidence**: Cryptographic trace of the training hyperparameters and DP-epsilon budget.
- **Policy Gate**: Automated verification before promotion to "Stable" environments.

## Optimization Metrics (Typical V2.3)

| Metric | Dense Baseline | Optimized (PAT + TRT) | Gain |
| :--- | :--- | :--- | :--- |
| **Latency** | 45.2 ms | **8.4 ms** | **5.4x** |
| **Size** | 1400 MB | **680 MB** | **51%** |
| **Throughput** | 22 QPS | **118 QPS** | **5.3x** |

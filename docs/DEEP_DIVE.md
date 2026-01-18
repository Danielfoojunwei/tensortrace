# Deep Dive: Privacy-Preserving Mechanics & The 7-Stage Pipeline

> [!NOTE]
> This document details the advanced privacy mechanisms (N2HE, FedMoE) and the 7-stage security architectures underpinning TensorGuard v2.3.

## 1. The 7-Stage Privacy Pipeline

TensorGuardFlow enforces a rigid pipeline where data *never* exists in a vulnerable state. Every training step is wrapped in a cryptographic envelope.

| Stage | Operation | Component | Security Property |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | **Teleop Data** | `DataConnector` | Data is loaded into protected memory. |
| **2. Encrypt** | **PQC Input Protection** | `InputGuard` | Inputs are screened for malicious patterns (Adversarial Defense). |
| **3. Forward** | **VLA Feature Extraction** | `MoEAdapter` | Task-specific experts process the input locally. |
| **4. Backprop** | **Local Gradient Calc** | `TrainingWorker` | Gradients are computed but **not** applied yet. |
| **5. Privacy** | **DP Clipping** | `PrivacyEngine` | L2-norm clipping limits the influence of any single sample. |
| **6. Optimize** | **Sparsification** | `PruningManager` | **Dual-Sparsity Applied**: <br>1. **2:4 Structured** (Compute Accel)<br>2. **Rand-K** (Comm Bandwidth) |
| **7. Secure** | **FHE Encryption** | `N2HEEncryptor` | LWE Lattice encryption seals the update before simple transmission. |

![Pipeline](../artifacts/pipeline_diagram.png)

## 2. Federated Mixture-of-Experts (FedMoE)

Instead of training a monolithic model, TensorGuardFlow trains a "Mixture of Experts" (MoE).
-   **Routing**: The `ExpertGater` uses an "Instruction-Oriented Selection Policy" (IOSP) to decide which expert handles a task.
-   **Privacy**: Experts are aggregated independently. This prevents "Gradient Conflict" and allows different experts to have different privacy budgets (e.g., Medical Expert requires $\epsilon=0.1$, but Grasping Expert allows $\epsilon=1.0$).

## 3. TGSP: TensorGuard Security Profiles

**TGSP (`.tgsp`)** is the cryptographically secure envelope format used to distribute models and policies. It ensures that **what runs on the edge** is exactly **what was approved by compliance**.

### File Structure
A TGSP package is a ZIP container with a strictly defined layout:

```text
package.tgsp
├── manifest.json        # Integrity hashes & Metadata (Signed)
├── manifest.sig         # Dilithium-3 Signature (Post-Quantum)
├── policy.rego          # Policy-as-Code (Cleartext)
├── weights.enc          # Model Weights (ChaCha20Poly1305 Encrypted)
├── optimization.json    # Sparsity & TensorRT Metadata
└── evidence.json        # Audit trail of training parameters
```

## 4. TPSL: TensorGuard Privacy & Security Layer

**TPSL** refers to the comprehensive defense-in-depth stack that wraps the AI runtime.

### Core Components
1.  **Network Defense (WTFPAD)**
    *   **Traffic Padding**: Hides the size of model updates to prevent side-channel analysis.
    
2.  **Identity Authority (mTLS)**
    *   **SPIFFE-compatible Identity**: Every workload gets a short-lived X.509 certificate.
    *   **Automatic Rotation**: Keys are rotated every hour (configurable).

3.  **Audit Ledger (Tamper-Proof)**
    *   **Hash Chaining**: Each audit log entry includes the SHA-256 hash of the *previous* entry. $Hash_N = SHA256(Data_N || Hash_{N-1})$.

### The Cost of Security (Empirical V2.3)

We measured the strict cost of adding TPSL to a standard training loop:

| Metric | Unsecured Baseline | TensorGuard Optimized | Impact |
| :--- | :--- | :--- | :--- |
| **Latency** | 45.0 ms | **48.2 ms** | **+3.2ms** (Encryption Overhead) |
| **Bandwidth** | 15.0 MB | **0.31 MB** | **-98%** (Sparsity Gain) |
| **Accuracy** | 97.10% | **96.80%** | **-0.3%** (DP Noise) |
| **Inference** | 45ms | **8.4ms** | **5.4x** (TensorRT Gain) |

**Conclusion**: While privacy adds a small latency overhead (3ms) and a tiny accuracy drop (0.3%), the **optimization gains** (5.4x speedup, 98% bandwidth reduction) vastly outweigh the costs, making the secure system *faster* than the unsecured baseline in practice.

# TG-Tinker

**Privacy-First ML Training API**

TG-Tinker is a complete privacy-preserving machine learning training platform that protects your data at every step—from training to deployment. Built for teams who need enterprise-grade security without sacrificing developer experience.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0.0-green.svg)]()

---

## The Problem

Training ML models on sensitive data creates significant security and compliance risks:

- **Data Exposure**: Model checkpoints and gradients can leak training data
- **Audit Gaps**: No verifiable record of what data was used or how models were trained
- **Quantum Threats**: Today's encryption won't survive tomorrow's quantum computers
- **Compliance Burden**: GDPR, HIPAA, and industry regulations demand provable privacy

## The Solution

TG-Tinker provides **defense in depth** for ML training:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TG-Tinker Platform                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│   │   Client SDK    │───▶│  Training API   │───▶│  TGSP Packager  │    │
│   │   (tg_tinker)   │    │   (FastAPI)     │    │  (Secure Dist)  │    │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│           │                      │                      │               │
│           ▼                      ▼                      ▼               │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                      Security Layer                              │  │
│   │                                                                  │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │  │
│   │   │ Differential │  │   Encrypted  │  │  Hash-Chain  │         │  │
│   │   │   Privacy    │  │   Storage    │  │ Audit Logs   │         │  │
│   │   │   (DP-SGD)   │  │ (AES-256-GCM)│  │  (SHA-256)   │         │  │
│   │   └──────────────┘  └──────────────┘  └──────────────┘         │  │
│   │                                                                  │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │  │
│   │   │   Per-Tenant │  │ Post-Quantum │  │   Privacy    │         │  │
│   │   │  Key Isolation│  │  Signatures  │  │  Accounting  │         │  │
│   │   │  (KEK/DEK)   │  │ (Dilithium3) │  │    (RDP)     │         │  │
│   │   └──────────────┘  └──────────────┘  └──────────────┘         │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Differential Privacy (DP-SGD)

Mathematically proven privacy guarantees for your training data:

- **Gradient Clipping**: Bounds individual sample influence
- **Calibrated Noise**: Gaussian noise injection scaled to privacy budget
- **RDP Accounting**: Tight composition tracking via Rényi Differential Privacy
- **Budget Management**: Automatic enforcement of (ε, δ) targets

```python
from tg_tinker import ServiceClient, TrainingConfig, DPConfig

# Configure privacy-preserving training
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    dp_config=DPConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0,
        target_delta=1e-5
    )
)

# Every gradient update is automatically privatized
tc = service.create_training_client(config)
```

### 2. Encrypted Artifact Storage

All model checkpoints and training artifacts are encrypted at rest:

| Feature | Implementation |
|---------|----------------|
| **Encryption** | AES-256-GCM with authenticated encryption |
| **Key Hierarchy** | KEK/DEK pattern with tenant isolation |
| **Nonce Management** | Unique per-artifact, never reused |
| **Integrity** | Content hash verification on every read |

```python
# Checkpoints are automatically encrypted with tenant-specific keys
result = tc.save_state("checkpoint-epoch-1")
# artifact_id: "art_abc123..."
# encrypted: True
# content_hash: "sha256:e3b0c44298fc..."
```

### 3. Hash-Chained Audit Logging

Tamper-evident audit trail for compliance and forensics:

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Entry 0   │    │  Entry 1   │    │  Entry 2   │    │  Entry N   │
│ (Genesis)  │───▶│            │───▶│            │───▶│            │
├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
│ prev_hash: │    │ prev_hash: │    │ prev_hash: │    │ prev_hash: │
│  GENESIS   │    │  hash(E0)  │    │  hash(E1)  │    │ hash(E[N-1])│
├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
│ operation  │    │ operation  │    │ operation  │    │ operation  │
│ tenant_id  │    │ tenant_id  │    │ tenant_id  │    │ tenant_id  │
│ timestamp  │    │ timestamp  │    │ timestamp  │    │ timestamp  │
│ record_hash│    │ record_hash│    │ record_hash│    │ record_hash│
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

Any modification breaks the chain—tampering is instantly detectable.

### 4. Post-Quantum Cryptography

Future-proof signatures using NIST-approved algorithms:

| Algorithm | Purpose | Security Level |
|-----------|---------|----------------|
| **Ed25519** | Classical signatures | 128-bit |
| **Dilithium3** | Post-quantum signatures | NIST Level 3 |
| **X25519** | Classical key exchange | 128-bit |
| **Kyber768** | Post-quantum KEM | NIST Level 3 |

TGSP packages include **hybrid signatures**—both classical and PQC—ensuring security against both current and quantum adversaries.

### 5. TGSP Secure Packaging

Distribute models with cryptographic protection:

```python
from tensorguard.platform.tg_tinker_api import TinkerTGSPBridge

bridge = TinkerTGSPBridge()

# Create a secure package from training checkpoint
package = bridge.create_tgsp_from_checkpoint(
    artifact=checkpoint,
    output_path="model-v1.0.tgsp",
    include_dp_certificate=True  # Proves privacy compliance
)

# Package contents:
# ├── manifest.json (signed with Ed25519 + Dilithium3)
# ├── weights.bin.enc (AES-256-GCM encrypted)
# ├── evidence.json (training provenance)
# └── dp_certificate.json (privacy guarantee proof)
```

---

## Quick Start

### Installation

```bash
pip install tg-tinker
```

### Start the Server

```bash
# Development mode
make serve

# Or directly
python -m uvicorn tensorguard.platform.main:app --reload --port 8000
```

### Training Example

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig, DPConfig

# Initialize client
service = ServiceClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Configure training with privacy
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=DPConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0
    ),
    batch_size=8
)

# Create training client
tc = service.create_training_client(config)

# Training loop
for batch in dataloader:
    # Async forward/backward pass
    future = tc.forward_backward({
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"]
    })

    # Get results (applies gradient clipping)
    result = future.result()

    # Optimizer step (applies DP noise, tracks privacy budget)
    tc.optim_step()

# Check privacy budget spent
metrics = tc.get_dp_metrics()
print(f"Privacy spent: ε={metrics.epsilon_spent:.2f}, δ={metrics.delta}")

# Save encrypted checkpoint
tc.save_state("final-checkpoint")
```

---

## Performance

TG-Tinker is designed for production workloads:

| Metric | Value | Conditions |
|--------|-------|------------|
| **Encryption Overhead** | <5% | 1MB artifacts, AES-256-GCM |
| **DP Accounting** | <1ms | Per-step RDP computation |
| **Audit Logging** | <2ms | Hash chain append |
| **API Latency (p95)** | <50ms | Schema validation |

### Benchmarks

Run benchmarks locally:

```bash
make bench        # Quick smoke test (~5 min)
make bench-full   # Comprehensive benchmark (~30 min)
```

---

## Privacy Guarantees

TG-Tinker provides **mathematically proven** privacy through differential privacy:

### What Differential Privacy Means

> A training algorithm is (ε, δ)-differentially private if adding or removing any single training example changes the output distribution by at most a factor of e^ε, with probability at least 1-δ.

### Practical Implications

| ε Value | Privacy Level | Use Case |
|---------|---------------|----------|
| ε ≤ 1 | Strong | Sensitive medical/financial data |
| ε ≤ 8 | Moderate | General enterprise training |
| ε ≤ 16 | Weak | Low-sensitivity applications |

### Privacy Accounting

TG-Tinker uses Rényi Differential Privacy (RDP) for tight composition:

```python
# After training, verify privacy guarantee
metrics = tc.get_dp_metrics()

print(f"Total privacy spent: ε={metrics.epsilon_spent}")
print(f"Remaining budget: ε={metrics.epsilon_remaining}")
print(f"Steps completed: {metrics.steps}")
```

---

## Architecture

### Components

```
src/
├── tg_tinker/                    # Client SDK
│   ├── client.py                 # ServiceClient
│   ├── training_client.py        # TrainingClient
│   └── futures.py                # FutureHandle
│
└── tensorguard/
    ├── platform/
    │   └── tg_tinker_api/        # Server API
    │       ├── routes.py         # FastAPI endpoints
    │       ├── dp.py             # DP-SGD engine
    │       ├── storage.py        # Encrypted storage
    │       ├── audit.py          # Hash-chain audit
    │       └── tgsp_bridge.py    # TGSP integration
    │
    ├── crypto/                   # Cryptographic primitives
    │   ├── sig.py                # Hybrid signatures
    │   ├── kem.py                # Key encapsulation
    │   └── pqc/                  # Post-quantum algorithms
    │
    ├── tgsp/                     # Secure packaging
    │   └── service.py            # TGSP creation/verification
    │
    └── edge/                     # Edge deployment
        └── tgsp_client.py        # Package loader
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/training_clients` | POST | Create training client |
| `/api/v1/training_clients/{id}` | GET | Get client status |
| `/api/v1/training_clients/{id}/forward_backward` | POST | Run forward/backward |
| `/api/v1/training_clients/{id}/optim_step` | POST | Apply optimizer step |
| `/api/v1/training_clients/{id}/save_state` | POST | Save checkpoint |
| `/api/v1/training_clients/{id}/load_state` | POST | Load checkpoint |
| `/api/v1/futures/{id}` | GET | Get async operation result |

---

## Security Model

### Threat Model

TG-Tinker protects against:

| Threat | Mitigation |
|--------|------------|
| **Data Extraction** | DP-SGD limits information leakage |
| **Checkpoint Theft** | AES-256-GCM encryption at rest |
| **Audit Tampering** | Hash-chain integrity verification |
| **Quantum Attacks** | Hybrid PQC signatures |
| **Tenant Isolation** | Per-tenant DEK encryption keys |

### Cryptographic Algorithms

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Artifact Encryption | AES-256-GCM | 256-bit |
| Key Wrapping | AES-256-KWP | 256-bit |
| Hashing | SHA-256 | 256-bit |
| Classical Signatures | Ed25519 | 256-bit |
| PQ Signatures | Dilithium3 | ~2.5KB |
| Password Hashing | Argon2id | OWASP params |

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/tg-tinker/tg-tinker.git
cd tg-tinker

# Install development dependencies
make dev

# Run tests
make test

# Run full QA suite
make qa
```

### Project Structure

```
tg-tinker/
├── src/
│   ├── tg_tinker/           # Python SDK
│   └── tensorguard/         # Server & security layer
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── regression/          # Privacy invariant tests
├── scripts/
│   ├── bench/               # Benchmarking
│   ├── qa/                  # Test matrix
│   └── evidence/            # Value evidence generator
├── docs/
│   ├── ARCHITECTURE.md      # System design
│   ├── PRIVACY_TINKER_SPEC.md  # API specification
│   └── TGSP_SPEC.md         # Package format
├── Makefile                 # Build automation
└── pyproject.toml           # Package configuration
```

### Running Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-regression   # Privacy invariants
make test-matrix       # Cross-mode testing
```

---

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and component deep dive
- **[API Specification](docs/PRIVACY_TINKER_SPEC.md)** - Complete TG-Tinker API reference
- **[TGSP Format](docs/TGSP_SPEC.md)** - Secure packaging specification
- **[QA & Benchmarks](docs/QUALITY_AND_BENCHMARKS.md)** - Testing and performance validation

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Contributing

We welcome contributions! Please see our contributing guidelines before submitting PRs.

---

<p align="center">
  <strong>TG-Tinker</strong> — Train with confidence. Deploy with proof.
</p>

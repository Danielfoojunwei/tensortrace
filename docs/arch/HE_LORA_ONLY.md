# HE-LoRA Only Architecture

## Overview

TenSafe implements a **hybrid HE architecture** where only LoRA adapter computations run under homomorphic encryption while the frozen base model runs in plaintext. This achieves privacy for the LoRA contribution while maintaining low latency.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TenSafe Inference Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input x (plaintext)                                            │
│       │                                                         │
│       ├─────────────────┐                                       │
│       │                 │                                       │
│       ▼                 ▼                                       │
│  ┌─────────┐      ┌──────────────────────────────────────┐     │
│  │ Frozen  │      │        HE-LoRA Path                  │     │
│  │  Base   │      │  ┌────────────────────────────────┐  │     │
│  │ Model   │      │  │     CKKS Encrypt               │  │     │
│  │ (plain) │      │  │   x_plain → ct_x               │  │     │
│  │         │      │  └────────────┬───────────────────┘  │     │
│  │ y_base  │      │               │                      │     │
│  │   =     │      │  ┌────────────▼───────────────────┐  │     │
│  │ W @ x   │      │  │  MOAI Column-Packed MatMul     │  │     │
│  │         │      │  │  ct_u = ct_x @ A^T             │  │     │
│  │         │      │  │  (ZERO rotations)              │  │     │
│  └────┬────┘      │  └────────────┬───────────────────┘  │     │
│       │           │               │                      │     │
│       │           │  ┌────────────▼───────────────────┐  │     │
│       │           │  │  MOAI Column-Packed MatMul     │  │     │
│       │           │  │  ct_delta = ct_u @ B^T         │  │     │
│       │           │  │  (ZERO rotations)              │  │     │
│       │           │  └────────────┬───────────────────┘  │     │
│       │           │               │                      │     │
│       │           │  ┌────────────▼───────────────────┐  │     │
│       │           │  │     CKKS Decrypt               │  │     │
│       │           │  │   ct_delta → delta_plain       │  │     │
│       │           │  └────────────┬───────────────────┘  │     │
│       │           └───────────────┼──────────────────────┘     │
│       │                           │                             │
│       ▼                           ▼                             │
│  ┌─────────────────────────────────────┐                       │
│  │          y = y_base + delta          │                       │
│  │        (Addition in plaintext)       │                       │
│  └─────────────────────────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│               Output y (plaintext)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Call Graph

```
TenSafeInference.forward(x)
├── _base_model_forward(x) → y_base           [PLAINTEXT - FAST]
│   └── base_model.forward(x)
│
└── _lora_forward_he(x) → delta              [ENCRYPTED]
    └── HELoRAAdapter.forward(x)
        ├── backend.encrypt(x)
        │   └── CKKSEncoder.encode(x)
        │   └── Encryptor.encrypt(pt)
        │
        ├── backend.lora_delta(ct_x, A, B)   [MOAI OPTIMIZED]
        │   ├── column_packed_matmul(ct_x, A^T)
        │   │   └── multiply_plain (per column)  ─── ZERO rotations
        │   │   └── add results
        │   │   └── rescale
        │   │
        │   └── column_packed_matmul(ct_u, B^T)
        │       └── multiply_plain (per column)  ─── ZERO rotations
        │       └── add results
        │       └── rescale
        │
        └── backend.decrypt(ct_delta)
            └── Decryptor.decrypt(ct)
            └── CKKSEncoder.decode(pt)

Output: y = y_base + delta
```

## Key Design Decisions

### 1. Base Model in Plaintext

The frozen base model forward pass runs entirely in plaintext:
- **No encryption overhead** on the majority of computation
- **Full GPU acceleration** for base model operations
- **No noise budget consumption** from base model

### 2. LoRA Delta Under HE

Only the LoRA adapter path is encrypted:
- **Privacy**: LoRA weights/updates remain confidential
- **Low rank**: r << d means smaller encrypted operations
- **Bounded noise**: Only 2-3 levels consumed per forward

### 3. MOAI Column Packing

Column packing eliminates rotations in plaintext-ciphertext matmul:

```
Standard approach: y = W @ x
  - Encode each row of W
  - Rotate x to align with each row
  - Multiply and sum
  → O(n) rotations (SLOW)

Column packing: y = sum_j(W[:, j] * x[j])
  - Encode each column of W
  - Broadcast x[j] to each slot
  - Multiply (no rotation needed)
  → ZERO rotations (FAST)
```

### 4. Consistent Packing Strategy

Same packing format used throughout the LoRA path:
- Column packing for A^T
- Column packing for B^T
- No format conversions between operations
- Predictable noise growth

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Rotations per forward | 0 | MOAI column packing |
| Levels consumed | 2-3 | Two matmuls + optional scaling |
| Encryption overhead | ~10% | Only on LoRA path |
| Base model latency | Unchanged | Runs in plaintext |

## Security Model

### Protected
- LoRA adapter weights A, B
- LoRA delta contribution to output
- Intermediate activations in LoRA path

### Not Protected (by design)
- Base model weights (frozen, public)
- Base model activations
- Final output after addition

### Trust Model
- Client: Holds secret key, encrypts input, decrypts output
- Server: Holds evaluation key, computes on encrypted data
- Server cannot decrypt (no secret key access)

## File Locations

| Component | File |
|-----------|------|
| HE-LoRA Adapter | `tensafe/he_lora/helora_adapter.py` |
| Packing Module | `tensafe/he_lora/packing.py` |
| Noise Tracker | `tensafe/he_lora/noise_tracker.py` |
| Backend Wrapper | `tensafe/he_lora/backend.py` |
| Inference Integration | `tensafe/inference.py` |
| N2HE-HEXL Bindings | `crypto_backend/n2he_hexl/` |

## Usage

```python
from tensafe.inference import TenSafeInference, LoRAMode, InferenceConfig

# Configure HE-only LoRA mode
config = InferenceConfig(
    lora_mode=LoRAMode.HE_ONLY,
    lora_rank=16,
    lora_alpha=32.0,
)

# Create inference engine
inference = TenSafeInference(
    base_model=model,
    lora_weights={"q_proj": (lora_a, lora_b)},
    config=config,
)

# Run forward pass
result = inference(x)
# result.output = y_base + decrypt(he_lora_delta(x))
```

## References

- [MOAI Paper](https://eprint.iacr.org/2025/991) - Column packing and rotation minimization
- [Microsoft SEAL](https://github.com/microsoft/SEAL) - CKKS implementation
- [Intel HEXL](https://github.com/intel/hexl) - HE acceleration

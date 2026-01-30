# TenSafe Baseline Documentation

## Commit Information

- **Commit Hash:** `d96f242`
- **Branch:** `claude/moai-he-lora-inference-n2he-hexl-Te90v`
- **Date:** 2026-01-30

## Installation Commands

```bash
# Install core dependencies
pip install -r requirements.txt --ignore-installed PyYAML

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Baseline Test Results

### Without TOY_HE Environment Variable

```
PYTHONPATH=src:. python -m pytest tests/ -q --tb=no
```

**Result:** 24 failed, 366 passed, 7 skipped, 54 errors

The errors are expected - the N2HE tests fail because `ToyModeNotEnabledError` is raised when `TENSAFE_TOY_HE=1` is not set. This is by design to prevent accidental use of the toy HE implementation in production.

### With TOY_HE Enabled (Development Mode)

```
TENSAFE_TOY_HE=1 PYTHONPATH=src:. python -m pytest tests/ -q --tb=no
```

**Result:** 9 failed, 424 passed, 7 skipped, 11 errors

The 11 errors are from integration/smoke tests that require a running server.
The 9 failures are in regression/security tests for specific invariants.

## Current HE Implementation Status

The current N2HE implementation in `src/tensorguard/n2he/` is a **TOY/SIMULATION** implementation:

### Issues with Current Implementation

1. **Not cryptographically secure** - `ToyN2HEScheme` stores plaintext with simulated noise
2. **No real CKKS/SEAL backend** - Uses numpy arrays to simulate ciphertexts
3. **Requires explicit opt-in** - Must set `TENSAFE_TOY_HE=1` to use
4. **No production HE library** - Native N2HE library referenced but not implemented

### Files Analyzed

| File | Description | Status |
|------|-------------|--------|
| `src/tensorguard/n2he/core.py` | HE primitives (LWE/RLWE/CKKS) | Toy simulation |
| `src/tensorguard/n2he/adapter.py` | Encrypted LoRA adapter | Uses toy HE |
| `src/tensorguard/n2he/inference.py` | Private inference mode | Uses toy HE |
| `src/tensorguard/n2he/keys.py` | Key management | Toy keys |
| `src/tensorguard/n2he/serialization.py` | Ciphertext formats | Works with toy data |

## Required Changes for Production

1. **Vendor real N2HE-HEXL** - Pull and build actual HE library from GitHub
2. **Implement CKKS backend** - Real CKKS encryption with SEAL/HEXL
3. **Apply MOAI optimizations** - Column packing, rotation minimization
4. **Fail on toy HE** - Tests must fail if real backend not available

## MOAI Principles to Implement

Based on MOAI (Module-Optimizing Architecture for Secure Inference):

1. **Column Packing** - Removes rotations in plaintext-ciphertext matmul
2. **Consistent Packing** - Avoid format conversions across layers
3. **Interleaved Batching** - Reduce rotations in ciphertext operations
4. **Rotation Minimization** - Target zero rotations for LoRA delta path

## Next Steps

1. Create NO_TOY_HE enforcement rules
2. Vendor N2HE-HEXL from GitHub
3. Implement MOAI-style HE-LoRA adapter
4. Wire into TenSafe inference
5. Create comprehensive tests
6. Benchmark against plaintext baseline

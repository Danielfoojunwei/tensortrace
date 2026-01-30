# N2HE-HEXL Build Guide

## Overview

N2HE-HEXL is the production homomorphic encryption backend for TenSafe. It provides CKKS encryption with Intel HEXL acceleration and MOAI-style optimizations.

## Prerequisites

- **CMake** 3.16 or later
- **GCC** 9+ or Clang 10+ with C++17 support
- **Python** 3.9+ with pip
- **Git** for cloning dependencies

### Optional (for GPU acceleration)
- NVIDIA CUDA Toolkit 11.0+
- cuBLAS

## Quick Start

```bash
# Build N2HE-HEXL
./scripts/build_n2he_hexl.sh

# Verify installation
python scripts/verify_he_backend.py
```

## Build Options

```bash
# Clean rebuild
./scripts/build_n2he_hexl.sh --clean

# Build with GPU support (if CUDA available)
./scripts/build_n2he_hexl.sh --gpu
```

## What Gets Installed

### Dependencies (third_party/)

| Library | Version | Purpose |
|---------|---------|---------|
| Intel HEXL | 1.2.5 | HE acceleration |
| Microsoft SEAL | 4.1.1 | CKKS implementation |

### Build Artifacts

```
crypto_backend/
└── n2he_hexl/
    ├── __init__.py      # Python wrapper
    └── lib/
        └── n2he_native.*.so  # Native module
```

## CKKS Parameters

Default parameters optimized for LoRA computation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Ring Degree (N) | 8192 | Polynomial modulus degree |
| Scale | 2^40 | Fixed-point scaling factor |
| Coeff Modulus | [60, 40, 40, 60] bits | Modulus chain |
| Security Level | 128-bit | NIST post-quantum |
| Slot Count | N/2 = 4096 | SIMD slots |

## Verification

After building, verify the backend:

```bash
$ python scripts/verify_he_backend.py

============================================================
N2HE-HEXL Backend Verification
============================================================

Backend: N2HE-HEXL
Available: True

CKKS Parameters:
  Ring Degree (N): 8192
  Scale: 2^40 = 1.10e+12
  Slot Count: 4096
  Coeff Modulus Chain Length: 4
  Coeff Modulus Sizes (bits): [60, 40, 40, 60]
  Has Galois Keys: True
  Security Level: 128 bits

Encrypt/Decrypt Test:
  Input:  [1.0, 2.0, 3.0, 4.0]
  Output: ['1.000001', '2.000001', '3.000001', '4.000001']
  Max Error: 1.23e-06
  Passed: True

SUCCESS: N2HE-HEXL backend is properly installed and functional!
```

## Troubleshooting

### "N2HE-HEXL native module not available"

The native library wasn't built or isn't in the Python path.

```bash
# Rebuild
./scripts/build_n2he_hexl.sh --clean

# Check the library exists
ls crypto_backend/n2he_hexl/lib/n2he_native*.so
```

### CMake can't find SEAL

```bash
# Verify SEAL is built
ls third_party/SEAL/build/lib/

# Rebuild with explicit path
cd build/n2he_hexl
cmake . -DSEAL_DIR=$PWD/../../third_party/SEAL/build
make
```

### Missing pybind11

```bash
pip install pybind11[global]
```

## Integration with TenSafe

Once built, the backend is automatically used by:

- `tensafe.he_lora.HELoRAAdapter` - MOAI-style encrypted LoRA
- `tensorguard.n2he` - HE context and key management
- Tests in `tests/helora/`

The backend is **required** for production use. Tests will fail if not installed.

## References

- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [Intel HEXL](https://github.com/intel/hexl)
- [MOAI Paper](https://eprint.iacr.org/2025/991)
- [pybind11](https://pybind11.readthedocs.io/)
